"""Shared MacBERT, metadata gate, detached posterior relations and ordinal loss."""

import torch
from torch import nn
from torch.nn import functional as F

TASKS = ("post_type", "sentiment", "demand", "priority")
CLASS_COUNTS = (9, 6, 8, 4)


class AttentionPool(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1, bias=False)
        )

    def forward(self, states, mask):
        scores = self.score(states).squeeze(-1).masked_fill(~mask, -torch.inf)
        return (scores.softmax(-1).unsqueeze(-1) * states).sum(1)


def ordinal_probabilities(logits):
    cumulative = logits.sigmoid()
    return torch.cat(
        (
            1 - cumulative[:, :1],
            cumulative[:, :-1] - cumulative[:, 1:],
            cumulative[:, -1:],
        ),
        dim=1,
    )


class GovRelMTL(nn.Module):
    def __init__(self, encoder, category_sizes):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.pool = AttentionPool(hidden)
        self.refine = nn.MultiheadAttention(hidden, 8, batch_first=True)
        self.refine_norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(0.1)
        self.text_projection = nn.Linear(2 * hidden, 256)
        self.categories = nn.ModuleList(
            nn.Embedding(size, 8) for size in category_sizes
        )
        self.metadata_mlp = nn.Sequential(
            nn.Linear(5 + 3 + 9 * len(category_sizes), 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.metadata_gate = nn.Linear(256 + 128, 256)
        self.metadata_projection = nn.Linear(128, 256, bias=False)
        self.fusion_norm = nn.LayerNorm(256)
        self.post_type = nn.Linear(hidden, 9)
        self.sentiment = nn.Linear(256 + 64, 6)
        self.demand = nn.Linear(256 + 64, 8)
        self.post_labels = nn.Embedding(9, 64)
        self.sentiment_labels = nn.Embedding(6, 64)
        self.demand_labels = nn.Embedding(8, 64)
        self.priority_score = nn.Linear(256 + 3 * 64, 1, bias=False)
        self.threshold_start = nn.Parameter(torch.zeros(1))
        self.threshold_gaps = nn.Parameter(torch.zeros(2))
        self.log_variance = nn.Parameter(torch.zeros(4))

    def encode_posts(self, inputs, mask):
        states = self.encoder(**inputs).last_hidden_state
        return self.post_type(self.pool(states, mask))

    def fuse(self, text, numeric, categories):
        available = categories.ne(0)
        if self.training:
            available = available & (
                torch.rand(categories.shape, device=categories.device) >= 0.2
            )
        categories = categories.masked_fill(~available, 0)
        embedded = [layer(categories[:, i]) for i, layer in enumerate(self.categories)]
        required = numeric.new_ones((numeric.shape[0], 3))
        metadata = self.metadata_mlp(
            torch.cat(
                [numeric, *embedded, required, available.to(numeric.dtype)], dim=1
            )
        )
        gate = self.metadata_gate(torch.cat((text, metadata), dim=1)).sigmoid()
        return self.fusion_norm(text + gate * self.metadata_projection(metadata))

    def forward_comments(
        self,
        post_probabilities,
        pair,
        post_mask,
        comment_mask,
        numeric,
        categories,
        post_index,
    ):
        states = self.encoder(**pair).last_hidden_state
        attention, _ = self.refine(
            states, states, states, key_padding_mask=~post_mask, need_weights=False
        )
        refined = self.refine_norm(states + self.dropout(attention))
        text = F.gelu(
            self.text_projection(
                torch.cat((states[:, 0], self.pool(refined, comment_mask)), dim=1)
            )
        )
        fused = self.fuse(text, numeric, categories)
        post = post_probabilities.detach()[post_index] @ self.post_labels.weight
        upstream = torch.cat((fused, post), dim=1)
        sentiment, demand = self.sentiment(upstream), self.demand(upstream)
        sentiment_relation = (
            sentiment.softmax(1).detach() @ self.sentiment_labels.weight
        )
        demand_relation = demand.softmax(1).detach() @ self.demand_labels.weight
        priority_features = torch.cat(
            (fused, post, sentiment_relation, demand_relation), dim=1
        )
        thresholds = self.threshold_start + torch.cat(
            (
                self.threshold_start.new_zeros(1),
                F.softplus(self.threshold_gaps).cumsum(0),
            )
        )
        return sentiment, demand, self.priority_score(priority_features) - thresholds


def effective_number(counts, beta):
    if (counts == 0).any():
        raise ValueError(
            "A task class or ordinal threshold outcome has no training examples"
        )
    weights = (1 - beta) / (1 - beta ** counts.to(torch.float64))
    return (weights / weights.mean(dim=-1, keepdim=True)).float()


def training_weights(posts, beta, device):
    comments = [row for post in posts for row in post["comments"]]
    sources = (posts, comments, comments)
    weights = [
        effective_number(
            torch.bincount(torch.tensor([row[task] for row in rows]), minlength=size),
            beta,
        )
        for rows, task, size in zip(sources, TASKS, CLASS_COUNTS)
    ]
    priority = torch.tensor([row["priority"] for row in comments])
    counts = torch.stack(
        [torch.bincount((priority > k).long(), minlength=2) for k in range(3)]
    )
    return tuple(
        weight.to(device) for weight in (*weights, effective_number(counts, beta))
    )


def comment_loss_sums(logits, labels, weights):
    sentiment, demand, priority = logits
    ys, yd, yr = labels
    targets = (yr[:, None] > torch.arange(3, device=yr.device)).long()
    outcome_weights = weights[2].gather(1, targets.T).T
    ordinal = F.binary_cross_entropy_with_logits(
        priority, targets.to(priority.dtype), reduction="none"
    )
    return torch.stack(
        (
            F.cross_entropy(sentiment, ys, weight=weights[0], reduction="sum"),
            F.cross_entropy(demand, yd, weight=weights[1], reduction="sum"),
            (ordinal * outcome_weights).sum() / 3,
        )
    )
