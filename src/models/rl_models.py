import torch.nn as nn


class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_budget_options):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.image_head = nn.Linear(hidden_dim, 1)

        # Dynamic query-size head:
        # one logit per possible budget option, e.g. [250, 500, 750, 1000]
        self.budget_head = nn.Linear(hidden_dim, num_budget_options)

    def forward(self, x, global_state=None):
        h = self.encoder(x)
        image_logits = self.image_head(h).squeeze(-1)  # [N]

        budget_logits = None
        if global_state is not None:
            g = self.encoder(global_state.unsqueeze(0))      # [1, H]
            budget_logits = self.budget_head(g).squeeze(0)   # [num_budget_options]

        return image_logits, budget_logits