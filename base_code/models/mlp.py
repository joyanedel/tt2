from torch import nn


class MLP(nn.Module):
    def __init__(self, n_classes: int, n_channels: int, width: int, height: int) -> None:
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=n_channels * width * height, out_features=100
            ),  # 1*28*28 (784 x 100)
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=100),  # (100 x 100)
            nn.ReLU(),
        )
        self.output = nn.Linear(in_features=100, out_features=n_classes)  # (100 x 10)

    def forward(self, x):
        x = self.sequential(x)
        return self.output(x)
