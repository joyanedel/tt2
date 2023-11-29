from torch import nn


class MLPDropout(nn.Module):
    def __init__(self, n_classes: int, n_channels: int, width: int, height: int) -> None:
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(
                in_features=n_channels * width * height, out_features=400
            ),  # 1*28*28 (784 x 400)
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=400, out_features=400),  # (400 x 400)
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.output = nn.Linear(in_features=400, out_features=n_classes)  # (400 x 10)

    def forward(self, x):
        x = self.sequential(x)
        return self.output(x)
