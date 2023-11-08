from torch import nn


class CNN2D(nn.Module):
    def __init__(self, n_classes: int, n_channels: int) -> None:
        super().__init__()
        self.input = nn.Conv2d(in_channels=n_channels, out_channels=20, kernel_size=(5, 5))
        self.sequential = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Flatten(),
            nn.Linear(in_features=800, out_features=500),
            nn.ReLU(),

            nn.Linear(in_features=500, out_features=n_classes),
        )
        self.output = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = self.input(x)
        x = self.sequential(x)

        return self.output(x)