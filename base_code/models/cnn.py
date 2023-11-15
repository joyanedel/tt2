from torch import nn


class CNN2D(nn.Module):
    def __init__(self, n_classes: int, n_channels: int) -> None:
        super().__init__()
        self.input = nn.Conv2d(in_channels=n_channels, out_channels=20, kernel_size=(5, 5)) # 28*28*1 -> 24*24*20
        self.sequential = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(num_features=20),

            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5)), # 24*24*20 -> 20*20*50
            nn.ReLU(),
            nn.BatchNorm2d(num_features=50),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)), # 20*20*50 -> 10*10*50
            nn.Dropout2d(0.25),

            nn.Conv2d(in_channels=50, out_channels=20, kernel_size=(5, 5)), # 10*10*50 -> 6*6*20
            nn.ReLU(),
            nn.BatchNorm2d(num_features=20),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)), # 6*6*20 -> 3*3*20
            nn.Dropout2d(0.25),

            nn.Flatten(),
            nn.Linear(in_features=180, out_features=64),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=64),
            nn.Dropout(0.25),

            nn.Linear(in_features=64, out_features=32),
            nn.LeakyReLU(),
            nn.BatchNorm1d(num_features=32),
            nn.Dropout(0.25),

            nn.Linear(in_features=32, out_features=n_classes),
        )
        self.output = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = self.input(x)
        x = self.sequential(x)

        return self.output(x)