import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, activation_fn):
        super(CNNModel, self).__init__()
        self.activation = activation_fn
        
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            self.activation,
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            self.activation,
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            self.activation,
            nn.MaxPool2d(2)
        )
        
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            self.activation,
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x
