import torch
import torch.nn as nn 
import torch.nn.functional as F 

class Feedforward():
    def __init__(self, num_classes=4):
        super().__init__(self)
        self.network = nn.Sequential(
            nn.Linear(63, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)
