import torch
import torch.nn as nn
import torch.nn.functional as F

class HardExampleMiner(nn.Module):
    """
    Lightweight classification model to identify "hard examples" from sensor data embeddings.
    Trained to distinguish between clean data and edge cases (occlusions, bad weather).
    """
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.fc3 = nn.Linear(64, 1) # Probability of being a "hard example"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) tensor of embeddings (e.g. from a frozen ResNet/PointNet)
        Returns:
            prob: (B, 1) probability score [0, 1]
        """
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

def load_pretrained_miner(path: str = None) -> HardExampleMiner:
    model = HardExampleMiner()
    if path:
        try:
            model.load_state_dict(torch.load(path))
        except FileNotFoundError:
            print(f"Warning: Model weights not found at {path}, using random init.")
    model.eval()
    return model
