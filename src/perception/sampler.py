import torch
import numpy as np
from src.perception.model import load_pretrained_miner

class Sampler:
    def __init__(self, model_path: str = None, device: str = "cpu"):
        self.device = device
        self.miner = load_pretrained_miner(model_path).to(device)

    def compute_difficulty_scores(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Runs the classifier on a batch of embeddings.
        
        Args:
            embeddings: (B, 512) numpy array.
            
        Returns:
            scores: (B,) numpy array of difficulty probabilities.
        """
        with torch.no_grad():
            tensor_input = torch.from_numpy(embeddings).float().to(self.device)
            scores = self.miner(tensor_input)
            return scores.cpu().numpy().flatten()

    def sample_hard_examples(self, data_ids: list, embeddings: np.ndarray, top_k: float = 0.2):
        """
        Selects the top_k fraction of hardest examples.
        
        Args:
            data_ids: List of identifiers for the data points.
            embeddings: (B, 512) Feature vectors.
            top_k: Fraction of data to keep (e.g., 0.2 for top 20%).
            
        Returns:
            selected_ids: List of IDs corresponding to hard examples.
        """
        scores = self.compute_difficulty_scores(embeddings)
        
        # Sort indices by score descending
        sorted_indices = np.argsort(scores)[::-1]
        
        num_keep = int(len(data_ids) * top_k)
        selected_indices = sorted_indices[:num_keep]
        
        return [data_ids[i] for i in selected_indices], scores[selected_indices]

# Mock function to simulate embedding extraction from a backbone
def extract_embeddings(mock_data_shape=(10, 512)):
    return np.random.randn(*mock_data_shape).astype(np.float32)
