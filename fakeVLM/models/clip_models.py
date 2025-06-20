import torch
import torch.nn as nn
from .clip import clip

class CLIPModel(nn.Module):
    """CLIP model with custom head for multi-class classification"""
    def __init__(self, name, n_classes=2):
        """
        Args:
            name (str): Name of the CLIP model to load
            n_classes (int): Number of classes for classification
        """
        super(CLIPModel, self).__init__()
        
        # Store config
        self.name = name
        self.n_classes = n_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.initialize_model()

    def initialize_model(self):
        """Initialize CLIP model and modify for multi-class classification"""
        # Load CLIP model using stored name attribute
        self.model, self.preprocess = clip.load(self.name, device='cpu')
        
        # Get embed dimension from model
        self.embed_dim = self.model.text_projection.shape[1] if hasattr(self.model, 'text_projection') else 512
        
        # Replace last layer for multi-class classification
        self.fc = nn.Linear(self.embed_dim, self.n_classes)  # Multi-class (n_classes categories)
        
        # Move to device
        self.model.to(self.device)

    def forward(self, x, return_feature=False):
        # Image encoding through CLIP backbone remains unchanged
        features = self.model.encode_image(x)
        
        # Feature return option remains unchanged
        if return_feature:
            return features
            
        # Final classification through modified FC layer
        return self.fc(features)

