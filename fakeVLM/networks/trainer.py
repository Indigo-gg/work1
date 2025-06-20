import functools
import torch
import torch.nn as nn
from networks.base_model import BaseModel, init_weights
import sys
from models import get_model

class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        """Initialize the trainer"""
        super(Trainer, self).__init__(opt)
        self.opt = opt
        
        # Create model
        self.model = get_model(opt.arch)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Initialize fc layer weights
        torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)
        
        # Handle backbone parameters
        if opt.fix_backbone:
            params = []
            for name, p in self.model.named_parameters():
                if name == "fc.weight" or name == "fc.bias":
                    params.append(p)
                else:
                    p.requires_grad = False
        else:
            print("Your backbone is not fixed. Are you sure you want to proceed? If this is a mistake, enable the --fix_backbone command during training and rerun")
            import time 
            time.sleep(3)
            params = self.model.parameters()

        # Configure optimizer with default values if not provided
        if opt.optim == 'adam':
            # Use AdamW optimizer with learning rate, betas, and weight decay
            self.optimizer = torch.optim.AdamW(
                params, 
                lr=getattr(opt, 'lr', 1e-4), 
                betas=(getattr(opt, 'beta1', 0.9), 0.999), 
                weight_decay=getattr(opt, 'weight_decay', 0)
            )
        elif opt.optim == 'sgd':
            # Use SGD optimizer with learning rate and weight decay
            self.optimizer = torch.optim.SGD(
                params, 
                lr=getattr(opt, 'lr', 1e-4), 
                weight_decay=getattr(opt, 'weight_decay', 0)
            )
        else:
            # Raise exception if optimizer type is not supported
            raise ValueError("optim should be [adam, sgd]")

        # Setup loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def adjust_learning_rate(self, min_lr=1e-6):
        """Adjust learning rate by dividing by 10. Return False if below minimum"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True

    def set_input(self, input):
        """Set input data and move to device"""
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).long()

    def forward(self):
        """Forward pass"""
        self.output = self.model(self.input)

    def get_loss(self):
        """Compute loss"""
        return self.loss_fn(self.output, self.label)

    def optimize_parameters(self):
        """Optimization step"""
        self.forward()
        self.loss = self.loss_fn(self.output, self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()



