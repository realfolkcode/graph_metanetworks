from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import torch.nn as nn
from sklearn.metrics import accuracy_score


class Trainer:
    """Image classifier trainer."""

    def __init__(self,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 num_epochs: int,
                 optimizer: Optimizer,
                 device: str):
        """Initializes an instance of Trainer.

        Args:
            train_loader: Train set data loader.
            val_loader: Validation set data loader.
            num_epochs: The number of epochs.
            optimizer: Optimizer.
            device: Device.
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
    
    def _train_epoch(self,
                     model: nn.Module):
        """One epoch pass."""
        model.train()

        for batch in self.train_loader:
            x, y_true = batch
            x = x.to(self.device)
            y_true = y_true.to(self.device)

            self.optimizer.zero_grad()
            y_pred = model(x)
            loss = self.criterion(y_pred, y_true)
            loss.backward()
            self.optimizer.step()
    
    @torch.no_grad()
    def validate(self,
                 model: nn.Module) -> Tuple[float, float]:
        """Validation.
        
        Args:
            model: Image classifier.
        
        Returns:
            Cross entropy loss and accuracy score. 
        """
        val_loss = 0
        predictions = []
        true_labels = []

        model.eval()

        for batch in self.val_loader:
            x, y_true = batch
            x = x.to(self.device)
            y_true = y_true.to(self.device)
            true_labels += y_true.flatten().tolist()

            y_pred = model(x)
            loss = self.criterion(y_pred, y_true)
            val_loss += loss.item()

            y_pred = y_pred.argmax(dim=-1, keepdim=True)
            predictions += y_pred.flatten().tolist()
        
        val_loss /= len(self.val_loader)
        acc = accuracy_score(true_labels, predictions)
        
        return val_loss, acc
    
    def train(self,
              model: nn.Module):
        """Trains an image classifier.

        Args:
            model: Image classifier.
        """
        for epoch in range(self.num_epochs):
            self._train_epoch(model)
        
        val_loss, val_acc = self.validate(model)
        print(f"Val loss: {val_loss}, Val accuracy: {val_acc}")
