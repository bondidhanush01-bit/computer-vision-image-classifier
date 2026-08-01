"""
Training script for image classifier
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import argparse
from tqdm import tqdm
import logging
from pathlib import Path
from typing import Tuple
import yaml

from src.utils import create_directories, load_config, get_device
from src.inference import ImageClassifier

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Trainer:
    """Training pipeline for image classifier"""
    
    def __init__(self, config: dict, device: str = None):
        """
        Initialize trainer
        
        Args:
            config: Configuration dictionary
            device: Device to use (cuda/cpu)
        """
        self.config = config
        self.device = device or get_device()
        
        # Create directories
        create_directories([
            self.config['checkpoint']['save_dir'],
            str(Path(self.config['logging']['log_file']).parent),
            self.config['logging']['tensorboard_dir']
        ])
        
        # Initialize model
        self.model = ImageClassifier(
            model_type=self.config['model']['type'],
            num_classes=self.config['model']['num_classes'],
            device=self.device
        ).model
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        
        # Data loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
    
    def _get_optimizer(self):
        """Get optimizer based on config"""
        optimizer_name = self.config['training']['optimizer'].lower()
        lr = self.config['training']['learning_rate']
        
        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr,
                           momentum=self.config['training']['momentum'])
        elif optimizer_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _get_scheduler(self):
        """Get learning rate scheduler based on config"""
        scheduler_name = self.config['training']['scheduler'].lower()
        epochs = self.config['training']['epochs']
        
        if scheduler_name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        elif scheduler_name == 'step':
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        elif scheduler_name == 'exponential':
            return optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
        else:
            return None
    
    def prepare_data(self):
        """Prepare data loaders"""
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        # Load CIFAR-10 dataset
        dataset = CIFAR10(root='data/cifar10', train=True, download=True, transform=transform_train)
        test_dataset = CIFAR10(root='data/cifar10', train=False, download=True, transform=transform_test)
        
        # Split training data
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        self.train_loader = DataLoader(train_set, batch_size=self.config['training']['batch_size'],
                                       shuffle=True, num_workers=4)
        self.val_loader = DataLoader(val_set, batch_size=self.config['training']['batch_size'],
                                     shuffle=False, num_workers=4)
        self.test_loader = DataLoader(test_dataset, batch_size=self.config['training']['batch_size'],
                                      shuffle=False, num_workers=4)
        
        logger.info(f"Data loaded: {len(train_set)} train, {len(val_set)} val, {len(test_dataset)} test")
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for images, labels in tqdm(self.train_loader, desc="Training"):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self) -> Tuple[float, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validating"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        accuracy = correct / total
        return total_loss / len(self.val_loader), accuracy
    
    def train(self):
        """Train model for configured number of epochs"""
        self.prepare_data()
        best_accuracy = 0
        
        for epoch in range(self.config['training']['epochs']):
            train_loss = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            logger.info(f"Epoch {epoch+1}/{self.config['training']['epochs']} - "
                       f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            if self.scheduler:
                self.scheduler.step()
            
            # Save best model
            if val_acc > best_accuracy and self.config['checkpoint']['save_best']:
                best_accuracy = val_acc
                self.save_checkpoint(f"{self.config['checkpoint']['save_dir']}/best_model.pth")
            
            # Periodic checkpoint
            if (epoch + 1) % self.config['checkpoint']['save_interval'] == 0:
                self.save_checkpoint(f"{self.config['checkpoint']['save_dir']}/model_epoch_{epoch+1}.pth")
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info(f"Checkpoint saved to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--device', default=None, help='Device to use')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override with command line args
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    
    device = args.device or get_device()
    
    # Train
    trainer = Trainer(config, device)
    trainer.train()


if __name__ == '__main__':
    main()
