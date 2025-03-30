import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import yaml
from pathlib import Path
import os
import torch.nn as nn


def load_config():
    """Load configuration from YAML file."""
    with open("/path to yaml file/config.yaml") as f:
        return yaml.safe_load(f)


def preprocess_data(config):
    """
    Read and preprocess data from CSV file.
    
    Args:
        config: Dictionary containing configuration parameters
        
    Returns:
        Tuple of (x, y) tensors containing normalized input and target features
    """
    dataframe = pd.read_csv(config["data"]["path"])
    x = dataframe[config["data"]["input_feature"]].values
    y = dataframe[config["data"]["target_feature"]].values

    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)

    x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    return x, y


class SimpleNN(nn.Module):
    """A simple neural network with configurable architecture."""
    
    def __init__(self, hidden_layers, activation, init_method):
        """
        Initialize the neural network.
        
        Args:
            hidden_layers: List of integers specifying hidden layer sizes
            activation: String name of activation function to use
            init_method: Weight initialization method ('xavier' or None)
        """
        super(SimpleNN, self).__init__()
        layers = []
        in_features = 1
        for out_features in hidden_layers:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(getattr(nn, activation)())
            in_features = out_features
        layers.append(nn.Linear(in_features, 1))
        layers.append(nn.Tanh())
        
        self.model = nn.Sequential(*layers)
        
        if init_method == "xavier":
            for layer in self.model:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.model(x)


def setup_optimizer(model, config):
    """
    Create optimizer based on configuration.
    
    Args:
        model: The neural network model
        config: Dictionary containing training configuration
        
    Returns:
        Configured optimizer instance
    """
    optimizer_type = config["training"]["optimizer"]
    learning_rate = config["training"]["learning_rate"]

    if optimizer_type == "SGD":
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.05)
    elif optimizer_type == "Adam":
        return optim.Adam(model.parameters(), lr=learning_rate, eps=config["adaptive"]["epsilon"], weight_decay=0.05)


def train_model(model, x, y, optimizer, config):
    """
    Train the neural network model.
    
    Args:
        model: The neural network model
        x: Input features tensor
        y: Target values tensor
        optimizer: Optimizer instance
        config: Dictionary containing training configuration
        
    Returns:
        Tuple of (losses, gradients) recorded during training
    """
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, 
        base_lr=0.0001, 
        max_lr=0.01, 
        step_size_up=500, 
        cycle_momentum=False
    )
    
    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    output_dir = config["output"]["save_dir"]
    grad_dir = os.path.join(output_dir, "gradients")
    os.makedirs(grad_dir, exist_ok=True)

    losses, grads = [], []

    for epoch in range(epochs):
        permutation = torch.randperm(len(x))
        x = x[permutation]
        y = y[permutation]

        for i in range(0, len(x), batch_size):
            batch_x = x[i:i + batch_size]
            batch_y = y[i:i + batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            grads.append([p.grad.detach().cpu().numpy().copy() for p in model.parameters()])
            losses.append(loss.item())
            optimizer.step()
            scheduler.step()
        
        if epoch % 10 == 0:
            with torch.no_grad():
                train_loss = criterion(model(x), y)
                print(f'Epoch {epoch:3d} | Loss: {train_loss:.4f} | LR: {optimizer.param_groups[0]["lr"]:.2e}')

    return losses, grads


def save_results(grad_dir, output_dir, losses, grads):
    """Save training results including gradients and losses."""
    np.save(os.path.join(grad_dir, "gradients.npy"), grads)
    np.save(os.path.join(output_dir, "losses.npy"), losses)


def main():
    """Main execution function for the training pipeline."""
    config = load_config()
    x, y = preprocess_data(config)
    model = SimpleNN(
        hidden_layers=config["model"]["hidden_layers"],
        activation=config["model"]["activation"],
        init_method=config["model"]["initialization"]
    )
    optimizer = setup_optimizer(model, config)
    losses, grads = train_model(model, x, y, optimizer, config)
    save_results(
        grad_dir=os.path.join(config["output"]["save_dir"], "gradients"),
        output_dir=config["output"]["save_dir"],
        losses=losses,
        grads=grads
    )


if __name__ == "__main__":
    main()