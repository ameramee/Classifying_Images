"""
EECS 445 - Introduction to Machine Learning
Winter 2025 - Project 2

Train Challenge
    Train a convolutional neural network to classify the heldout images
    Periodically output training information, and saves model checkpoints
    Usage: python train_challenge.py
"""
from copy import deepcopy

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import get_train_val_test_loaders
from model.challenge import Challenge
from train_common import evaluate_epoch, early_stopping, restore_checkpoint, save_checkpoint, train_epoch
from utils import config, set_random_seed, make_training_plot
from train_target import freeze_layers, train

def train(
    tr_loader: DataLoader,
    va_loader: DataLoader,
    te_loader: DataLoader,
    model: torch.nn.Module,
    model_name: str,
    num_layers: int = 0,
) -> None:
    """
    This function trains the target model. Only the weights of unfrozen layers of the model passed 
    into this function will be updated in training.
    
    Args:
        tr_loader: DataLoader for training data
        va_loader: DataLoader for validation data
        te_loader: DataLoader for test data
        model: subclass of torch.nn.Module, model to train on
        model_name: str, checkpoint path for the model
        num_layers: int, the number of source model layers to freeze
    """
    set_random_seed()
    
    # TODO: define loss function, and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = 0.01)

    print("Loading challenge model with", num_layers, "layers frozen")
    model, start_epoch, stats = restore_checkpoint(model, model_name)

    axes = make_training_plot("Target Training")

    evaluate_epoch(
        axes,
        tr_loader,
        va_loader,
        te_loader,
        model,
        criterion,
        start_epoch,
        stats,
        include_test=True,
    )

    # initial val loss for early stopping
    prev_val_loss = stats[0][1]

    # TODO: patience for early stopping
    patience = 5
    curr_patience = 0

    # Loop over the entire dataset multiple times
    epoch = start_epoch
    while curr_patience < patience:
        # Train model
        train_epoch(tr_loader, model, criterion, optimizer)
        # Evaluate model
        evaluate_epoch(
            axes,
            tr_loader,
            va_loader,
            te_loader,
            model,
            criterion,
            epoch + 1,
            stats,
            include_test=True,
        )

        # Save model parameters
        save_checkpoint(model, epoch + 1, model_name, stats)

        curr_patience, prev_val_loss = early_stopping(stats, curr_patience, prev_val_loss)
        epoch += 1

    print("Finished Training")

    # Keep plot open
    print(f"Saving training plot to target_training_plot_frozen_layers={num_layers}.png...")
    plt.savefig(f"target_training_plot_frozen_layers={num_layers}.png", dpi=200)
    plt.ioff()
    plt.show()

def main():
    set_random_seed()
    
    # Data loaders
    tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
        task="target",
        batch_size=config("challenge.batch_size"),
    )
    
    freeze_none = Challenge()
    print("Loading source...")
    freeze_none, _, _ = restore_checkpoint(
        freeze_none,
        config("Csource.checkpoint"),
        force=True,
        pretrain=True,
    )

    #freeze_one = deepcopy(freeze_none)
    #freeze_two = deepcopy(freeze_none)
    freeze_three = deepcopy(freeze_none)

    #freeze_layers(freeze_one, 1)
    #freeze_layers(freeze_two, 2)
    freeze_layers(freeze_three, 3)

    #train(tr_loader, va_loader, te_loader, freeze_none, config("target.frozen_checkpoint").format(layer=0), 0)
    #train(tr_loader, va_loader, te_loader, freeze_one, config("target.frozen_checkpoint").format(layer=1), 1)
    #train(tr_loader, va_loader, te_loader, freeze_two, config("target.frozen_checkpoint").format(layer=2), 2)
    train(tr_loader, va_loader, te_loader, freeze_three, config("challenge.checkpoint").format(layer=3), 3)
    # Model
    #model = Challenge()
'''
    # TODO: define loss function, and optimizer
    model = Challenge()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = 0.01)
   
    # Attempts to restore the latest checkpoint if exists
    print("Loading challenge...")
    model, start_epoch, stats = restore_checkpoint(model, config("challenge.checkpoint"))

    axes = make_training_plot()

    # Evaluate the randomly initialized model
    evaluate_epoch(
        axes,
        tr_loader,
        va_loader,
        te_loader,
        model,
        criterion,
        start_epoch,
        stats,
        include_test=True,
    )

    # initial val loss for early stopping
    prev_val_loss = stats[0][1]

    # TODO: define patience for early stopping
    patience = 5
    curr_patience = 0

    # Loop over the entire dataset multiple times
    epoch = start_epoch
    while curr_patience < patience:
        # Train model
        train_epoch(tr_loader, model, criterion, optimizer)

        # Evaluate model
        evaluate_epoch(
            axes,
            tr_loader,
            va_loader,
            te_loader,
            model,
            criterion,
            epoch + 1,
            stats,
            include_test=True,
        )

        # Save model parameters
        save_checkpoint(model, epoch + 1, config("challenge.checkpoint"), stats)

        # Updates early stopping parameters
        curr_patience, prev_val_loss = early_stopping(stats, curr_patience, prev_val_loss)

        epoch += 1
    print("Finished Training")
    # Save figure and keep plot open
    plt.savefig("challenge_training_plot.png", dpi=200)
    plt.ioff()
    plt.show()
'''

if __name__ == "__main__":
    main()
