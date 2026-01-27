#!/usr/bin/env python3
"""
Training Checkpoint Storage Example
===================================

Demonstrates using BlobStore for storing ML training checkpoints.
This is a non-caching use case - pure checkpoint storage with training metadata.

Features demonstrated:
- Periodic checkpoint saving during training
- Storing model state with training metrics
- Finding best checkpoint by metric
- Resuming training from checkpoint

Usage:
    uv run python examples/checkpoint_storage.py
"""

import sys
from pathlib import Path
import tempfile
from datetime import datetime
import time

# Add source directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cacheness.storage import BlobStore
import numpy as np


class MockNeuralNetwork:
    """A mock neural network for demonstration purposes."""
    
    def __init__(self, input_size: int = 784, hidden_size: int = 128, output_size: int = 10):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize random weights
        self.weights = {
            "w1": np.random.randn(input_size, hidden_size) * 0.01,
            "b1": np.zeros(hidden_size),
            "w2": np.random.randn(hidden_size, output_size) * 0.01,
            "b2": np.zeros(output_size),
        }
        
        self.epoch = 0
        self.best_loss = float("inf")
    
    def state_dict(self):
        """Get model state as dictionary."""
        return {
            "weights": {k: v.copy() for k, v in self.weights.items()},
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "architecture": {
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "output_size": self.output_size,
            }
        }
    
    def load_state_dict(self, state_dict):
        """Load model state from dictionary."""
        self.weights = {k: v.copy() for k, v in state_dict["weights"].items()}
        self.epoch = state_dict["epoch"]
        self.best_loss = state_dict["best_loss"]
    
    def train_epoch(self, learning_rate: float = 0.001):
        """Simulate one epoch of training."""
        # Simulate weight updates
        for key in self.weights:
            self.weights[key] += np.random.randn(*self.weights[key].shape) * learning_rate
        self.epoch += 1
        
        # Simulate loss (decreasing with noise)
        loss = 2.0 * np.exp(-0.1 * self.epoch) + np.random.rand() * 0.1
        
        # Track best loss
        if loss < self.best_loss:
            self.best_loss = loss
            
        return loss


def main():
    print("=" * 60)
    print("Training Checkpoint Storage with BlobStore")
    print("=" * 60)
    
    # Create a temporary directory for this example
    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoints_dir = Path(tmp_dir) / "checkpoints"
        
        # Create a BlobStore for checkpoint storage
        checkpoint_store = BlobStore(
            cache_dir=checkpoints_dir,
            compression="lz4",  # Fast compression for frequent saves
        )
        
        try:
            # Training parameters
            run_name = "mnist_classifier"
            total_epochs = 50
            checkpoint_every = 10
            learning_rate = 0.001
            
            print(f"\n🏋️ Starting training run: {run_name}")
            print(f"   Total epochs: {total_epochs}")
            print(f"   Checkpoint every: {checkpoint_every} epochs")
            print("-" * 40)
            
            # Initialize model
            model = MockNeuralNetwork()
            
            # Track best model info (stored in the checkpoint data itself)
            best_epoch = 0
            best_loss = float("inf")
            
            # Training loop
            print("\n📈 Training progress:")
            for epoch in range(1, total_epochs + 1):
                # Train one epoch
                loss = model.train_epoch(learning_rate)
                
                # Track best
                if loss < best_loss:
                    best_loss = loss
                    best_epoch = epoch
                
                # Print progress every 10 epochs
                if epoch % 10 == 0 or epoch == 1:
                    print(f"   Epoch {epoch:3d}/{total_epochs}: loss = {loss:.4f}")
                
                # Save checkpoint every checkpoint_every epochs
                # Key encodes epoch number for easy identification
                if epoch % checkpoint_every == 0:
                    checkpoint_key = f"{run_name}/epoch_{epoch:04d}"
                    checkpoint_store.put(model.state_dict(), key=checkpoint_key)
                    print(f"   💾 Saved checkpoint: {checkpoint_key}")
            
            # Save final checkpoint
            final_key = f"{run_name}/final"
            checkpoint_store.put(model.state_dict(), key=final_key)
            print(f"   💾 Saved final checkpoint: {final_key}")
            
            # List all checkpoints
            print(f"\n📋 All checkpoints for {run_name}:")
            print("-" * 40)
            checkpoints = checkpoint_store.list(prefix=run_name)
            
            for key in sorted(checkpoints):
                meta = checkpoint_store.get_metadata(key)
                size = meta.get("file_size", 0)
                print(f"   {key} ({size:,} bytes)")
            
            print(f"\n🏆 Best training loss was {best_loss:.4f} at epoch {best_epoch}")
            
            # Demonstrate loading checkpoint (resume training scenario)
            print("\n🔄 Simulating training resume from epoch_0030:")
            print("-" * 40)
            
            # Create a new model and load a checkpoint
            new_model = MockNeuralNetwork()
            print(f"   New model initial epoch: {new_model.epoch}")
            
            checkpoint_data = checkpoint_store.get(f"{run_name}/epoch_0030")
            if checkpoint_data:
                new_model.load_state_dict(checkpoint_data)
                print(f"   Loaded checkpoint from epoch: {new_model.epoch}")
                print(f"   Best loss in checkpoint: {new_model.best_loss:.4f}")
                
                # Continue training for a few more epochs
                print("\n   Continuing training for 5 more epochs:")
                for _ in range(5):
                    loss = new_model.train_epoch(learning_rate * 0.1)  # Lower LR
                    print(f"      Epoch {new_model.epoch}: loss = {loss:.4f}")
            
            # Show checkpoint cleanup
            print("\n🗑️  Cleaning up old checkpoints...")
            print("-" * 40)
            
            # Keep only final and last 2 epoch checkpoints
            all_checkpoints = sorted(checkpoint_store.list(prefix=run_name))
            to_delete = [k for k in all_checkpoints if "epoch_0010" in k or "epoch_0020" in k]
            for key in to_delete:
                checkpoint_store.delete(key)
                print(f"   Deleted: {key}")
            
            remaining = checkpoint_store.list(prefix=run_name)
            print(f"   Remaining checkpoints: {len(remaining)}")
            
            print("\n" + "=" * 60)
            print("✨ Checkpoint Storage Example Complete!")
            print("=" * 60)
        
        finally:
            checkpoint_store.close()


if __name__ == "__main__":
    main()
