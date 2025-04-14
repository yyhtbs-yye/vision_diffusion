import pytorch_lightning as pl
import torch
import gc

class MemoryMonitorCallback(pl.Callback):
    """Callback to monitor and manage GPU memory usage during training."""
    
    def __init__(self, clean_on_val_start=True, clean_on_val_end=True):
        super().__init__()
        self.clean_on_val_start = clean_on_val_start
        self.clean_on_val_end = clean_on_val_end
    
    def on_validation_start(self, trainer, pl_module):
        """Called when validation begins."""
        if self.clean_on_val_start and torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            # Optionally log memory usage
            print(f"GPU memory allocated at validation start: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    def on_validation_end(self, trainer, pl_module):
        """Called when validation ends."""
        if self.clean_on_val_end and torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            # Optionally log memory usage
            print(f"GPU memory allocated after validation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Optionally monitor memory after each training batch."""
        if batch_idx % 100 == 0 and torch.cuda.is_available():
            print(f"GPU memory after batch {batch_idx}: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Usage:
# Add this to your PyTorch Lightning Trainer:
# trainer = pl.Trainer(
#     callbacks=[MemoryMonitorCallback()],
#     ...
# )