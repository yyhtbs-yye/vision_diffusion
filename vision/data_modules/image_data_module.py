import pytorch_lightning as pl
import yaml
from torch.utils.data import DataLoader
import torch
from .samplers import InfiniteSampler, DefaultSampler
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from vision.torch_datasets.new_image_dataset import NewImageDataset

class ImageDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for image datasets."""
    
    def __init__(self, config=None):
        super().__init__()
        
        self.config = config
            
        self.train_dataset = None
        self.val_dataset = None
        # Initialize cached dataloaders to None
        self._train_dataloader = None
        self._val_dataloader = None
        
    def setup(self, stage=None):
        """Setup datasets."""
        if stage == 'fit' or stage is None:
            # Setup training dataset
            train_cfg = self.config['train_dataloader']['dataset']
            self.train_dataset = NewImageDataset(train_cfg)
            
            # Setup validation dataset
            val_cfg = self.config['val_dataloader']['dataset']
            self.val_dataset = NewImageDataset(val_cfg)
    
    def _create_dataloader(self, dataset, loader_cfg):
        """Create dataloader based on config with optimizations for large datasets."""
        # Create sampler
        sampler_cfg = loader_cfg.get('sampler', {'type': 'DefaultSampler', 'shuffle': False})
        if sampler_cfg['type'] == 'InfiniteSampler':
            sampler = InfiniteSampler(
                dataset_size=len(dataset),
                shuffle=sampler_cfg.get('shuffle', True)
            )
        else:  # DefaultSampler
            sampler = DefaultSampler(
                dataset_size=len(dataset),
                shuffle=sampler_cfg.get('shuffle', False)
            )
            
        # Performance optimizations for DataLoader
        # Pin memory for faster CPU to GPU transfers
        pin_memory = torch.cuda.is_available()
        
        # Use persistent workers to avoid worker process creation overhead
        persistent_workers = loader_cfg.get('persistent_workers', True)
        
        # Prefetch factor controls how many samples loaded in advance by each worker
        prefetch_factor = 2
        
        # Create dataloader with optimized settings
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=loader_cfg.get('batch_size', 16),
            num_workers=loader_cfg.get('num_workers', 4),
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if persistent_workers else None,
            sampler=sampler
        )
        
        return dataloader
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """Get train dataloader, creating it only once for efficiency."""
        if self._train_dataloader is None:
            self._train_dataloader = self._create_dataloader(
                self.train_dataset,
                self.config['train_dataloader']
            )
        return self._train_dataloader
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        """Get validation dataloader, creating it only once for efficiency."""
        if self._val_dataloader is None:
            self._val_dataloader = self._create_dataloader(
                self.val_dataset,
                self.config['val_dataloader']
            )
        return self._val_dataloader

