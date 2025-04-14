import os
import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from vision.data_modules.image_data_module import ImageDataModule
from vision.models.generation.pixel_diffusion import PixelDiffusionModel  # Import the new DDPM model
from vision.callbacks.memory_mgmt import MemoryMonitorCallback

# Path to configuration file
config_path = "configs/pixel_diffusion_train.yaml"  # Updated config path

# Load YAML configuration
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Extract configurations
model_config = config['model']
optimizer_config = config['optimizer']
validation_config = config['validation']
train_config = config['train']
data_config = config['data']
logging_config = config['logging']
checkpoint_config = config['checkpoint']

# Create a new model or load from checkpoint
checkpoint_path = None  # Set this to your checkpoint path if needed

if checkpoint_path and os.path.exists(checkpoint_path):
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = PixelDiffusionModel.load_from_checkpoint(
        checkpoint_path, 
        model_config=model_config,
        optimizer_config=optimizer_config,
        validation_config=validation_config
    )
else:
    print("Creating new model")
    model = PixelDiffusionModel(
        model_config=model_config,
        optimizer_config=optimizer_config,
        validation_config=validation_config
    )

# Create data module
data_module = ImageDataModule(data_config)

# Set up logger
logger = TensorBoardLogger(
    save_dir=logging_config['log_dir'],
    name=logging_config['experiment_name']
)

# Set up callbacks
callbacks = []

# Checkpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join(logging_config['log_dir'], logging_config['experiment_name'], 'checkpoints'),
    filename='{epoch:02d}-{' + checkpoint_config['save_best_metric'] + ':.4f}',
    monitor=checkpoint_config['save_best_metric'],
    mode=checkpoint_config['save_best_mode'],
    save_top_k=checkpoint_config['save_top_k'],
    save_last=checkpoint_config['save_last']
)
callbacks.append(checkpoint_callback)

# Learning rate monitor
lr_monitor = LearningRateMonitor(logging_interval='step')
callbacks.append(lr_monitor)


callbacks.append(MemoryMonitorCallback())


# Initialize trainer
trainer = Trainer(
    max_steps=train_config['max_steps'],
    accelerator="auto",
    devices=4 if torch.cuda.is_available() else None,
    logger=logger,
    callbacks=callbacks,
    val_check_interval=train_config['val_check_interval'],
    log_every_n_steps=logging_config['log_every_n_steps']
)

# Train model
trainer.fit(model=model, datamodule=data_module)