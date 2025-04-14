from torch.utils.data import Dataset
from vision.torch_datasets.img_io import LoadImageFromFile
from vision.torch_datasets.transforms import Crop, Normalize
import os
from pathlib import Path

class NewImageDataset(Dataset):
    """Basic paired image dataset for training/inference."""
    
    def __init__(self, dataset_config):
        super().__init__()
        
        # Extract configuration from dataset_config dictionary
        self.paths = dataset_config.get('paths', {})
        self.data_prefix = dataset_config.get('data_prefix', {})
        
        self.image_paths = self._scan_images()

        pipeline_cfg = dataset_config.get('pipeline', [])
        self.transform_pipeline = self._build_pipeline(pipeline_cfg)

    def _scan_images(self):
        """Scan images in all folders and return their intersection."""
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        paths_by_key = {}
        base_names_by_key = {}
        
        # First, collect all image paths for each key
        for key in self.paths:
            folder_path = self.paths[key]
            path_prefix = os.path.join(folder_path, self.data_prefix.get(key, ''))
            image_paths = []
            
            for ext in extensions:
                paths = list(Path(path_prefix).glob(f'**/*{ext}'))
                image_paths.extend([str(path) for path in paths])
                            
            if len(image_paths) == 0:
                raise ValueError(f'No images found in {folder_path}')
                
            # Store paths and extract base filenames for intersection check
            paths_by_key[key] = image_paths
            base_names_by_key[key] = {os.path.basename(p): p for p in image_paths}
            
            print(f'Found {len(image_paths)} images in {folder_path}')
        
        # Find common base filenames across all folders
        all_keys = list(self.paths.keys())
        if not all_keys:
            raise ValueError("No paths specified")
            
        # Start with all base filenames from the first key
        common_base_names = set(base_names_by_key[all_keys[0]].keys())
        
        # Find intersection with all other keys
        for key in all_keys[1:]:
            common_base_names &= set(base_names_by_key[key].keys())
        
        if not common_base_names:
            raise ValueError(f"No common images found across all folders")
        
        # Sort the common base names to maintain consistent order
        common_base_names = sorted(common_base_names)
        
        # Create a dictionary of paths for each common image
        result = []
        for base_name in common_base_names:
            paths_dict = {}
            for key in self.paths:
                paths_dict[f"{key}_path"] = base_names_by_key[key][base_name]
            result.append(paths_dict)
        
        print(f'Found {len(result)} common images across all folders')
        return result

    def _build_pipeline(self, pipeline_cfg):
        """Build the data processing pipeline."""
        transforms_list = []
        
        for transform_cfg in pipeline_cfg:
            transform_cfg = transform_cfg.copy()  # Create a copy to avoid modifying original
            transform_type = transform_cfg.pop('type')
            
            if transform_type == 'LoadImageFromFile':
                transform = LoadImageFromFile(**transform_cfg)
            elif transform_type == 'Crop':
                transform = Crop(**transform_cfg)
            elif transform_type == 'Normalize':
                transform = Normalize(**transform_cfg)
            else:
                raise ValueError(f'Unknown transform type: {transform_type}')
                
            transforms_list.append(transform)
            
        return transforms_list
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        data = self.image_paths[idx]
        
        # Apply transforms
        for transform in self.transform_pipeline:
            data = transform(data)
            
        return data
    

if __name__=="__main__":
    import yaml
    from torch.utils.data import DataLoader

    # Load YAML config
    with open('/home/admyyh/python_workspaces/stable_diffusion/configs/pixel_diffusion_sr_train.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Extract dataset config
    dataset_config = config['data']['train_dataloader']['dataset']

    # Initialize dataset
    dataset = NewImageDataset(dataset_config)

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=2,  # Small batch for testing
        shuffle=True,
        num_workers=0  # Set to 0 for simplicity in testing
    )

    # Test one batch
    for batch in dataloader:
        print("Batch keys:", batch.keys())
        print("GT shape:", batch['gt'].shape)
        print("LQ shape:", batch['lq'].shape)
        break  # Only process one batch for testing