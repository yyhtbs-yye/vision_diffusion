import os
import argparse
import json
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration

class ImageFolderDataset(Dataset):
    """Dataset for processing a folder of images."""
    
    def __init__(self, image_folder, processor):
        self.processor = processor
        self.image_paths = []
        
        # Get all image files
        for root, _, files in os.walk(image_folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.image_paths.append(os.path.join(root, file))
        
        # Sort for reproducibility
        self.image_paths.sort()
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        try:
            # Load and preprocess the image
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(image, return_tensors="pt")
            
            # Remove batch dimension from processor output
            for k, v in inputs.items():
                inputs[k] = v.squeeze(0)
                
            return {
                "image_path": image_path,
                "inputs": inputs
            }
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            # Return placeholder for corrupted images
            blank_image = Image.new('RGB', (224, 224), color='black')
            inputs = self.processor(blank_image, return_tensors="pt")
            
            # Remove batch dimension
            for k, v in inputs.items():
                inputs[k] = v.squeeze(0)
                
            return {
                "image_path": image_path,
                "inputs": inputs,
                "error": str(e)
            }

def setup_ddp(rank, world_size):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", 
                           rank=rank, world_size=world_size)
    
    # Set device for current process
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """Clean up distributed training resources."""
    dist.destroy_process_group()

def setup_blip_model(rank):
    """Set up the BLIP model for image captioning."""
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    print(f"Rank {rank}: Loading BLIP model on {device}...")
    
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
    
    if dist.is_initialized():
        # Convert model to DDP
        model = DDP(model, device_ids=[rank])
    
    return processor, model, device

def collate_fn(batch):
    """Custom collate function to handle errors."""
    valid_batch = [item for item in batch if "error" not in item]
    if not valid_batch:
        # If all items in batch had errors, return dummy data
        return {"error": "All images in batch had errors"}
    
    # Process valid items
    image_paths = [item["image_path"] for item in valid_batch]
    
    # Collect inputs and stack them
    combined_inputs = {}
    for key in valid_batch[0]["inputs"].keys():
        combined_inputs[key] = torch.stack([item["inputs"][key] for item in valid_batch])
    
    return {
        "image_paths": image_paths,
        "inputs": combined_inputs
    }

def process_image_folder(
    rank, 
    world_size,
    image_folder, 
    output_json,
    batch_size=32, 
    use_beam_search=True,
    num_beams=5
):
    """
    Process all images in a folder and generate captions using distributed processing.
    
    Args:
        rank: Current process rank
        world_size: Total number of processes
        image_folder: Path to folder containing images
        output_json: Path to output JSON file
        batch_size: Batch size for processing
        use_beam_search: Whether to use beam search for better captions
        num_beams: Number of beams for beam search
    """
    # Initialize distributed
    if world_size > 1:
        setup_ddp(rank, world_size)
    
    # Set up the model
    processor, model, device = setup_blip_model(rank)
    
    # Create dataset and dataloader
    dataset = ImageFolderDataset(image_folder, processor)
    
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # Process images and generate captions
    results = {}
    
    # Only show progress bar on rank 0
    dataloader_iterator = tqdm(dataloader, desc=f"Rank {rank}: Generating captions") if rank == 0 else dataloader
    
    for batch in dataloader_iterator:
        # Skip batch if all images had errors
        if "error" in batch:
            continue
        
        # Move inputs to device
        for k, v in batch["inputs"].items():
            batch["inputs"][k] = v.to(device)
        
        # Generate captions
        with torch.no_grad():
            if use_beam_search:
                output_ids = model.module.generate(**batch["inputs"], max_length=50, num_beams=num_beams) if hasattr(model, 'module') else model.generate(**batch["inputs"], max_length=50, num_beams=num_beams)
            else:
                output_ids = model.module.generate(**batch["inputs"], max_length=50) if hasattr(model, 'module') else model.generate(**batch["inputs"], max_length=50)
        
        # Decode captions
        captions = processor.batch_decode(output_ids, skip_special_tokens=True)
        
        # Add context if it's a face image
        for i, caption in enumerate(captions):
            if "face" not in caption.lower() and "person" not in caption.lower():
                captions[i] = "a portrait of " + caption
        
        # Store results
        for image_path, caption in zip(batch["image_paths"], captions):
            rel_path = os.path.relpath(image_path, image_folder)
            results[rel_path] = caption
    
    # Gather results from all processes
    all_results = {}
    
    if world_size > 1:
        # Gather results on rank 0
        gathered_results = [None] * world_size if rank == 0 else None
        dist.gather_object(results, gathered_results, dst=0)
        
        if rank == 0:
            # Combine all results
            for result_dict in gathered_results:
                all_results.update(result_dict)
                
            # Save results
            with open(output_json, 'w') as f:
                json.dump(all_results, f, indent=2)
                
            print(f"Captioning complete! Generated captions for {len(all_results)} images.")
            print(f"Results saved to {output_json}")
    else:
        # Save results directly in single-process mode
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Captioning complete! Generated captions for {len(results)} images.")
        print(f"Results saved to {output_json}")
    
    # Clean up
    if world_size > 1:
        cleanup_ddp()

def create_paired_dataset(
    image_folder, 
    caption_json, 
    output_folder,
    copy_images=False
):
    """
    Create a paired dataset with images and captions.
    
    Args:
        image_folder: Path to folder containing images
        caption_json: Path to JSON file with captions
        output_folder: Path to output folder
        copy_images: Whether to copy images to output folder
    """
    import shutil
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Load captions
    with open(caption_json, 'r') as f:
        captions = json.load(f)
    
    # Create metadata file with pairs
    metadata = []
    for rel_path, caption in tqdm(captions.items(), desc="Creating paired dataset"):
        image_path = os.path.join(image_folder, rel_path)
        
        if copy_images:
            # Copy image to output folder
            output_path = os.path.join(output_folder, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.copy(image_path, output_path)
        
        # Add to metadata
        metadata.append({
            "file_name": rel_path,
            "caption": caption
        })
    
    # Save metadata
    metadata_path = os.path.join(output_folder, "metadata.jsonl")
    with open(metadata_path, 'w') as f:
        for item in metadata:
            f.write(json.dumps(item) + '\n')
    
    print(f"Created paired dataset with {len(metadata)} items")
    print(f"Metadata saved to {metadata_path}")

def main_worker(rank, world_size, args):
    """Main worker function for distributed processing."""
    process_image_folder(
        rank=rank,
        world_size=world_size,
        image_folder=args.image_folder,
        output_json=args.output_json,
        batch_size=args.batch_size,
        use_beam_search=args.beam_search,
        num_beams=args.num_beams
    )

def main():
    parser = argparse.ArgumentParser(description="Generate captions for FFHQ images using BLIP")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to folder containing FFHQ images")
    parser.add_argument("--output_json", type=str, required=True, help="Path to output JSON file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count(), help="Number of GPUs to use")
    parser.add_argument("--beam_search", action="store_true", help="Use beam search for better captions")
    parser.add_argument("--num_beams", type=int, default=5, help="Number of beams for beam search")
    parser.add_argument("--create_paired", action="store_true", help="Create a paired dataset with images and captions")
    parser.add_argument("--output_folder", type=str, help="Path to output folder for paired dataset")
    parser.add_argument("--copy_images", action="store_true", help="Copy images to output folder for paired dataset")
    
    args = parser.parse_args()
    
    # Limit number of GPUs to available ones
    args.num_gpus = min(args.num_gpus, torch.cuda.device_count())
    
    # Check if output directory exists, create if not
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    
    # Process images
    if args.num_gpus > 1 and torch.cuda.is_available():
        # Use distributed data parallel with multiple GPUs
        print(f"Using {args.num_gpus} GPUs for distributed processing")
        mp.spawn(
            main_worker,
            args=(args.num_gpus, args),
            nprocs=args.num_gpus,
            join=True
        )
    else:
        # Use single process (may still use one GPU)
        print("Using single process mode")
        process_image_folder(
            rank=0,
            world_size=1,
            image_folder=args.image_folder,
            output_json=args.output_json,
            batch_size=args.batch_size,
            use_beam_search=args.beam_search,
            num_beams=args.num_beams
        )
    
    # Create paired dataset if requested (only in primary process)
    if args.create_paired and args.output_folder:
        create_paired_dataset(
            args.image_folder,
            args.output_json,
            args.output_folder,
            args.copy_images
        )

if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()