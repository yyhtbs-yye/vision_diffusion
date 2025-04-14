import torch

class InfiniteSampler(torch.utils.data.Sampler):
    """Infinite sampler for iteration-based training."""
    
    def __init__(self, dataset_size, shuffle=True, seed=0):
        self.dataset_size = dataset_size
        self.shuffle = shuffle
        self.seed = seed
        
    def __iter__(self):
        # Deterministically shuffle based on seed
        g = torch.Generator()
        g.manual_seed(self.seed)
        
        while True:
            if self.shuffle:
                indices = torch.randperm(self.dataset_size, generator=g).tolist()
            else:
                indices = list(range(self.dataset_size))
                
            for idx in indices:
                yield idx
                
    def __len__(self):
        return 2**31  # A large number that won't be reached in practice


class DefaultSampler(torch.utils.data.Sampler):
    """Default sampler for epoch-based validation."""
    
    def __init__(self, dataset_size, shuffle=False):
        self.dataset_size = dataset_size
        self.shuffle = shuffle
        
    def __iter__(self):
        if self.shuffle:
            return iter(torch.randperm(self.dataset_size).tolist())
        else:
            return iter(range(self.dataset_size))
            
    def __len__(self):
        return self.dataset_size
