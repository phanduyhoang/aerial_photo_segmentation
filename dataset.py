import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class AerialDataset(Dataset):
    def __init__(self, root_dir='data/tiles', base_dir='.', transform=None):
        """
        Args:
            root_dir (str): Path to the tiles directory containing 'images' and 'masks' subdirectories
            base_dir (str): Base directory for the project
            transform (callable, optional): Optional transform to be applied on images
        """
        self.root_dir = os.path.join(base_dir, root_dir)
        self.transform = transform
        
        # Get all image files
        self.image_dir = os.path.join(self.root_dir, 'images')
        self.mask_dir = os.path.join(self.root_dir, 'masks')
        self.image_files = sorted(os.listdir(self.image_dir))
        
        # Load class weights if available
        self.class_weights = None
        weights_path = os.path.join(base_dir, 'src', 'class_weights.pt')
        if os.path.exists(weights_path):
            self.class_weights = torch.load(weights_path)
        
        # Define color to class mapping
        self.COLOR_MAP = {
            'empty':  (255, 255, 255),  # white
            'soil':   (255, 0, 0),      # red
            'road':   (0, 6, 255),      # blue
            'forest': (157, 255, 226),  # light blue
            'grass':  (18, 255, 0),     # green
            'house':  (35, 145, 105)    # dark green
        }
        
        # Create reverse mapping from RGB to class index
        self.rgb_to_class = {}
        for idx, (_, color) in enumerate(self.COLOR_MAP.items()):
            self.rgb_to_class[color] = idx
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)  # Same filename for mask
        
        # Read image and mask
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')
        
        # Convert to numpy arrays
        image_np = np.array(image)
        mask_np = np.array(mask)
        
        # Convert mask RGB to class indices
        h, w = mask_np.shape[:2]
        mask_classes = np.zeros((h, w), dtype=np.int64)
        
        for rgb, class_idx in self.rgb_to_class.items():
            mask_match = np.all(mask_np == rgb, axis=2)
            mask_classes[mask_match] = class_idx
        
        # Convert to tensors
        image = transforms.ToTensor()(image)
        mask = torch.from_numpy(mask_classes)
        
        # Apply additional transforms if specified
        if self.transform:
            image = self.transform(image)
        
        return image, mask

def get_dataloader(batch_size=8, num_workers=4, shuffle=True, base_dir='.'):
    """
    Creates train and validation dataloaders
    
    Args:
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for data loading
        shuffle (bool): Whether to shuffle the data
        base_dir (str): Base directory for the project
    
    Returns:
        train_loader, val_loader: DataLoader objects for training and validation
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = AerialDataset(transform=transform, base_dir=base_dir)
    
    # Split into train and validation sets (90-10 split)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

if __name__ == '__main__':
    # Test the dataloader
    train_loader, val_loader = get_dataloader(batch_size=4)
    
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    
    # Get a batch of data
    images, masks = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"Images: {images.shape}")  # Should be [B, C, H, W]
    print(f"Masks: {masks.shape}")    # Should be [B, H, W]
    print(f"Unique classes in batch: {torch.unique(masks).tolist()}")
