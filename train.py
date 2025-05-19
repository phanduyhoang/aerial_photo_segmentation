import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from dataset import get_dataloader
from loss_function import CombinedLoss
from tqdm import tqdm

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    
    # Progress bar
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for images, masks in pbar:
        # Move data to device
        images = images.to(device)
        masks = masks.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)['out']  # DeepLabV3+ returns a dict
        
        # Calculate loss
        loss = criterion(outputs, masks)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update statistics
        running_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(train_loader)
    return epoch_loss

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc='Validation'):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            
            running_loss += loss.item()
    
    return running_loss / len(val_loader)

def main():
    # Set device and optimize CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        cudnn.benchmark = True
        torch.cuda.empty_cache()
    print(f'Using device: {device}')
    
    # Hyperparameters
    num_epochs = 100
    batch_size = 36  # Reduced batch size to avoid CUDA memory issues
    learning_rate = 1e-4
    num_workers = 4
    
    # Get dataloaders
    train_loader, val_loader = get_dataloader(
        batch_size=batch_size,
        num_workers=num_workers
    )
    print(f'Train batches: {len(train_loader)}, Val batches: {len(val_loader)}')
    
    # Initialize model with ImageNet weights
    model = deeplabv3_resnet50(weights='COCO_WITH_VOC_LABELS_V1')
    
    # Replace the classifier with a new one that has 6 output channels
    model.classifier = DeepLabHead(2048, 6)  # 2048 is the number of input channels
    
    # Move model to device
    model = model.to(device)
    
    # Initialize loss and optimizer
    criterion = CombinedLoss(
        ce_weight_path='class_weights.pt',
        alpha=0.7,
        smooth=1e-6,
        ignore_index=-100  # Fixed ignore_index
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    print('Starting training...')
    try:
        for epoch in range(num_epochs):
            # Train
            train_loss = train_one_epoch(
                model, train_loader, criterion,
                optimizer, device, epoch
            )
            
            # Validate
            val_loss = validate(model, val_loader, criterion, device)
            
            # Print epoch statistics
            print(f'\nEpoch {epoch+1}/{num_epochs}:')
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Val Loss: {val_loss:.4f}')
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, 'best_model.pth')
                print(f'Saved new best model with val_loss: {val_loss:.4f}')
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, f'checkpoint_epoch_{epoch+1}.pth')
                
    except KeyboardInterrupt:
        print('Training interrupted by user')
        # Save final model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, 'interrupted_model.pth')
        print('Saved model state at interruption')

if __name__ == '__main__':
    main() 