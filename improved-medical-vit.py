import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
from torchvision import transforms, models
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import random
import json
import time
from vit_pytorch import ViT
from timm.models.vision_transformer import VisionTransformer
import timm
import wandb

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Configuration
CONFIG = {
    'base_path': '/path/to/dataset',  # Update with the correct path
    'batch_size': 32,
    'image_size': 224,
    'num_epochs': 50,
    'initial_lr': 1e-4,
    'weight_decay': 0.01,
    'dropout_rate': 0.2,
    'model_name': 'vit_base_patch16_224',  # Using a pre-trained ViT from timm
    'use_pretrained': True,
    'use_amp': True,  # Use mixed precision training
    'use_wandb': False,  # Set to True if you want to use Weights & Biases
    'save_path': 'models',
    'early_stopping_patience': 10,
}

# Get device
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

DEVICE = get_device()
print(f"Using device: {DEVICE}")

# Dataset Class with improved handling
class MedicalImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, split=None):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split  # 'train', 'val', 'test', or None for all
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        self.load_images()

    def load_images(self):
        """Load images with improved error handling and logging"""
        subsets = ['train', 'val', 'test'] if self.split is None else [self.split]
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        
        for subset in subsets:
            subset_path = os.path.join(self.root_dir, subset)
            
            if not os.path.exists(subset_path):
                continue
                
            for class_name in sorted(os.listdir(subset_path)):
                class_path = os.path.join(subset_path, class_name)
                
                if not os.path.isdir(class_path):
                    continue
                    
                # Add class to mapping if not already exists
                if class_name not in self.class_to_idx:
                    idx = len(self.class_to_idx)
                    self.class_to_idx[class_name] = idx
                    self.idx_to_class[idx] = class_name
                
                # Load images for this class
                loaded_count = 0
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    file_ext = os.path.splitext(img_name.lower())[1]
                    
                    if file_ext in valid_extensions:
                        try:
                            # Quick check that we can open the image
                            with Image.open(img_path) as img:
                                pass
                            
                            self.image_paths.append(img_path)
                            self.labels.append(self.class_to_idx[class_name])
                            loaded_count += 1
                        except Exception as e:
                            print(f"Error with image {img_path}: {e}")
                
                print(f"Loaded {loaded_count} images for class '{class_name}'")

        print(f"Found {len(self.image_paths)} images across {len(self.class_to_idx)} classes")
        for cls_name, idx in self.class_to_idx.items():
            count = self.labels.count(idx)
            print(f"  - Class '{cls_name}': {count} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            # Open image with PIL
            image = Image.open(img_path).convert("RGB")
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
            
        except Exception as e:
            print(f"Error loading image {img_path} during __getitem__: {e}")
            # Return a placeholder instead
            placeholder = torch.zeros((3, CONFIG['image_size'], CONFIG['image_size']))
            return placeholder, label

# Custom early stopping handler
class EarlyStopping:
    def __init__(self, patience=7, verbose=True, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.counter = 0

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} → {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# Data Augmentation with separate transforms for training and validation
def get_transforms():
    # Strong augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        # Add random erasing for robustness
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Only resize, convert to tensor, and normalize for validation/testing
    val_transform = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

# Create model with improved architecture
def create_model(num_classes):
    """Create a Vision Transformer model for medical image classification"""
    print(f"Creating model: {CONFIG['model_name']} with {num_classes} classes")
    
    # Option 1: Use pretrained ViT from timm
    if CONFIG['use_pretrained']:
        model = timm.create_model(
            CONFIG['model_name'],
            pretrained=True,
            num_classes=num_classes,
            drop_rate=CONFIG['dropout_rate'],
        )
        print(f"Loaded pretrained model: {CONFIG['model_name']}")
    else:
        # Option 2: Custom ViT from vit_pytorch 
        model = ViT(
            image_size=CONFIG['image_size'],
            patch_size=16,
            num_classes=num_classes,
            dim=768,
            depth=12,
            heads=12,
            mlp_dim=3072,
            dropout=CONFIG['dropout_rate'],
            emb_dropout=CONFIG['dropout_rate']
        )
        print("Created custom ViT model")
    
    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    model = model.to(DEVICE)
    return model

# Training function with monitoring and optimization
def train_model(model, train_loader, valid_loader, num_classes, num_epochs=15):
    # Initialize Weights & Biases if enabled
    if CONFIG['use_wandb']:
        wandb.init(project="medical-image-classification", 
                  config=CONFIG,
                  name=f"vit-medical-{time.strftime('%Y%m%d-%H%M%S')}")
    
    # Create directory for model checkpoints
    os.makedirs(CONFIG['save_path'], exist_ok=True)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    
    # Use different learning rates for different parts of the model
    if CONFIG['use_pretrained']:
        # Lower learning rate for pretrained layers
        head_params = list(model.head.parameters()) if hasattr(model, 'head') else []
        if not head_params and hasattr(model, 'fc'):
            head_params = list(model.fc.parameters())
            
        base_params = [p for n, p in model.named_parameters() 
                      if not any(h_p is p for h_p in head_params)]
        
        optimizer = optim.AdamW([
            {'params': base_params, 'lr': CONFIG['initial_lr'] * 0.1},
            {'params': head_params, 'lr': CONFIG['initial_lr']}
        ], weight_decay=CONFIG['weight_decay'])
    else:
        optimizer = optim.AdamW(model.parameters(), 
                              lr=CONFIG['initial_lr'], 
                              weight_decay=CONFIG['weight_decay'])
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=CONFIG['initial_lr'] * 0.01
    )
    
    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=CONFIG['early_stopping_patience'],
        verbose=True,
        path=os.path.join(CONFIG['save_path'], 'best_model.pth')
    )
    
    # For mixed precision training
    scaler = torch.cuda.amp.GradScaler() if CONFIG['use_amp'] else None
    
    # Training loop
    best_val_accuracy = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for i, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision if enabled
            if CONFIG['use_amp']:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                # Backward pass with scaler
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Accuracy': f'{100 * train_correct / train_total:.2f}%'
            })
        
        # Calculate training metrics for epoch
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = 100 * train_correct / train_total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                if CONFIG['use_amp']:
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate validation metrics
        epoch_val_loss = val_loss / len(valid_loader)
        epoch_val_acc = 100 * val_correct / val_total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        
        # Print epoch results
        print(f"Epoch {epoch+1}: "
              f"Train Loss: {epoch_train_loss:.4f}, "
              f"Train Accuracy: {epoch_train_acc:.2f}%, "
              f"Val Loss: {epoch_val_loss:.4f}, "
              f"Val Accuracy: {epoch_val_acc:.2f}%")
        
        # Update learning rate
        scheduler.step()
        
        # Log to W&B if enabled
        if CONFIG['use_wandb']:
            wandb_log = {
                'epoch': epoch,
                'train_loss': epoch_train_loss,
                'train_acc': epoch_train_acc,
                'val_loss': epoch_val_loss,
                'val_acc': epoch_val_acc,
                'learning_rate': scheduler.get_last_lr()[0]
            }
            wandb.log(wandb_log)
        
        # Check for best model and save
        if epoch_val_acc > best_val_accuracy:
            best_val_accuracy = epoch_val_acc
            torch.save(model.state_dict(), os.path.join(CONFIG['save_path'], 'best_model.pth'))
            print(f"✓ New best model saved with validation accuracy: {epoch_val_acc:.2f}%")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss,
                'val_acc': epoch_val_acc,
            }, os.path.join(CONFIG['save_path'], f'checkpoint_epoch{epoch+1}.pth'))
        
        # Early stopping check
        early_stopping(epoch_val_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Plot training progress
    plot_training_progress(train_losses, train_accs, val_losses, val_accs)
    
    # Close W&B if used
    if CONFIG['use_wandb']:
        wandb.finish()
    
    # Return best validation accuracy
    return best_val_accuracy

# Plot training progress
def plot_training_progress(train_losses, train_accs, val_losses, val_accs):
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.close()

# Validation function with detailed metrics
def validate_model(model, test_loader, idx_to_class):
    """Comprehensive model validation function"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Validating"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Forward pass
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            # Update counters
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            # Store predictions, probabilities and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    # Combine all probabilities
    all_probs = np.vstack(all_probs)
    
    # Compute metrics
    accuracy = 100 * correct_predictions / total_predictions
    
    # Class names for the report
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    
    # Generate classification report
    report = classification_report(
        all_labels, 
        all_preds, 
        target_names=class_names,
        output_dict=True
    )
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, class_names, title='Confusion Matrix')
    
    # Return all metrics
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': report,
        'confusion_matrix': cm,
        'all_preds': all_preds,
        'all_labels': all_labels,
        'all_probs': all_probs
    }

# Plot confusion matrix
def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, 
                yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

# Main function
def main():
    print("Starting Medical Image Classification with Vision Transformer")
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Load datasets with appropriate transforms
    print("\nLoading datasets...")
    train_dataset = MedicalImageDataset(
        CONFIG['base_path'], 
        transform=train_transform,
        split='train'
    )
    
    val_dataset = MedicalImageDataset(
        CONFIG['base_path'], 
        transform=val_transform,
        split='val'
    )
    
    test_dataset = MedicalImageDataset(
        CONFIG['base_path'], 
        transform=val_transform,
        split='test'
    )
    
    # Make sure we have the same class mapping across datasets
    # This ensures consistent class indices
    if len(val_dataset.class_to_idx) > 0:
        train_dataset.class_to_idx = val_dataset.class_to_idx
        train_dataset.idx_to_class = val_dataset.idx_to_class
    if len(test_dataset.class_to_idx) > 0:
        train_dataset.class_to_idx = test_dataset.class_to_idx
        train_dataset.idx_to_class = test_dataset.idx_to_class
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # Get number of classes
    num_classes = len(train_dataset.class_to_idx)
    print(f"Number of classes: {num_classes}")
    
    # Create model
    model = create_model(num_classes)
    
    # Train model
    print("\nTraining model...")
    best_val_accuracy = train_model(
        model, 
        train_loader, 
        val_loader, 
        num_classes, 
        num_epochs=CONFIG['num_epochs']
    )
    
    # Load best model for evaluation
    print("\nLoading best model for evaluation...")
    model.load_state_dict(torch.load(os.path.join(CONFIG['save_path'], 'best_model.pth')))
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = validate_model(model, test_loader, train_dataset.idx_to_class)
    
    # Print results
    print("\n🔍 Model Evaluation Results:")
    print(f"Test Accuracy: {test_results['accuracy']:.2f}%")
    print(f"Precision: {test_results['precision']:.4f}")
    print(f"Recall: {test_results['recall']:.4f}")
    print(f"F1 Score: {test_results['f1_score']:.4f}")
    
    # Print per-class metrics
    print("\n📊 Per-Class Performance:")
    report_dict = test_results['classification_report']
    for class_name in train_dataset.class_to_idx.keys():
        if class_name in report_dict:
            metrics = report_dict[class_name]
            print(f"Class: {class_name}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1-score']:.4f}")
            print(f"  Support: {metrics['support']}")
    
    # Save results
    with open('evaluation_results.json', 'w') as f:
        # Convert numpy objects to Python native types
        results = {
            'accuracy': float(test_results['accuracy']),
            'precision': float(test_results['precision']),
            'recall': float(test_results['recall']),
            'f1_score': float(test_results['f1_score']),
            'classification_report': test_results['classification_report']
        }
        json.dump(results, f, indent=4)
    
    print("\nModel training and evaluation complete! Results saved to evaluation_results.json")

if __name__ == "__main__":
    main()
