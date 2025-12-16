import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ==========================================
# Step 1: Import Libraries
# ==========================================
print("=" * 70)
print("🚀 Starting the Advanced Deep Learning Pipeline (Tuned Version)...")
print("=" * 70)

# ==========================================
# 1. Configuration & Data Preparation
# ==========================================
CLASSES = ['Decidual_tissue', 'Hemorrhage', 'Chorionic_villi', 'Trophoblastic_tissue']
NUM_CLASSES = len(CLASSES)

# -- Hyperparameter Tuning --
BATCH_SIZE = 32           # Keeps training stable
LEARNING_RATE = 1e-3      # Lower LR for fine-tuning the last block
NUM_EPOCHS = 20           # 20 epochs for Cosine Annealing to work effectively
WEIGHT_DECAY = 1e-4       # Regularization to prevent overfitting

# Set device (GPU/MPS/CPU)
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("✅ Using MPS (Apple Silicon) Acceleration")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("✅ Using CUDA (NVIDIA GPU) Acceleration")
else:
    DEVICE = torch.device("cpu")
    print("⚠️ Using CPU (No GPU detected)")

# Define Root Directory
DATA_ROOT = os.path.join(
    os.path.expanduser("~"), 
    "Desktop", 
    "computer_vision", 
    "computer_vision_cnn", 
    "POC_Dataset"
)

if not os.path.exists(DATA_ROOT):
    print(f"❌ Error: Dataset path not found: {DATA_ROOT}")
    exit()

# ==========================================
# Step 2: Create the Custom Dataset Class
# ==========================================
class CustomDataset(Dataset):
    """
    The main task of the Dataset class is to return a pair of [input, label] every time it is called.
    Strictly follows the requested 5-step preprocessing logic.
    """
    def __init__(self, root_dir, data_type="Training", transform=None):
        """
        __init__: Loads image files, stores file paths and labels using PIL/OS logic.
        """
        self.root_dir = root_dir
        self.data_type = data_type
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(CLASSES)}
        
        target_dir = os.path.join(self.root_dir, self.data_type)
        
        # Load file paths
        for class_name in CLASSES:
            class_dir = os.path.join(target_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for file_name in os.listdir(class_dir):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                    self.image_paths.append(os.path.join(class_dir, file_name))
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        """
        __len__: Returns the total number of images (length of the dataset).
        """
        return len(self.image_paths)

    def _preprocess_contour(self, img_path):
        """
        Implements the REQUIRED 5-step preprocessing:
        Step 1: Convert to binary image.
        Step 2: Apply morphological operations (erosion/dilation).
        Step 3: Select the largest contour & calculate extreme points.
        Step 4: Crop the image using extreme points.
        Step 5: Resize to 224x224 (This is handled in __getitem__ via transforms).
        """
        try:
            # Read image
            img = cv2.imread(img_path)
            if img is None: return None
            
            # Convert to Grayscale for thresholding
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            # Step 1: Convert to binary image
            thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]

            # Step 2: Morphological operations (remove noise)
            thresh = cv2.erode(thresh, None, iterations=2)
            thresh = cv2.dilate(thresh, None, iterations=2)

            # Step 3: Select largest contour and find extreme points
            cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not cnts:
                # Fallback: Return original if no contour found
                return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                
            c = max(cnts, key=cv2.contourArea)
            
            # Calculate extreme points (Top, Bottom, Left, Right)
            extLeft = tuple(c[c[:, :, 0].argmin()][0])
            extRight = tuple(c[c[:, :, 0].argmax()][0])
            extTop = tuple(c[c[:, :, 1].argmin()][0])
            extBot = tuple(c[c[:, :, 1].argmax()][0])

            # Step 4: Crop the image using the extreme points
            new_img = img[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
            
            # Convert to PIL format for PyTorch
            if new_img.size == 0:
                 return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            return Image.fromarray(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))

        except Exception as e:
            try:
                # Emergency fallback
                return Image.open(img_path).convert('RGB')
            except:
                return None

    def __getitem__(self, idx):
        """
        __getitem__: Retrieves one training example.
        Applies Step 1-4 (Contour Preprocessing) -> Step 5 (Resize & Transform).
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Apply preprocessing (Steps 1-4)
        image = self._preprocess_contour(img_path)
        
        if image is None:
            image = torch.zeros((3, 224, 224))
        
        # Apply transforms (Step 5: Resize happens here)
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)

# ==========================================
# Step 3: Instantiate Dataset and DataLoader
# ==========================================
print("\n[Data Preparation] Configuring Augmentations and DataLoaders...")

# Transforms: Tuning for better generalization
# Using Bicubic interpolation as strictly required.
train_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),    # Tissues can be rotated arbitrarily
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1), # Enhance robustness
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = CustomDataset(root_dir=DATA_ROOT, data_type="Training", transform=train_transform)
test_dataset = CustomDataset(root_dir=DATA_ROOT, data_type="Testing", transform=test_transform)

num_workers = 0

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

print(f"✅ Training Samples: {len(train_dataset)}")
print(f"✅ Testing Samples: {len(test_dataset)}")

# ==========================================
# Model Definition
# ==========================================
print("\n[Model Definition] Initializing DenseNet121 (Fine-Tuning Strategy)...")

def get_densenet_model(num_classes):
    """
    Model Definition:
    We use DenseNet121 pretrained on ImageNet.
    STRATEGY: We unfreeze the last few layers ('denseblock4') to adapt to Tissue features.
    """
    weights = models.DenseNet121_Weights.DEFAULT
    model = models.densenet121(weights=weights)
    
    # 1. Freeze Initial Layers (Low-level features like edges/blobs are generic)
    for param in model.parameters():
        param.requires_grad = False
        
    # 2. Unfreeze the Last DenseBlock (High-level features specific to Tissue)
    # This allows the model to learn tissue textures while keeping base stability.
    for param in model.features.denseblock4.parameters():
        param.requires_grad = True
    for param in model.features.norm5.parameters():
        param.requires_grad = True
    
    # 3. Modify Classifier
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),  # Dropout to reduce overfitting
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    return model

model = get_densenet_model(NUM_CLASSES).to(DEVICE)

# Loss Function & Optimizer
criterion = nn.CrossEntropyLoss()

# IMPROVEMENT: Using AdamW (Adam with Weight Decay) for better generalization
# We verify that we are passing parameters that require gradients (the unfrozen ones)
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                        lr=LEARNING_RATE, 
                        weight_decay=WEIGHT_DECAY)

# IMPROVEMENT: Using CosineAnnealingLR for smoother convergence
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

# ==========================================
# Training & Evaluation Loops
# ==========================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Training Loop: Iterate epochs -> Iterate batches -> Forward -> Backward -> Update
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc="Training", unit="batch")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward Pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward Pass & Optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion, device):
    """
    Evaluation: Calculate Accuracy, Precision, Recall, F1 on Testing Set.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    epoch_loss = running_loss / len(loader.dataset)
    
    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    # Weighted average is best for imbalanced datasets
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return epoch_loss, acc, precision, recall, f1

# ==========================================
# Main Execution Pipeline
# ==========================================
if __name__ == '__main__':
    best_f1 = 0.0
    save_path = "best_densenet_model.pth"

    print("\n[Pipeline] Starting Training Loop...")
    print("-" * 80)

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n📅 Epoch {epoch+1}/{NUM_EPOCHS} | LR: {current_lr:.6f}")
        
        # 1. Train Loop
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        
        # 2. Evaluation Loop
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(model, test_loader, criterion, DEVICE)
        
        # 3. Update Scheduler
        scheduler.step()
        
        duration = time.time() - start_time
        
        print(f"   [Train] Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"   [Test ] Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | F1: {val_f1:.4f}")
        print(f"   ⏳ Time: {duration:.1f}s")
        
        # Save Model (Validation F1 Score is the standard for checking performance)
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), save_path)
            print(f"   💾 Best Model Saved! (New Best F1: {best_f1:.4f})")

    print("-" * 80)
    print(f"🎉 Final Best F1-Score: {best_f1:.4f}")
    print("   Pipeline executed with Advanced Fine-Tuning.")