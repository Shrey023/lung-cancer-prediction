import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter
import timm
import os
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from timm.data.mixup import Mixup

# Required for Windows multiprocessing
from torch.multiprocessing import freeze_support

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 7
SAVE_PATH = 'lung_cancer_model_convnext.pth'

# Data Augmentation
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset
data_dir = 'dataset'
image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), transform_train if x == 'train' else transform_val)
    for x in ['train', 'valid']
}

# Use drop_last=True to ensure batch size stays even (required for Mixup)
dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    for x in ['train', 'valid']
}

class_names = image_datasets['train'].classes
num_classes = len(class_names)

# Use ConvNeXt Base
model = timm.create_model('convnext_base.fb_in1k', pretrained=True)
model.head.fc = nn.Linear(model.head.fc.in_features, num_classes)
model = model.to(device)

# Loss & Optimizer
targets = np.array(image_datasets['train'].targets)
class_weights = compute_class_weight('balanced', classes=np.unique(targets), y=targets)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

mixup_args = dict(mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.5, num_classes=num_classes)
mixup_fn = Mixup(**mixup_args)

criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# Logging
os.makedirs("runs", exist_ok=True)
writer = SummaryWriter(log_dir=f"runs/convnext_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

# Metrics tracking
train_losses, valid_losses = [], []
train_top1, train_top5 = [], []
valid_top1, valid_top5 = [], []

def accuracy(output, target, topk=(1, 5)):
    """
    Computes top-k accuracy for classification.
    Handles both regular and mixup/cutmix labels.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    # If target is one-hot (from mixup), convert to class indices
    if target.dim() == 2:
        target = target.argmax(dim=1)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.expand_as(pred))

    results = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        results.append(correct_k.mul_(100.0 / batch_size))
    return results

def train_one_epoch(epoch):
    model.train()
    running_loss = 0.0
    top1_sum, top5_sum = 0.0, 0.0
    total = 0

    for inputs, labels in dataloaders['train']:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs, labels = mixup_fn(inputs, labels)  # Mixup applied here

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        acc1, acc5 = accuracy(outputs, labels, topk=(1, min(5, num_classes)))
        top1_sum += acc1.item()
        top5_sum += acc5.item()
        total += inputs.size(0)

    epoch_loss = running_loss / total
    epoch_acc1 = top1_sum / len(dataloaders['train'])
    epoch_acc5 = top5_sum / len(dataloaders['train'])

    writer.add_scalar("Loss/train", epoch_loss, epoch)
    writer.add_scalar("Accuracy/train_top1", epoch_acc1, epoch)
    writer.add_scalar("Accuracy/train_top5", epoch_acc5, epoch)

    print(f'Epoch {epoch+1} | Train Loss: {epoch_loss:.4f}, Top-1 Acc: {epoch_acc1:.2f}%, Top-5 Acc: {epoch_acc5:.2f}%')
    return epoch_loss, epoch_acc1


def validate(epoch):
    model.eval()
    running_loss = 0.0
    top1_sum, top5_sum = 0.0, 0.0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloaders['valid']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            acc1, acc5 = accuracy(outputs, labels, topk=(1, min(5, num_classes)))
            top1_sum += acc1.item()
            top5_sum += acc5.item()
            total += inputs.size(0)

    epoch_loss = running_loss / total
    epoch_acc1 = top1_sum / len(dataloaders['valid'])
    epoch_acc5 = top5_sum / len(dataloaders['valid'])

    writer.add_scalar("Loss/valid", epoch_loss, epoch)
    writer.add_scalar("Accuracy/valid_top1", epoch_acc1, epoch)
    writer.add_scalar("Accuracy/valid_top5", epoch_acc5, epoch)

    print(f'Epoch {epoch+1} | Valid Loss: {epoch_loss:.4f}, Top-1 Acc: {epoch_acc1:.2f}%, Top-5 Acc: {epoch_acc5:.2f}%')
    return epoch_loss, epoch_acc1


def plot_metrics(train_losses, valid_losses, train_top1, valid_top1):
    import matplotlib.pyplot as plt
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_top1, label='Training Accuracy')
    plt.plot(epochs, valid_top1, label='Validation Accuracy')
    plt.title('Top-1 Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, valid_losses, label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/convnext_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.close()


# Training loop
if __name__ == '__main__':
    freeze_support()

    best_valid_acc = 0.0
    early_stop_counter = 0
    best_epoch = 0

    try:
        for epoch in range(NUM_EPOCHS):
            train_loss, train_acc = train_one_epoch(epoch)
            val_loss, val_acc = validate(epoch)
            scheduler.step()

            # Save best model
            if val_acc > best_valid_acc:
                best_valid_acc = val_acc
                best_epoch = epoch
                torch.save(model.state_dict(), SAVE_PATH)

            # Early stopping
            if val_loss < float('inf'):
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

        print(f"\n✅ Best Validation Accuracy: {best_valid_acc:.2f}% at epoch {best_epoch+1}")
        plot_metrics(train_losses, valid_losses, train_top1, valid_top1)

    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user.")
    finally:
        writer.close()