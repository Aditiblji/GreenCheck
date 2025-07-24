

# Import necessary libraries
import torch
import torch.nn as nn
from torch.optim import Adam

from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchsummary import summary

# Basic Libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device} ')

'''
    Data Transformations
    Training: Agumentation and Normalization
'''
train_transforms = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(0.9),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.RandomAffine(90, (0.3, 0.3), (1.0, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

val_test_transforms = transforms.Compose([
                       transforms.Resize((224,224)),
                       transforms.ToTensor(),
                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



data_dir = 'Path to the folder'
dataset = datasets.ImageFolder(root=data_dir, transform=val_test_transforms)

# Lets get class names and number of class as we it should be 38 for Plant Village
class_names = dataset.classes
num_classes = len(class_names)
print(f'Number of classes: {num_classes}')
print(f'Classes: {class_names}')

# lets split Plant Village in Train, Val and Test
# Train -> 70%   Val -> 15%  Test -> 15%
indices = list(range(len(dataset)))
labels = [dataset.targets[i] for i in range(len(dataset))]

# 1st Split  Train(70%) + (Val+Test) (30%)
train_idx, temp_idx, train_labels, temp_labels = train_test_split(indices, labels, test_size=0.3, stratify=labels, random_state=42)

# 2nd Split  Val (15%) + Test (15%) from the 30%
val_idx, test_idx, val_labels, test_labels = train_test_split(temp_idx, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42)

# Now lets create subset of Village Datasets
train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)
test_dataset = Subset(dataset, test_idx)

# Apply train transforms to training dataset
train_dataset.dataset.transform = train_transforms

# DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Now lets Visualize Class Distribution
class_counts = Counter(labels)
class_counts_df = pd.DataFrame.from_dict(class_counts, orient='index', columns=['count'])
class_counts_df['class'] = [class_names[i] for i in class_counts_df.index]
class_counts_df = class_counts_df.sort_values('count', ascending=False)

plt.figure(figsize=(12,6))
sns.barplot(data=class_counts_df, x='class', y='count')
plt.title("Class Distribution in Plant Village Dataset")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
plt.savefig('plnt_vlg_class_distribution.png')
plt.close()

class GreenCheck(nn.Module):
    def __init__(self, num_classes):
        super(GreenCheck, self).__init__()
        # convolution layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*28*28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # forward pass
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = GreenCheck(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

num_epochs = 10
train_losses, val_losses = [], []
train_accurates, val_accurates = [], []

for epoch in range(num_epochs):
    # Here we train the model
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    train_losses.append(train_loss)
    train_accurates.append(train_acc)


    # Lets perform Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            print("Loaded a batch of images and labels.")
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_loss = val_loss / len(val_loader)
    val_acc = 100 * correct / total
    val_losses.append(val_loss)
    val_accurates.append(val_acc)

    print(f'Epoch [{epoch+1}/{num_epochs}]')
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

# Plot Training Metrics
epochs = range(1, num_epochs+1)
plt.figure(figsize=(12,5))

# Loss Plot
plt.subplot(1,2,1)
sns.lineplot(x=epochs, y=train_losses, label='Train Loss')
sns.lineplot(x=epochs, y=val_losses, label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Accuracy Plot
plt.subplot(1, 2, 2)
sns.lineplot(x=epochs, y=train_accurates, label='Train Acc')
sns.lineplot(x=epochs, y=val_accurates, label='Val Acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')

plt.tight_layout()
plt.show()
plt.savefig('training_metrics.png')
plt.close()

#  Test Evaluation
model.eval()
test_correct = 0
test_total = 0
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = 100 * test_correct / test_total
print(f'Test Accuracy: {test_acc:.2f}%')

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
plt.savefig('confusion_matrix.png')
plt.close()

# Classification Report
print('\nClassification Report:')
print(classification_report(all_labels, all_preds, target_names=class_names))