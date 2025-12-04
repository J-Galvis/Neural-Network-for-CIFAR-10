import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from datasets import load_dataset
from PIL import Image
import platform
import torch
import time
import csv
import os

HOST = 'localhost'
PORT = 6000

NUM_WORKERS=0  # Disabled for ImageNet to avoid memory issues
NUM_EPOCHS=60
BATCH_SIZE=8   # Further reduced for ImageNet memory constraints

SAVE_FILE = './Results/imagenet_trained_model.pth'


# ImageNet standard preprocessing
TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom dataset class for HuggingFace ImageNet
class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Load ImageNet dataset from HuggingFace
# Login using e.g. `huggingface-cli login` to access this dataset
print("Loading ImageNet dataset from HuggingFace...")
ds = load_dataset("ILSVRC/imagenet-1k")
TRAINSET = ImageNetDataset(ds["train"], transform=TRANSFORM)
print(f"Successfully loaded ImageNet training set with {len(TRAINSET)} samples")

TRAINLOADER = torch.utils.data.DataLoader(TRAINSET, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=False, persistent_workers=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)    
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1, bias=False)   
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        
        # Second convolutional block
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, bias=False)   
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1, bias=False)  
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.25)
        
        # Third convolutional block
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1, bias=False)  
        self.bn5 = nn.BatchNorm2d( 256)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(0.25)
        
        # Fourth convolutional block
        self.conv7 = nn.Conv2d(256, 512, 3, padding=1, bias=False)  
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, 3, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(512)
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier for ImageNet (1000 classes)
        self.fc = nn.Linear(512, 1000)
        self.dropout = nn.Dropout(0.5)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # First conv block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second conv block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Third conv block
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Fourth conv block
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classifier
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    
def trainNet(num_epochs: int):

    num_workers = 0 if platform.system() == 'Windows' else os.cpu_count()  # Fix Windows multiprocessing shit 

    torch.set_num_threads(num_workers if num_workers > 0 else 1)

    # ImageNet preprocessing with data augmentation for training
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Loading ImageNet training dataset...")
    # Login using e.g. `huggingface-cli login` to access this dataset
    ds = load_dataset("ILSVRC/imagenet-1k")
    trainset = ImageNetDataset(ds["train"], transform=transform)
    print(f"Successfully loaded ImageNet training set with {len(trainset)} samples")

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=0, pin_memory=False, persistent_workers=False)

    net = Net()
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(net.parameters(), lr=0.001, weight_decay=1e-2, 
                           betas=(0.9, 0.999), eps=1e-8)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=0.01,
        epochs=num_epochs,
        steps_per_epoch=len(trainloader),
        pct_start=0.3,
        div_factor=10,
        final_div_factor=100
    )

    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Starting training...")
    total_start_time = time.time()
    
    epoch_times = []
    
    for epoch in range(num_epochs):  
        epoch_start_time = time.time()
        net.train()

        accumulation_steps = 4
        optimizer.zero_grad()
        
        for batch_idx, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss = loss / accumulation_steps 
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
            else:
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps
                
                loss.backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

        if len(trainloader) % accumulation_steps != 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)
        print(f'Epoch {epoch + 1}/{num_epochs} completed in {epoch_duration:.2f} seconds')

    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f'Total training time: {total_duration:.2f} seconds')
    print('Finished Training')
   
    with open('./Results/epoch_times.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'Time (seconds)'])  # Header
        for i, epoch_time in enumerate(epoch_times):
            writer.writerow([i + 1, round(epoch_time, 2)])
    
    total_training_file = './Results/total_training_time.csv'
    file_exists = os.path.exists(total_training_file)
   
    with open(total_training_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['Number of Epochs', 'Total Training Time (seconds)'])
        writer.writerow([len(epoch_times), round(total_duration, 2)])
   
    print('Times saved to epoch_times.csv and total_training_time.csv')
    torch.save(net.state_dict(), './Results/imagenet_trained_model.pth') #This saves the trained model


if __name__ == "__main__":
    trainNet(25)