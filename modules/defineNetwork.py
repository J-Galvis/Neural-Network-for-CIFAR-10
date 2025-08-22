import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import time
import csv
import os

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
                                        
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)    # 32x32x3 -> 32x32x32
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)   # 32x32x32 -> 32x32x32
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)                # 32x32x32 -> 16x16x32
        
        # Second convolutional block
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)   # 16x16x32 -> 16x16x64
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)   # 16x16x64 -> 16x16x64
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)                # 16x16x64 -> 8x8x64
        
        # Third convolutional block
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)  # 8x8x64 -> 8x8x128
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1) # 8x8x128 -> 8x8x128
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)                # 8x8x128 -> 4x4x128
        
        # Global Average Pooling instead of large FC layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1) # 4x4x128 -> 1x1x128
        
        # Smaller fully connected layers
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # First conv block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        
        # Second conv block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        
        # Third conv block
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten: batch_size x 128
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

def trainNet(num_epochs: int):

    net = Net()
    criterion = nn.CrossEntropyLoss()
    
    # IMPROVEMENT 1: Better learning rate and weight decay
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    
    # IMPROVEMENT 2: Learning rate scheduler for better convergence
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    print("Starting training...")
    total_start_time = time.time()
    
    # List to store epoch times and metrics
    epoch_times = []
    
    for epoch in range(num_epochs):  
        epoch_start_time = time.time()
        running_loss = 0.0
        
        # IMPROVEMENT 3: Track training accuracy
        correct_train = 0
        total_train = 0
        
        # Set model to training mode
        net.train()
        
        for batch_idx, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            # IMPROVEMENT 4: Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)
        print(f'Epoch {epoch + 1}/{num_epochs} completed in {epoch_duration:.2f} seconds')

    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f'Total training time: {total_duration:.2f} seconds')
    print('Finished Training')
   
    # Save epoch times to CSV
    with open('./Results/epoch_times.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'Time (seconds)'])  # Header
        for i, epoch_time in enumerate(epoch_times):
            writer.writerow([i + 1, round(epoch_time, 2)])
    
    # Save total training time to CSV (append mode)
    total_training_file = './Results/total_training_time.csv'
    file_exists = os.path.exists(total_training_file)
   
    with open(total_training_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header only if file doesn't exist
        if not file_exists:
            writer.writerow(['Number of Epochs', 'Total Training Time (seconds)'])
        writer.writerow([len(epoch_times), round(total_duration, 2)])
   
    print('Times saved to epoch_times.csv and total_training_time.csv')
    torch.save(net.state_dict(), './Results/cifar10_trained_model.pth') #This saves the trained model


if __name__ == "__main__":
    trainNet(10)