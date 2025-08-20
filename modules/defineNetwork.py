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
                                        
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    print("Starting training...")
    total_start_time = time.time()
    
    # List to store epoch times
    epoch_times = []
    
    for epoch in range(10):  
        epoch_start_time = time.time()
        running_loss = 0.0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999: 
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)
        print(f'Epoch {epoch + 1} completed in {epoch_duration:.2f} seconds')

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