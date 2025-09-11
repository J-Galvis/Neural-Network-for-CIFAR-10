import torch
import torchvision.datasets as datasets
from defineNetwork import Net, transform
import os
import csv

def testingNetwork( testloader, net):

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return (100 * correct / total)

def defaultTesting():

    print("starting testing . . . ")
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

    # Code to initialize testloader and net
    net = Net()
    net.load_state_dict(torch.load('./Results/cifar10_trained_model.pth'))
    testingNetwork(testloader, net)
    return

if __name__ == "__main__":

    print("starting testing . . . ")
    results_dir = './Results'
    net_time_file = os.path.join(results_dir, 'net_Times.csv')

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

    # Code to initialize testloader and net
    net = Net()
    net.load_state_dict(torch.load('./Results/cifar10_trained_model.pth'))
    AccuracyOfNet =testingNetwork(testloader, net)
    print('Accuracy of the network on the 10000 test images: %d %%' % AccuracyOfNet)

    # Read existing data
    existing_data = []
    if os.path.exists(net_time_file):
        with open(net_time_file, 'r', newline='') as f:
            reader = csv.reader(f)
            existing_data = list(reader)
    
    # Add accuracy to the last row or create new row if file is empty
    existing_data[-1].append(f"{AccuracyOfNet:.2f}")

    # Write back all data
    with open(net_time_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(existing_data)