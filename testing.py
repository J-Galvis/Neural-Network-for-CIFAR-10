import torch
import torchvision.datasets as datasets
from defineNetwork import Net, transform

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

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    
    return

if __name__ == "__main__":

    print("starting tseting . . . ")

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
    
    # Code to initialize testloader and net
    net = Net()
    net.load_state_dict(torch.load('./cifar10_trained_model.pth'))

    testingNetwork(testloader, net)