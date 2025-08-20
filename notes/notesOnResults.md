## Original Network before improvements

```
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
```

After running this for a while, here are the results of total_training_time.csv

|Number of Epochs|Total Time (seconds)|
|----|----|
2|169.17
3|231.49
4|309.55
5|380.28
6|658.38
7|538.84
8|914.91
9|979.25
10|742.55

The following are the results of the trainig of this net with 10 epochs

|Epoch|Time (seconds)|
|---|---|
1|73.08
2|71.4
3|74.78
4|75.13
5|74.66
6|73.62
7|75.44
8|76.61
9|74.84
10|73.0

## Current code: