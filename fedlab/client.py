import torchvision.transforms as transforms
import torch
import argparse
import sys
import os
from torch import nn
import torchvision.datasets as datasets

sys.path.append("../../")
from fedlab.core.client.manager import ActiveClientManager
from fedlab.core.client.trainer import SGDClientTrainer
from fedlab.utils.dataset.sampler import RawPartitionSampler
from fedlab.core.network import DistNetwork
from fedlab.utils.functional import AverageMeter, evaluate



# torch model
class MLP(nn.Module):

    def __init__(self, input_size=784, output_size=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class AsyncClientTrainer(SGDClientTrainer):

    def __init__(self,
                 model,
                 data_loader,
                 epochs,
                 optimizer,
                 criterion,
                 cuda=False,
                 logger=None):
        super().__init__(model, data_loader, epochs, optimizer, criterion,
                         cuda, logger)
        self.time = 0

    def local_process(self, payload):
        self.time = payload[1].item()
        return super().local_process(payload)

    @property
    def uplink_package(self):
        return [self.model_parameters, torch.Tensor([self.time])]


parser = argparse.ArgumentParser(description='Distbelief training example')
parser.add_argument('--ip', type=str, default='127.0.0.1')
parser.add_argument('--port', type=str, default='3002')
parser.add_argument('--world_size', type=int)
parser.add_argument('--rank', type=int)

parser.add_argument("--epoch", type=int, default=2)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--batch_size", type=int, default=100)
args = parser.parse_args()

if torch.cuda.is_available():
    args.cuda = True
else:
    args.cuda = False

data_dir = '../../dataset/'


classes = ['Common_rust', 'Gray_Leaf', 'Healthy', 'Northern_Leaf_Blight']

num_classes = 4
batch_size = 32
num_workers = 4

train_transforms = transforms.Compose([
                           transforms.Resize(size=[224, 224]),
                           #transforms.Resize(size=[299,299]),
                           transforms.RandomRotation([0,360]),
                           transforms.RandomVerticalFlip(),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                       ])

test_transforms = transforms.Compose([
                           transforms.Resize(size=[224, 224]),
                           transforms.ToTensor(),
                           transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                       ])

trains_dir = []
valids_dir = []
train_loaders = []
valid_loaders = []
test_loaders = []

folds = os.listdir(data_dir + '5-fold')
folds.sort()

all_size_train = []
all_size_valid = []

for i in folds:
    train_dir = os.path.join(data_dir + '5-fold/', i)
    valid_dir = os.path.join(data_dir + '5-fold/', i)
    test_dir = os.path.join(data_dir + '5-fold/', i)

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers)

    validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers)

    testloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size,
                                              shuffle=False, num_workers=num_workers)

    train_loaders.append(trainloader)
    valid_loaders.append(validloader)
    test_loaders.append(testloader)

    print("----------------------------------------------------------------------------------------")
    print(i)
    print('Num training images: ', len(train_data), trainloader)
    print('Num valid images: ', len(valid_data), validloader)
    print('Num test images: ', len(valid_data), (testloader))

    all_size_train.append(len(train_data))
    all_size_valid.append(len(valid_data))

print("----------------------------------------------------------------------------------------")
print("\n\n----------------------------------------------------------------------------------------")
print("Num train full size:", sum(all_size_train))
print("Num valid full size:", sum(all_size_valid))
print('Num test images: ', len(valid_data), (testloader))
print("Num full size (train+valid+test):", sum(all_size_train) + sum(all_size_valid) + len(valid_data))

model = MLP()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()
handler = AsyncClientTrainer(model,
                             trainloader,
                             epochs=args.epoch,
                             optimizer=optimizer,
                             criterion=criterion,
                             cuda=args.cuda)

network = DistNetwork(address=(args.ip, args.port),
                      world_size=args.world_size,
                      rank=args.rank)

Manager = ActiveClientManager(trainer=handler, network=network)
Manager.run()

print("Final Score Client: "+str(evaluate(model, criterion, testloader)))
