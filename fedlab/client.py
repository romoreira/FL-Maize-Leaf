import torchvision.transforms as transforms
import torch
import argparse
import sys
import os
from torch import nn
import torchvision.datasets as datasets
from torchvision import datasets, models, transforms

sys.path.append("../../")
from fedlab.core.client.manager import ActiveClientManager
from fedlab.core.client.trainer import SGDClientTrainer
from fedlab.utils.dataset.sampler import RawPartitionSampler
from fedlab.core.network import DistNetwork
from fedlab.utils.functional import AverageMeter, evaluate


"""
# torch model
class MLP(nn.Module):

    def __init__(self, input_size=150528, output_size=32):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
"""
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

parser.add_argument("--epoch", type=int, default=30)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--batch_size", type=int, default=4)
args = parser.parse_args()

if torch.cuda.is_available():
    args.cuda = True
else:
    args.cuda = False

data_dir = '../dataset/'


classes = ['Common_rust', 'Gray_Leaf', 'Healthy', 'Northern_Leaf_Blight']

num_classes = 4
batch_size = 4
num_workers = 4

train_transforms = transforms.Compose([
                           transforms.Resize(size=[224, 224]),
                           transforms.RandomVerticalFlip(0.5),
                           transforms.RandomRotation(30),
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

    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
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

#model = MLP()

model_ft = models.squeezenet1_0(pretrained=True)
for param in model_ft.parameters():
    param.requires_grad = True
model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
model_ft.num_classes = num_classes

optimizer = torch.optim.SGD(model_ft.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()
handler = AsyncClientTrainer(model_ft,
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

print("Final Score Client: "+str(evaluate(model_ft, criterion, testloader)))
