from cgitb import handler
import os
import sys
import argparse
import torch
from torch import nn
sys.path.append("../../")
from fedlab.core.network import DistNetwork
from fedlab.core.server.handler import AsyncParameterServerHandler
from fedlab.core.server.manager import AsynchronousServerManager
from fedlab.utils.functional import evaluate
from torchvision import datasets, models, transforms

import torchvision
import torchvision.transforms as transforms

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distbelief training example')

    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=str, default='3002')
    parser.add_argument('--world_size', type=int)
    args = parser.parse_args()

    #model = MLP()

    model_ft = models.squeezenet1_0(pretrained=True)
    for param in model_ft.parameters():
        param.requires_grad = True
    model_ft.classifier[1] = nn.Conv2d(512, 4, kernel_size=(1, 1), stride=(1, 1))
    model_ft.num_classes = 4

    handler = AsyncParameterServerHandler(model_ft, alpha=0.5, total_time=5)

    network = DistNetwork(address=(args.ip, args.port),
                          world_size=args.world_size,
                          rank=0)
    Manager = AsynchronousServerManager(handler=handler, network=network)

    Manager.run()



    batch_size = 16
    num_workers = 4
    test_transforms = transforms.Compose([
        transforms.Resize(size=[224, 224]),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    data_dir = '../../dataset/'
    classes = ['Common_rust', 'Gray_Leaf', 'Healthy', 'Northern_Leaf_Blight']
    test_dir = os.path.join(data_dir + '5-fold/')
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)




    criterion = nn.CrossEntropyLoss()

    print("Final Score Server: "+str(evaluate(model_ft, criterion, testloader)))
