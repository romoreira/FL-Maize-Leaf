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

    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = True

    def initialize_model(self, model_name, num_classes, feature_extract, use_pretrained=True):
        # Inicializando cada variável específica para cada modelo
        model_ft = None
        input_size = 0

        if model_name == "resnet":
            """ Resnet18
            """
            # model_ft = models.resnet18(pretrained=use_pretrained)
            model_ft = models.resnet34(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        else:
            print("Invalid model name, exiting...")
            exit()

        return model_ft, input_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distbelief training example')

    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=str, default='3002')
    parser.add_argument('--world_size', type=int)
    args = parser.parse_args()

    model = MLP()

    handler = AsyncParameterServerHandler(model, alpha=0.5, total_time=5)

    network = DistNetwork(address=(args.ip, args.port),
                          world_size=args.world_size,
                          rank=0)
    Manager = AsynchronousServerManager(handler=handler, network=network)

    Manager.run()

    root = "../../tests/data/mnist/"
    testset = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=16,
        drop_last=True)

    criterion = nn.CrossEntropyLoss()

    #print("Final Score Server: "+str(evaluate(model, criterion, testloader)))
