import torch
from torch import nn
from torch.nn import functional as F


# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         self.conv1 = nn.Conv1d(12, 64, kernel_size=3, stride=1, padding=1)
#         self.relu1 = nn.ReLU()
#         self.pool1 = nn.MaxPool1d(kernel_size=1, stride=1)

#         self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
#         self.relu2 = nn.ReLU()
#         self.pool2 = nn.MaxPool1d(kernel_size=1, stride=1)

#         self.fc1 = nn.Linear(128, 128)
#         self.fc2 = nn.Linear(128, 4)

#         # Apply Xavier initialization to the weights of the linear layers
#         nn.init.xavier_uniform_(self.fc1.weight)
#         nn.init.xavier_uniform_(self.fc2.weight)

#         # init conv1d weights with xavier as well
#         nn.init.xavier_uniform_(self.conv1.weight)
#         nn.init.xavier_uniform_(self.conv2.weight)


#     def forward(self, x):
#         x = x.view(x.size(0), 12, -1)  # Reshape the input tensor
#         x = self.pool1(self.relu1(self.conv1(x)))
#         x = self.pool2(self.relu2(self.conv2(x)))
#         x = x.view(x.size(0), -1)  # Flatten the tensor
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x


# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=12, out_channels=64, kernel_size=5, stride=1, padding=2)
#         self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
#         # self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
#         self.pool = nn.MaxPool1d(kernel_size=1, stride=1, padding=0)
#         self.fc1 = nn.Linear(128, 512)
#         self.fc2 = nn.Linear(512, 128)
#         self.fc3 = nn.Linear(128, 4)  # 4 classes

#         nn.init.xavier_uniform_(self.fc1.weight)
#         nn.init.xavier_uniform_(self.fc2.weight)
#         nn.init.xavier_uniform_(self.fc3.weight)

#         #init conv1d weights with xavier as well
#         nn.init.xavier_uniform_(self.conv1.weight)
#         nn.init.xavier_uniform_(self.conv2.weight)

#     def forward(self, x):
#         # x = x.unsqueeze(0)  # Add an extra dimension
#         # x = x.permute(0, 2, 1)  # Swap the channels to the second dimension
#         x = x.view(x.size(0), 12, -1)  # Reshape the input tensor
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(x.size(0), -1)  # Flatten the tensor
#         # print(x.shape)
#         # x = torch.flatten(x, 1)  # Flatten all dimensions except the batch dimension
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv1d(12, 128, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=1, stride=1)

        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=1, stride=1)

        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 4)

        # Apply Xavier initialization to the weights of the linear layers
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

        # init conv1d weights with xavier as well
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, x):
        x = x.view(x.size(0), 12, -1)  # Reshape the input tensor
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
