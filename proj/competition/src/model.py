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


# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         self.conv1 = nn.Conv1d(12, 6, kernel_size=3, stride=1, padding=1)
#         self.relu1 = nn.ReLU()
#         self.pool1 = nn.MaxPool1d(kernel_size=1, stride=1)

#         self.conv2 = nn.Conv1d(6, 4, kernel_size=3, stride=1, padding=1)
#         self.relu2 = nn.ReLU()
#         self.pool2 = nn.MaxPool1d(kernel_size=1, stride=1)

#         # self.conv3 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
#         # self.relu3 = nn.ReLU()
#         # self.pool3 = nn.MaxPool1d(kernel_size=1, stride=1)

#         self.fc1 = nn.Linear(256, 256)
#         self.fc2 = nn.Linear(4, 4)

#         # Apply Xavier initialization to the weights of the linear layers
#         # nn.init.xavier_uniform_(self.fc1.weight)
#         nn.init.xavier_uniform_(self.fc2.weight)

#         # init conv1d weights with xavier as well
#         nn.init.xavier_uniform_(self.conv1.weight)
#         nn.init.xavier_uniform_(self.conv2.weight)

#     def forward(self, x):
#         x = x.view(x.size(0), 12, -1)  # Reshape the input tensor
#         x = self.pool1(self.relu1(self.conv1(x)))
#         x = self.pool2(self.relu2(self.conv2(x)))
#         # x = self.pool3(self.relu3(self.conv3(x)))
#         x = x.view(x.size(0), -1)  # Flatten the tensor
#         # x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x


# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=6, out_channels=64, kernel_size=3, stride=1, padding=1)
#         self.pool1 = nn.MaxPool1d(kernel_size=1, stride=1)
#         self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
#         self.pool2 = nn.MaxPool1d(kernel_size=1, stride=1)

#         self.lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=2, batch_first=True)
#         self.fc1 = nn.Linear(256, 128)
#         self.fc2 = nn.Linear(128, 4)  # 4 classes
#         nn.init.xavier_uniform_(self.conv1.weight)
#         nn.init.xavier_uniform_(self.conv2.weight)
#         nn.init.xavier_uniform_(self.fc1.weight)
#         nn.init.xavier_uniform_(self.fc2.weight)


#     def forward(self, x):
#         # Convolutional layers
#         # x = x.permute(0, 2, 1)  # Swap to (batch, channels, seq_length)
#         x = x.view(x.size(0), 12, -1)  # Reshape the input tensor
#         # trim the input to 6 channels
#         x = x[:, :6, :]
#         x = self.pool1(F.relu(self.conv1(x)))
#         x = self.pool2(F.relu(self.conv2(x)))

#         # Preparing for LSTM layer
#         x = x.permute(0, 2, 1)  # Swap to (batch, seq_length, features) for LSTM
#         x, (hn, cn) = self.lstm(x)

#         # Only use the output of the last LSTM cell
#         x = x[:, -1, :]

#         # Fully connected layers
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x


# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()

#         # Convolutional layers
#         self.conv1 = nn.Conv1d(6, 32, kernel_size=5, padding=2)
#         self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
#         self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)

#         # Pooling layers
#         self.pool1 = nn.MaxPool1d(kernel_size=2)
#         self.pool2 = nn.MaxPool1d(kernel_size=2)
#         self.pool3 = nn.MaxPool1d(kernel_size=2)

#         # Fully connected layers
#         self.fc1 = nn.Linear(128 * 12, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, 4)  # Output for 4 classes

#         # Activation functions
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         # Reshape input to (batch_size, 6, sequence_length)
#         x = x.view(x.size(0), 6, -1)

#         # Convolutional layers
#         x = self.relu(self.conv1(x))
#         x = self.pool1(x)
#         x = self.relu(self.conv2(x))
#         x = self.pool2(x)
#         x = self.relu(self.conv3(x))
#         x = self.pool3(x)

#         # Flatten for fully connected layers
#         x = x.view(x.size(0), -1)

#         # Fully connected layers
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.fc3(x)

#         return x


### PAPER
# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         self.features = 12
#         # Block-1
#         self.conv1_1 = nn.Conv1d(in_channels=self.features, out_channels=16, kernel_size=3, stride=1, padding=1)
#         self.bn1_1 = nn.BatchNorm1d(num_features=16)
#         self.conv1_2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
#         self.bn1_2 = nn.BatchNorm1d(num_features=32)
#         self.pool1 = nn.MaxPool1d(kernel_size=1, stride=2, padding=0)

#         # Block-2
#         self.conv2_1 = nn.Conv1d(in_channels=32, out_channels=128, kernel_size=3, stride=1, padding=1)
#         self.bn2_1 = nn.BatchNorm1d(num_features=128)
#         self.conv2_2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
#         self.bn2_2 = nn.BatchNorm1d(num_features=256)
#         self.pool2 = nn.MaxPool1d(kernel_size=1, stride=2, padding=0)

#         # Block-3
#         self.conv3_1 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
#         self.bn3_1 = nn.BatchNorm1d(num_features=128)
#         self.conv3_2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
#         self.bn3_2 = nn.BatchNorm1d(num_features=256)
#         self.pool3 = nn.MaxPool1d(kernel_size=1, stride=2, padding=0)

#         # Dense Layers
#         self.fc1 = nn.Linear(256, 128)  # Adjust the input features to match output from block-3
#         self.bn4 = nn.BatchNorm1d(num_features=128)
#         self.fc2 = nn.Linear(128, 256)
#         self.bn5 = nn.BatchNorm1d(num_features=256)

#         self.lstm1 = nn.LSTM(input_size=256, hidden_size=100, batch_first=True)  # Adjust input_size to match output from block-3
#         self.lstm2 = nn.LSTM(input_size=100, hidden_size=100, batch_first=True)
#         # self.gru1 = nn.GRU(input_size=256, hidden_size=100, batch_first=True)  # Adjust input_size to match output from block-3
#         # self.gru2 = nn.GRU(input_size=100, hidden_size=100, batch_first=True)

#         # Classification Layer
#         self.classifier = nn.Linear(100, 4)

#         # init everything with xavier
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#             elif isinstance(m, nn.BatchNorm1d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         # Block-1
#         x = x.view(x.size(0), self.features, -1)  # Reshape the input tensor
#         # trim first 6
#         x = F.relu(self.bn1_1(self.conv1_1(x)))
#         x = F.relu(self.bn1_2(self.conv1_2(x)))
#         x = self.pool1(x)


#         # Block-2
#         x = F.relu(self.bn2_1(self.conv2_1(x)))
#         x = F.relu(self.bn2_2(self.conv2_2(x)))
#         x = self.pool2(x)

#         # Block-3
#         x = F.relu(self.bn3_1(self.conv3_1(x)))
#         x = F.relu(self.bn3_2(self.conv3_2(x)))
#         x = self.pool3(x)

#         # Flatten
#         x = torch.flatten(x, 1)

#         # Reshape for LSTM
#         x = x.view(x.size(0), 1, -1)

#         # LSTM Layers
#         x, _ = self.lstm1(x)
#         x, _ = self.lstm2(x)

#         # GRU Layers
#         # x, _ = self.gru1(x)
#         # x, _ = self.gru2(x)

#         # Reshape for Linear layer
#         x = x.view(x.size(0), -1)

#         # Classification Layer
#         x = F.log_softmax(self.classifier(x), dim=1)
#         return x


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # Hyperparameters (these need to be tuned for your specific problem)
        num_features = (
            6  # For example, 3 for gyro (x, y, z) and 3 for acceleration (x, y, z)
        )
        hidden_dim = 128
        num_layers = 2
        num_classes = 4  # Adjust for your specific task
        self.num_features = num_features  # Number of input features (e.g., x, y, z coordinates for gyro and acceleration)
        self.hidden_dim = hidden_dim  # Hidden dimension for LSTM
        self.num_layers = num_layers  # Number of LSTM layers

        # CNN layers
        self.conv1 = nn.Conv1d(
            in_channels=num_features,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # Apply Conv1D
        # Input shape: (batch_size, num_features, sequence_length)
        # Output shape: (batch_size, 64, sequence_length)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        # Reshaping output for LSTM input
        # Required shape: (batch_size, sequence_length, new_feature_size)
        x = x.permute(0, 2, 1)

        # LSTM
        # Output shape: (batch_size, sequence_length, hidden_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Selecting the last output (considering only the last time step)
        # Output shape: (batch_size, hidden_dim)
        final_output = lstm_out[:, -1, :]

        # Fully connected layer
        # Output shape: (batch_size, num_classes)
        out = self.fc(final_output)

        return out
