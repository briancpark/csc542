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


# class ConvNet(nn.Module):
#     def __init__(self, dropout_rate=0.1):
#         super(ConvNet, self).__init__()
#         # Hyperparameters
#         num_features = 6
#         hidden_dim = 128
#         conv_hidden_dim = 64
#         lstm_conv_dim = 32
#         final_conv_dim = 16  # New convolutional layer dimension
#         num_layers = 2
#         num_classes = 4

#         self.dropout_rate = dropout_rate

#         # CNN layers
#         self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=conv_hidden_dim, kernel_size=5, stride=2, padding=2)
#         self.bn1 = nn.BatchNorm1d(conv_hidden_dim)
#         self.relu = nn.ReLU()
#         self.dropout1 = nn.Dropout(self.dropout_rate)
#         self.maxpool = nn.MaxPool1d(kernel_size=5, stride=1)

#         self.conv2 = nn.Conv1d(in_channels=conv_hidden_dim, out_channels=lstm_conv_dim, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm1d(lstm_conv_dim)
#         self.relu2 = nn.ReLU()
#         self.dropout2 = nn.Dropout(self.dropout_rate)
#         self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=1)

#         # New Conv3 layer and associated components
#         self.conv3 = nn.Conv1d(in_channels=lstm_conv_dim, out_channels=final_conv_dim, kernel_size=3, stride=1, padding=1)
#         self.bn3 = nn.BatchNorm1d(final_conv_dim)
#         self.relu3 = nn.ReLU()
#         self.dropout3 = nn.Dropout(self.dropout_rate)
#         self.maxpool3 = nn.MaxPool1d(kernel_size=2, stride=2)  # Adjust kernel_size and stride as needed


#         # LSTM layer with dropout
#         self.lstm = nn.LSTM(input_size=final_conv_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=self.dropout_rate if num_layers > 1 else 0)
#         self.dropout_lstm = nn.Dropout(self.dropout_rate)

#         # Fully connected layer
#         self.fc1 = nn.Linear(hidden_dim, 16)
#         self.fc2 = nn.Linear(16, num_classes)

#         # Xavier initialization
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu2(x)
#         x = self.maxpool2(x)

#         # New Conv3 layer forward pass
#         x = self.conv3(x)
#         x = self.bn3(x)
#         x = self.relu3(x)
#         x = self.maxpool3(x)

#         # Reshape for LSTM
#         x = x.permute(0, 2, 1)

#         # LSTM
#         lstm_out, (h_n, c_n) = self.lstm(x)
#         final_output = lstm_out[:, -1, :]

#         # Fully connected
#         x = self.fc1(final_output)
#         out = self.fc2(x)

#         return out


import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_dim)
        energies = self.fc(x)
        weights = F.softmax(energies.squeeze(-1), dim=1)
        # weights shape: (batch_size, seq_len)
        weights = weights.unsqueeze(1)
        # weights shape: (batch_size, 1, seq_len)
        # output shape: (batch_size, hidden_dim)
        output = torch.bmm(weights, x).squeeze(1)
        return output, weights


class ConvNet(nn.Module):
    def __init__(self, num_classes=4):
        super(ConvNet, self).__init__()
        # Hyperparameters
        num_features = (
            16  # For example, 3 for gyro (x, y, z) and 3 for acceleration (x, y, z)
        )
        num_filters = [16, 32, 64]  # Number of filters for each Conv1D layer
        hidden_dim = 200  # Hidden dimension for LSTM
        num_layers = 2  # Number of layers in LSTM
        stride = [1, 1, 1]
        dropout_rate = 0.2
        conv_dropout_rate = 0.1
        kernel_size = [7, 5, 3]
        padding = [4, 2, 1]
        dilation = [1, 1, 1]

        # Convolutional layers
        self.conv1 = nn.Conv1d(
            in_channels=num_features,
            out_channels=num_filters[0],
            kernel_size=kernel_size[0],
            stride=stride[0],
            padding=padding[0],
            dilation=dilation[0],
        )
        self.bn1 = nn.BatchNorm1d(num_filters[0])
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(
            in_channels=num_filters[0],
            out_channels=num_filters[1],
            kernel_size=kernel_size[1],
            stride=stride[1],
            padding=padding[1],
            dilation=dilation[1],
        )
        self.bn2 = nn.BatchNorm1d(num_filters[1])
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(
            in_channels=num_filters[1],
            out_channels=num_filters[2],
            kernel_size=kernel_size[2],
            stride=stride[2],
            padding=padding[2],
            dilation=dilation[2],
        )
        self.bn3 = nn.BatchNorm1d(num_filters[2])
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv1_dropout = nn.Dropout(conv_dropout_rate)
        self.conv2_dropout = nn.Dropout(conv_dropout_rate)
        self.conv3_dropout = nn.Dropout(conv_dropout_rate)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=num_filters[2],
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate,
        )

        # Attention layer
        self.attention = Attention(hidden_dim)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, num_classes)

        # init everything with xavier
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initialize hn and cn
        # hn = torch.zeros(2, x.size(0), 200).to(x.device)
        # cn = torch.zeros(2, x.size(0), 200).to(x.device)

        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv1_dropout(x)
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv2_dropout(x)
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv3_dropout(x)
        x = self.pool3(x)

        # Dropout layer
        x = self.dropout(x)

        # Preparing input for LSTM
        x = x.permute(0, 2, 1)  # Now x is of shape (batch_size, L, num_filters)

        # LSTM layer
        x, (hn, cn) = self.lstm(x)  # x is of shape (batch_size, seq_len, hidden_dim)

        # LSTM layer
        # x, (hn, cn) = self.lstm(x, (hn, cn))

        # Attention layer
        x, weights = self.attention(x)

        # Use the last output of the LSTM for classification
        # x = x[:, -1, :]

        # Fully connected layer
        x = self.fc(x)

        # No softmax here if using nn.CrossEntropyLoss in training, as it applies softmax internally
        return x
