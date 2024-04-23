import torch
from torch import nn
from torch.nn import functional as F


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
        num_filters = [32, 16, 16]  # Number of filters for each Conv1D layer
        hidden_dim = 175  # Hidden dimension for LSTM
        hidden_dim_fc = 32  # Hidden dimension for fully connected layer
        num_layers = 3  # Number of layers in LSTM
        stride = [1, 1, 1]
        dropout_rate = 0.3
        conv_dropout_rate = 0.1
        kernel_size = [7, 5, 5]
        padding = [3, 2, 2]
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
