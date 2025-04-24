
import torch.nn as nn
import torch


class TwitterCNNModel(nn.Module):
    def __init__(self, d, nd, kernels):
        super(TwitterCNNModel, self).__init__()

        output_shape = [360, 359, 358]  # By experimentation
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=d, out_channels=nd, kernel_size=kernels[0]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=output_shape[0]),
            nn.Flatten()
        )
        self.conv1 = self.conv1.double()
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=d, out_channels=nd, kernel_size=kernels[1]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=output_shape[1]),
            nn.Flatten()
        )
        self.conv2 = self.conv2.double()
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=d, out_channels=nd, kernel_size=kernels[2]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=output_shape[2]),
            nn.Flatten()
        )
        self.conv3 = self.conv3.double()
        self.fcLayer = nn.Sequential(
            nn.Linear(3*nd, 120),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 2),
            nn.ReLU(),
            nn.Softmax(dim=1)
        )
        self.fcLayer = self.fcLayer.double()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv2(x)
        x = torch.concat([x1, x2, x3], dim=1)
        # eg:- tensor([[0.5321, 0.4679]], dtype=torch.float64, grad_fn=<SoftmaxBackward0>)
        x = self.fcLayer(x)

        return x
