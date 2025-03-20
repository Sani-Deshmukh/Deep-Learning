import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes=10, input_size=28):  # Default to MNIST settings
        super(CNN, self).__init__()
        # TODO: Initialize layers
        # Use nn.Conv2d for the convolutional layer: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        # Use 4 filters with a filter size of 3 with padding 1 and stride 1
        # Note that the input images are grayscale
        # Use nn.Linear for the fully connected layer: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html

        self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1, stride=1)
        self.fc = nn.Linear(input_size * input_size * 4, num_classes)
        
    
    # TODO: Implement forward pass
    def forward(self, x):
        # pass through conv layer
        x = self.conv(x)

        # apply relu activation
        # Use F.relu
        x = F.relu(x)

        # flatten the output
        # Use torch.flatten
        x = torch.flatten(x, start_dim=1)

        # pass through fully connected layer
        x = self.fc(x)

        return x

class ManualCNN(nn.Module):
    def __init__(self, num_classes=10, input_size=28):
        super(ManualCNN, self).__init__()
        # TODO: Initialize layers same as before
        self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1, stride=1)
        self.fc = nn.Linear(input_size * input_size * 4, num_classes)

        # Set filters manually
        self.conv.weight.data = torch.tensor([
            # Filter 1
            [[
                [1, 0, -1], #tries to detect verticals
                [1, 0, -1],
                [1, 0, -1],
            ]],
            # Filter 2
            [[
                [0, 1, 0], #diagonals (extra importance to the center area)
                [1, -4, 1],
                [0, 1, 0],
            ]],
            # Filter 3
            [[
                [1, 1, 1], # horizontal edge detections 
                [0, 0, 0],
                [-1, -1, -1],
            ]],
            # Filter 4
            [[
                [0, -1, 0],
                [-1, 0, -1], 
                [0, -1, 0],
            ]],
        ], dtype=torch.float32)

        # Ensure that the conv weights and biases are not trainable                                        
        self.conv.weight.requires_grad = False
        self.conv.bias.requires_grad = False

    # TODO: Implement forward pass
    def forward(self, x):
        # pass through conv layer
        x = self.conv(x)

        # apply relu activation
        # Use F.relu
        x = F.relu(x)

        # flatten the output
        # Use torch.flatten
        x = torch.flatten(x, start_dim=1)

        # pass through fully connected layer
        x = self.fc(x)

        return x
    
    # path: results/manual_cnn/mnist/21-52-22_02-26-2025/manual_cnn.pth