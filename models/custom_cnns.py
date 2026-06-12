import torch.nn as nn

class CNN_3Layer(nn.Module):
    """
    A 3-Layer Convolutional Neural Network for 2D Medical Image Classification.
    Inputs: 224x224 RGB images.
    Outputs: Binary classification logits (Tumor vs. No Tumor).
    """
    def __init__(self):
        super(CNN_3Layer, self).__init__()
        
        # --- Layer 1: Conv Block ---
        # Input: 3 channels (RGB), Output: 16 filters, Kernel: 3x3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU() # Activation function
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Reduces dimensions by half
        
        # --- Layer 2: Conv Block ---
        # Input: 16 channels, Output: 32 filters, Kernel: 3x3
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # --- Layer 3: Conv Block ---
        # Input: 32 channels, Output: 64 filters, Kernel: 3x3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # --- Fully Connected (Dense) Layers ---
        # After three 2x2 poolings, a 224x224 image becomes 28x28.
        # 64 filters * 28 * 28 = 50,176 flattened features.
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(in_features=64 * 28 * 28, out_features=128) # 128 Neurons
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5) # 50% Dropout to prevent overfitting
        
        self.fc2 = nn.Linear(in_features=128, out_features=2) # 2 Neurons (Output Classes)

    def forward(self, x):
        # Pass input through the 3 convolutional layers
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        
        # Flatten and pass through dense layers
        x = self.flatten(x)
        x = self.dropout(self.relu4(self.fc1(x)))
        x = self.fc2(x)
        
        return x
