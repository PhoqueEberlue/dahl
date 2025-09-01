from os import _exit
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

# For reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Setting up device for GPU usage if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}.")

# Define a transformation pipeline. 
# Here, we're only converting the images to PyTorch tensor format.
transform = transforms.Compose([transforms.ToTensor()])

# Using torchvision, load the Fashion MNIST training dataset.
# root specifies the directory where the dataset will be stored.
# train=True indicates that we want the training dataset.
# download=True will download the dataset if it's not present in the specified root directory.
# transform applies the defined transformations to the images.
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

# Create a data loader for the training set.
# It will provide batches of data, in this case, batches of size 4.
# shuffle=True ensures that the data is shuffled at the start of each epoch.
# num_workers=2 indicates that two subprocesses will be used for data loading.
trainloader = torch.utils.data.DataLoader(trainset, batch_size=2, shuffle=False, num_workers=1)

# Similarly, load the Fashion MNIST test dataset.
testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=2, shuffle=False, num_workers=1)

# Define the class labels for the Fashion MNIST dataset.
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=15, profile="full", sci_mode=False)

class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()
        # Input: [batch_size, 1, 28, 28]
        self.conv1 = nn.Conv2d(1, 4, 3)  # Output: [batch_size, 32, 26, 26]
        
        # Input: [batch_size, 32, 26, 26]
        # self.conv2 = nn.Conv2d(32, 64, 3) # Output: [batch_size, 64, 11, 11]
        
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(4 * 13 * 13, 10)  # Flattening: [batch_size, 64*5*5]
        # self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Input: [batch_size, 1, 28, 28]
        
        x = F.relu(self.conv1(x))

        # print(f"res conv: shape {x.shape}, {x}")

        # weights, biases = list(self.conv1.parameters())
        # print(f"Weights shape {weights.shape}\n biases shape: {len(biases)}")

        # Shape: [batch_size, 32, 26, 26]
        x = self.pool(x)
        # Shape: [batch_size, 32, 13, 13]
        

        # x = F.relu(self.conv2(x))
        # Shape: [batch_size, 64, 11, 11]
        # x = F.max_pool2d(x, 2)
        # Shape: [batch_size, 64, 5, 5]
        
        x = x.view(-1, 4 * 13 * 13) # Flattening

        weights, biases = list(self.fc1.parameters())
        # print(f"weigths value: {weights} {weights.shape}")
        # print(f"biases value: {biases} {biases.shape}")

        x = self.fc1(x)
        return F.softmax(x, dim=1)


model = BasicCNN().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Example: register hooks for intermediate activations and gradients
activations = {}
grads = {}
backward_inputs = {}


def save_activation(name):
    def hook(module, input, output):
        activations[name] = output.detach()
        output.retain_grad()  # keep gradient for this output
    return hook

def save_grad(name):
    def hook(grad):
        grads[name] = grad.detach()
    return hook

def save_backward_input(name):
    def hook(module, grad_input, grad_output):
        # grad_input: tuple of gradients w.r.t. the inputs of this layer
        # grad_output: tuple of gradients w.r.t. the outputs of this layer
        backward_inputs[name] = grad_input[0]  # grad w.r.t input
    return hook

# Register hooks on layers
model.conv1.register_forward_hook(save_activation("conv1"))
model.pool.register_forward_hook(save_activation("pool"))
model.fc1.register_forward_hook(save_activation("fc1"))

# Register backward hook
model.fc1.register_full_backward_hook(save_backward_input("fc1"))
model.pool.register_full_backward_hook(save_backward_input("pool"))
model.conv1.register_full_backward_hook(save_backward_input("conv1"))

# Number of complete passes through the dataset
num_epochs = 5

# Start the training loop
for epoch in range(num_epochs):
    # Set the model to training mode
    model.train()
    
    # Iterate over each batch of the training data
    for images, labels in trainloader:
        # Move the images and labels to the computational device (CPU or GPU)
        images, labels = images.to(device), labels.to(device)
        
        # Clear the gradients from the previous iteration
        optimizer.zero_grad()
        
        # Forward pass: Pass the images through the model to get the predicted outputs
        outputs = model(images)

        # Compute the loss between the predicted outputs and the true labels
        outputs.retain_grad()
        loss = criterion(outputs, labels)
        print(loss)

        # Backward pass: Compute the gradient of the loss w.r.t. model parameters
        loss.backward()

        # Save gradients for intermediate outputs
        # for name, act in activations.items():
        #     if act.grad is not None:
        #         grads[name] = act.grad.detach()
        
        # Print stuff (shapes here, to avoid huge prints)
        # print(f"Epoch {epoch} | Loss: {loss.item()}")
        # for name in activations:
        #     print(f"  {name} activation: {activations[name].shape}")
        #     if name in grads:
        #         print(f"  {name} grad: {grads[name].shape}")
        # print(outputs.grad)
        # print(f"{model.fc1.weight.grad} --- {model.fc1.weight.grad.shape}")

        optimizer.step()
        # print(f"Backward conv {backward_inputs["conv1"]} {backward_inputs["conv1"].shape}")
        
        # Update the model parameters

        print(f"weights: {model.conv1.weight.data} {model.conv1.weight.shape}")
        print(f"biasess: {model.conv1.bias.data} {model.conv1.bias.shape}")

        _exit(0);

    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Set the model to evaluation mode. This is important as certain layers like dropout behave differently during training and evaluation.
model.eval()

# Lists to store all predictions and true labels
all_preds = []
all_labels = []

# We don't want to compute gradients during evaluation, hence wrap the code inside torch.no_grad()
with torch.no_grad():
    # Iterate over all batches in the test loader
    for images, labels in testloader:
        # Transfer images and labels to the computational device (either CPU or GPU)
        images, labels = images.to(device), labels.to(device)
        
        # Pass the images through the model to get predictions
        outputs = model(images)
        
        # Get the class with the maximum probability as the predicted class
        _, predicted = torch.max(outputs, 1)
        
        # Extend the all_preds list with predictions from this batch
        all_preds.extend(predicted.cpu().numpy())
        
        # Extend the all_labels list with true labels from this batch
        all_labels.extend(labels.cpu().numpy())

# Print a classification report which provides an overview of the model's performance for each class
print(classification_report(all_labels, all_preds, target_names=classes))

# Compute the confusion matrix using true labels and predictions
cm = confusion_matrix(all_labels, all_preds)

# Visualize the confusion matrix using seaborn's heatmap
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", cmap=plt.cm.Blues, xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted Label')  # x-axis label
plt.ylabel('True Label')       # y-axis label
plt.title('Confusion Matrix')  # Title of the plot
plt.show()                     # Display the plot
