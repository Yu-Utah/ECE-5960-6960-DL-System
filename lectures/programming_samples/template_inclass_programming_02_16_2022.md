```python
# import libraries
import torch
import numpy as np
import torch.nn as nn
```


```python
from torchvision import datasets
import torchvision.transforms as transforms

# how many samples per batch to load
batch_size = 512

# convert data to torch.FloatTensor
transform = transforms.ToTensor()

#transform=transforms.Compose([transforms.ToTensor(),
#                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                             ])

# choose the training and test datasets
train_data = datasets.MNIST(root='data', train=True,
                                   download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False,
                                  download=True, transform=transform)

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

criterion = nn.CrossEntropyLoss()
```


```python
import torch.nn as nn
import torch.nn.functional as F

## Define the NN architecture
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear( 28*28 , 512 ) 
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear( 512, 10)
        # linear layer (n_hidden -> ?)
        # self.fc3 = nn.Linear(,)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, 28*28) 
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# initialize the NN
model_mlp = MLP().cuda()
print(model_mlp)
for parameter in model_mlp.parameters():
    print(parameter.shape)
```

    MLP(
      (fc1): Linear(in_features=784, out_features=512, bias=True)
      (fc2): Linear(in_features=512, out_features=10, bias=True)
    )
    torch.Size([512, 784])
    torch.Size([512])
    torch.Size([10, 512])
    torch.Size([10])



```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(in_channels= ,out_channels= , kernel_size= ,stride= ,padding= ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size= ),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d( ),     
            nn.ReLU(),                      
            nn.MaxPool2d( ),                
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear( , 10)
    def forward(self, x):
 
        return output, x    # return x for visualization

# initialize the NN
model_cnn = CNN().cuda()
print(model_cnn)
pcount = 0
for parameter in model_cnn.parameters():
    print(parameter.shape)
```


```python
# training code
def train(model, optimizer, epochs=10):
    model.train() # prep model for training

    for epoch in range(n_epochs):
        # monitor training loss
        train_loss = 0.0
        
        ###################
        # train the model #
        ###################
        for data, target in train_loader:
            data = data.cuda()
            target = target.cuda()
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)[0]
            #print(output, data.shape)
            # calculate the loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item()*data.size(0)
            
        # print training statistics 
        # calculate average loss over an epoch
        train_loss = train_loss/len(train_loader.dataset)

        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch+1, 
            train_loss
            ))
```


```python
# initialize lists to monitor test loss and accuracy
def test(model):
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    model.eval() # prep model for *evaluation*

    for data, target in test_loader:
        data = data.cuda()
        target = target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)[0]
        # calculate the loss
        loss = criterion(output, target)
        # update test loss 
        test_loss += loss.item()*data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        # calculate test accuracy for each object class
        for i in range(data.shape[0]):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # calculate and print avg test loss
    test_loss = test_loss/len(test_loader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(10):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                str(i), 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))
```


```python
# number of epochs to train the model
n_epochs = 10  # suggest training between 20-50 epochs
# specify optimizer
#model = Net()
optimizer = torch.optim.SGD(model_cnn.parameters(), lr=0.05)
train(model_cnn, optimizer)
```


```python
test(model_cnn)
```


```python
!nvcc --version
```

    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2020 NVIDIA Corporation
    Built on Mon_Oct_12_20:09:46_PDT_2020
    Cuda compilation tools, release 11.1, V11.1.105
    Build cuda_11.1.TC455_06.29190527_0



```python
class CNN_BN(nn.Module):
    def __init__(self):
        super(CNN_BN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d( ),                              
            nn.ReLU6(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d( ),     
            nn.ReLU6(),                      
            nn.MaxPool2d(2),                
        )
        # fully connected layer, output 10 classes
        self.bn1 = nn.BatchNorm1d( , affine=False)
        self.out = nn.Linear( , 10)
    def forward(self, x):
        ...
        return output, x    # return x for visualization

 
```


```python
# number of epochs to train the model
n_epochs = 10  # suggest training between 20-50 epochs
# specify optimizer
#model = Net()
# re-initialize the NN
model_cnn3 = CNN_BN().cuda()
print(model_cnn3)

optimizer = torch.optim.Adam(model_cnn3.parameters(), lr=0.02)
train(model_cnn3, optimizer)
```


```python
test(model_cnn3)
```


```python
# number of epochs to train the model
n_epochs = 10  # suggest training between 20-50 epochs
# specify optimizer
#model = Net()
# re-initialize the NN
model_cnn3 = CNN_BN().cuda()
print(model_cnn3)

optimizer = torch.optim.Adam(model_cnn3.parameters(), lr=0.015)
train(model_cnn3, optimizer)
test(model_cnn3)
```
