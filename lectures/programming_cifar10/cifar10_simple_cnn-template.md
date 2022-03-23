```python
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
```


```python
 
# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
```

    CUDA is available!  Training on GPU ...



```python
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 20
# percentage of training set to use as validation
valid_size = 0.2

# convert data to a normalized torch.FloatTensor
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# choose the training and test datasets
train_data = datasets.CIFAR10('data', train=True,
                              download=True, transform=transform)
test_data = datasets.CIFAR10('data', train=False,
                             download=True, transform=transform)

# obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
    sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
    num_workers=num_workers)

# specify the image classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

```

    Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to data/cifar-10-python.tar.gz



      0%|          | 0/170498071 [00:00<?, ?it/s]


    Extracting data/cifar-10-python.tar.gz to data
    Files already downloaded and verified



```python
import matplotlib.pyplot as plt
%matplotlib inline

# helper function to un-normalize and display an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image
# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy() # convert images to numpy for display
# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
# display 20 images
for idx in np.arange(20):
  ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
  imshow(images[idx])
  ax.set_title(classes[labels[idx]])
```


    
![png](output_3_0.png)
    



```python
import torch.nn as nn
import torch.nn.functional as F
# define the CNN architecture

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
 
  def forward(self, x):
 
    return x
# create a complete CNN
model = Net()
print(model)
# move tensors to GPU if CUDA is available
if train_on_gpu:
  model.cuda()  
```

    Net(
      (conv1): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
      (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
      (fc1): Linear(in_features=400, out_features=120, bias=True)
      (fc2): Linear(in_features=120, out_features=84, bias=True)
      (fc3): Linear(in_features=84, out_features=10, bias=True)
    )



```python
import torch.optim as optim
# specify loss function
criterion = nn.CrossEntropyLoss()
# specify optimizer
optimizer = optim.SGD(model.parameters(), lr=.01)
```


```python
# number of epochs to train the model
n_epochs = 30
#List to store loss to visualize
train_losslist = []
valid_loss_min = np.Inf # track change in validation loss

for epoch in range(1, n_epochs+1):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    
    ###################
    # train the model #
    ###################
    model.train()
    for data, target in train_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()*data.size(0)
        
    ######################    
    # validate the model #
    ######################
    model.eval()
    for data, target in valid_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update average validation loss 
        valid_loss += loss.item()*data.size(0)
    
    # calculate average losses
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = valid_loss/len(valid_loader.dataset)
    train_losslist.append(train_loss)
        
    # print training/validation statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), 'model_cifar.pt')
        valid_loss_min = valid_loss

```

    Epoch: 1 	Training Loss: 1.755835 	Validation Loss: 0.382908
    Validation loss decreased (inf --> 0.382908).  Saving model ...
    Epoch: 2 	Training Loss: 1.399113 	Validation Loss: 0.317636
    Validation loss decreased (0.382908 --> 0.317636).  Saving model ...
    Epoch: 3 	Training Loss: 1.249966 	Validation Loss: 0.296770
    Validation loss decreased (0.317636 --> 0.296770).  Saving model ...
    Epoch: 4 	Training Loss: 1.168192 	Validation Loss: 0.284308
    Validation loss decreased (0.296770 --> 0.284308).  Saving model ...
    Epoch: 5 	Training Loss: 1.098089 	Validation Loss: 0.272896
    Validation loss decreased (0.284308 --> 0.272896).  Saving model ...
    Epoch: 6 	Training Loss: 1.039336 	Validation Loss: 0.254903
    Validation loss decreased (0.272896 --> 0.254903).  Saving model ...
    Epoch: 7 	Training Loss: 0.990966 	Validation Loss: 0.249208
    Validation loss decreased (0.254903 --> 0.249208).  Saving model ...
    Epoch: 8 	Training Loss: 0.946494 	Validation Loss: 0.240058
    Validation loss decreased (0.249208 --> 0.240058).  Saving model ...
    Epoch: 9 	Training Loss: 0.911885 	Validation Loss: 0.242715
    Epoch: 10 	Training Loss: 0.877549 	Validation Loss: 0.229736
    Validation loss decreased (0.240058 --> 0.229736).  Saving model ...
    Epoch: 11 	Training Loss: 0.843890 	Validation Loss: 0.233621
    Epoch: 12 	Training Loss: 0.815301 	Validation Loss: 0.233717
    Epoch: 13 	Training Loss: 0.785242 	Validation Loss: 0.224305
    Validation loss decreased (0.229736 --> 0.224305).  Saving model ...
    Epoch: 14 	Training Loss: 0.761783 	Validation Loss: 0.224723
    Epoch: 15 	Training Loss: 0.740285 	Validation Loss: 0.228966
    Epoch: 16 	Training Loss: 0.714905 	Validation Loss: 0.223246
    Validation loss decreased (0.224305 --> 0.223246).  Saving model ...
    Epoch: 17 	Training Loss: 0.691634 	Validation Loss: 0.224270
    Epoch: 18 	Training Loss: 0.668464 	Validation Loss: 0.230441
    Epoch: 19 	Training Loss: 0.647775 	Validation Loss: 0.230232
    Epoch: 20 	Training Loss: 0.624430 	Validation Loss: 0.225017
    Epoch: 21 	Training Loss: 0.606848 	Validation Loss: 0.231148
    Epoch: 22 	Training Loss: 0.585181 	Validation Loss: 0.232151
    Epoch: 23 	Training Loss: 0.565676 	Validation Loss: 0.232049
    Epoch: 24 	Training Loss: 0.543727 	Validation Loss: 0.235741
    Epoch: 25 	Training Loss: 0.525590 	Validation Loss: 0.250984
    Epoch: 26 	Training Loss: 0.510860 	Validation Loss: 0.241870
    Epoch: 27 	Training Loss: 0.491420 	Validation Loss: 0.253959
    Epoch: 28 	Training Loss: 0.474755 	Validation Loss: 0.261524
    Epoch: 29 	Training Loss: 0.454667 	Validation Loss: 0.264738
    Epoch: 30 	Training Loss: 0.442971 	Validation Loss: 0.278660



    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-11-f3a38475feb2> in <module>()
         63         torch.save(model.state_dict(), 'model_cifar.pt')
         64         valid_loss_min = valid_loss
    ---> 65 plt.plot(n_epochs, train_losslist)
         66 plt.xlabel("Epoch")
         67 plt.ylabel("Loss")


    /usr/local/lib/python3.7/dist-packages/matplotlib/pyplot.py in plot(scalex, scaley, data, *args, **kwargs)
       2761     return gca().plot(
       2762         *args, scalex=scalex, scaley=scaley, **({"data": data} if data
    -> 2763         is not None else {}), **kwargs)
       2764 
       2765 


    /usr/local/lib/python3.7/dist-packages/matplotlib/axes/_axes.py in plot(self, scalex, scaley, data, *args, **kwargs)
       1645         """
       1646         kwargs = cbook.normalize_kwargs(kwargs, mlines.Line2D)
    -> 1647         lines = [*self._get_lines(*args, data=data, **kwargs)]
       1648         for line in lines:
       1649             self.add_line(line)


    /usr/local/lib/python3.7/dist-packages/matplotlib/axes/_base.py in __call__(self, *args, **kwargs)
        214                 this += args[0],
        215                 args = args[1:]
    --> 216             yield from self._plot_args(this, kwargs)
        217 
        218     def get_next_color(self):


    /usr/local/lib/python3.7/dist-packages/matplotlib/axes/_base.py in _plot_args(self, tup, kwargs)
        340 
        341         if x.shape[0] != y.shape[0]:
    --> 342             raise ValueError(f"x and y must have same first dimension, but "
        343                              f"have shapes {x.shape} and {y.shape}")
        344         if x.ndim > 2 or y.ndim > 2:


    ValueError: x and y must have same first dimension, but have shapes (1,) and (30,)



    
![png](output_6_2.png)
    



```python
plt.plot(range(n_epochs), train_losslist)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Performance of Model 1")
plt.show()
```


    
![png](output_7_0.png)
    



```python
# track test loss
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model.eval()
# iterate over test data
for data, target in test_loader:
    # move tensors to GPU if CUDA is available
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # calculate the batch loss
    loss = criterion(output, target)
    # update test loss 
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)    
    # compare predictions to true label
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    # calculate test accuracy for each object class
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# average test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))
```

    Test Loss: 1.405041
    
    Test Accuracy of airplane: 71% (715/1000)
    Test Accuracy of automobile: 86% (860/1000)
    Test Accuracy of  bird: 49% (499/1000)
    Test Accuracy of   cat: 30% (301/1000)
    Test Accuracy of  deer: 50% (500/1000)
    Test Accuracy of   dog: 50% (502/1000)
    Test Accuracy of  frog: 78% (780/1000)
    Test Accuracy of horse: 63% (637/1000)
    Test Accuracy of  ship: 64% (648/1000)
    Test Accuracy of truck: 58% (580/1000)
    
    Test Accuracy (Overall): 60% (6022/10000)

