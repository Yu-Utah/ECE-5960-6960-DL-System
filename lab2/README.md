# LAB 2 -- Backpropogation, Computation Grpah, and hand-crafted MLP

## Q1-a: Construct the computation graph and perform at least two iterations of GD optimization. Show all your steps.

    1) function: f(x, w0, w1, b0, b1) = ( (x*w0) + b0 ) * w0 + b1
    2) initial values: w0=3, b0 = -1, w1 = 2, b1 = 2
    3) Data point: x = 1, y=5
    
## Q1-b: Any simplified GD process can be found ? (think about the sigmoid example covered in the class) Show all your steps.


## Q2: Construct a 2-layer Regression MLP with Sigmoid as hidden activation function. (Hint: what is your output ? what is the output loss that guides training?). 

## Requirements
    1) Your input data will have dim=32
    2) Your hidden size will be dim=64
    3) Use the code structure shown below to complete Q2.




```python
import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
```


```python
# randomly generate data
sample, D_in, n_neurons, D_out = 100, 16, 16, 1
# make sure to have a fixed random seed to have reproducible runs
np.random.seed(0)
```


```python
# random data points
x, y = randn(sample, D_in), randn(sample, D_out) # generate random data points (x => features, y => labels)
# print the first data in the dataset
print(x[0])#feature
print(y[0])#label
```

    [ 1.76405235  0.40015721  0.97873798  2.2408932   1.86755799 -0.97727788
      0.95008842 -0.15135721 -0.10321885  0.4105985   0.14404357  1.45427351
      0.76103773  0.12167502  0.44386323  0.33367433]
    [0.43077113]



```python
# Random initialization for weight param: D_in => n_neurons => D_out
w1, w2 = randn(D_in, n_neurons), randn(n_neurons, D_out)
```

```python
def mse_loss(delta):
    ... # to be filled

def sigmoid_act(x):
    ... # to be filled

def forward(w1,w2,x,y):
    ... #to be filled
    return loss,grad_y_pred,h

def evaluation(w1,w2,x,y):
    ... #to be filled
    return loss
    
def backward(w1,w2,h,x,grad_y_pred,lr=1e-4): 
    """
    w1, w2 : weights to be updated
    h      : hidden (forward value) used for backprop
    x      : input used for backprop
    grad_y_pred : output gradients
    lr     : learning rate
    """
    if not grad_y_pred.all():
        return w1,w2
    else:
        ... #to be filled
        return w1,w2
```

```python
# Test code for evaluating your implementation

batched=0 # batched = 0 => each epoch takes whoe training set 
          # batched = 1 => 50% in one epoch
for i in range(300): # loops 300 epochs
    if batched==1:
        loss,grad_y_pred,h = forward(w1,w2,x,y)
        w1,w2 = backward(w1,w2,h,x,grad_y_pred,lr=1e-4)
        print("%d-th iteration, mse = %.10f" % (i, evaluation(w1,w2,x,y)))
    else:
        x_first_half = x[0:50]
        y_first_half = y[0:50]
        loss,grad_y_pred,h = forward(w1,w2,x_first_half,y_first_half)
        w1,w2 = backward(w1,w2,h,x_first_half,grad_y_pred,lr=1e-4)
        
        x_second_half = x[50:]
        y_second_half = y[50:]
        loss,grad_y_pred,h = forward(w1,w2,x_second_half,y_second_half)
        w1,w2 = backward(w1,w2,h,x_second_half,grad_y_pred,lr=1e-4)
        print("%d-th iteration, mse = %.10f" % (i, evaluation(w1,w2,x,y)))
```

# Example output

    0-th iteration, mse = 12.0249009221
    1-th iteration, mse = 10.3846364600
    2-th iteration, mse = 9.0210579456
    3-th iteration, mse = 7.8867975424
    4-th iteration, mse = 6.9426745644
    5-th iteration, mse = 6.1562586891
    6-th iteration, mse = 5.5006919688
    7-th iteration, mse = 4.9537209843
    8-th iteration, mse = 4.4969002513
    9-th iteration, mse = 4.1149356071
    10-th iteration, mse = 3.7951423149
    11-th iteration, mse = 3.5269973816
    12-th iteration, mse = 3.3017693937
    13-th iteration, mse = 3.1122122301
    14-th iteration, mse = 2.9523114784
    15-th iteration, mse = 2.8170743829


```python

```
