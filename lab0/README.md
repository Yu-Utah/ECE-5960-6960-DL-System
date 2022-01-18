# LAB0 Warm-up - Virtual Python Env and Torch setup

===================
Installing on macOS
===================

### Download the installer:

   * `Miniconda installer for macOS <https://conda.io/miniconda.html>`_.

   * `Anaconda installer for macOS <https://www.anaconda.com/download/>`_.

### Install:

   * Miniconda---In your terminal window, run:



        bash Miniconda3-latest-MacOSX-x86_64.sh


### Follow the prompts on the installer screens.

   If you are unsure about any setting, accept the defaults. You
   can change them later.

### To make the changes take effect, close and then re-open your
   terminal window.

### Test your installation. In your terminal window or
   Anaconda Prompt, run the command ``conda list``. A list of installed packages appears
   if it has been installed correctly.
   
===================
Installing on Linux (only difference is the installer file download)
===================

### Download the installer:

   * `Miniconda installer for Linux <https://docs.conda.io/en/latest/miniconda.html#linux-installers>`_.

   * `Anaconda installer for Linux <https://www.anaconda.com/download/>`_.
 

### In your terminal window, run:

   * Miniconda:


        bash Miniconda3-latest-Linux-x86_64.sh

 
    
# Install PyTorch

   * Enter your virtual conda env and then 



        conda install pytorch torchvision torchaudio -c pytorch
        
        
# Test Your PyTorch setup

 
```python
import torch
import numpy as np
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")
```
   
You should see something like this (but second tensor will be random; try multiple times you will see the randomness)

```bash
Ones Tensor:
 tensor([[1, 1],
        [1, 1]])

Random Tensor:
 tensor([[0.4557, 0.7406],
        [0.5935, 0.1859]])
```
   

