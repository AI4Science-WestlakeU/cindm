
# coding: utf-8

# In[1]:


import numpy as np
import pprint as pp
from sklearn.model_selection import train_test_split
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data_utils
import matplotlib.pylab as plt
import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
from pytorch_net.net import MLP, ConvNet, load_model_dict, train, get_Layer
from pytorch_net.util import Early_Stopping, get_param_name_list, get_variable_name_list, standardize_symbolic_expression


# ### Apart from section 0, each section is independent on its own

# ## 0. Preparing dataset:

# In[2]:


# Preparing some toy dataset:
X = np.random.randn(1000,1)
y = X ** 2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
X_train = Variable(torch.FloatTensor(X_train))
y_train = Variable(torch.FloatTensor(y_train))
X_test = Variable(torch.FloatTensor(X_test))
y_test = Variable(torch.FloatTensor(y_test))


# ## 1. Constuct a simple MLP:

# In[3]:


# Constuct the network:
input_size = 1
struct_param = [
    [2, "Simple_Layer", {}],   # (number of neurons in each layer, layer_type, layer settings)
    [400, "Simple_Layer", {"activation": "relu"}],
    [1, "Simple_Layer", {"activation": "linear"}],
]
settings = {"activation": "relu"} # Default activation if the activation is not specified in "struct_param" in each layer.
                                    # If the activation is specified, it will overwrite this default settings.

net = MLP(input_size = input_size,
          struct_param = struct_param,
          settings = settings,
         )


# In[ ]:


# Get the prediction of the Net:
net(X_train)


# In[ ]:


# Get intermediate activation of the net:
net.inspect_operation(X_train,                   # Input
                      operation_between = (0,2), # Operation_between selects the subgraph. 
                                                 # Here (0, 2) means that the inputs feeds into layer_0 (first layer)
                                                 # and before the layer_2 (third layer).
                     )


# In[ ]:


# Save model:
pickle.dump(net.model_dict, open("net.p", "wb"))  # net.model_dict constains all the information (structural and parameters) of the network

# Load model:
net_loaded = load_model_dict(pickle.load(open("net.p", "rb")))

# Check the loaded net and the original net is identical:
net_loaded(X_train) - net(X_train)


# ## 2. Using symbolic layers, and simplification:
# ### 2.1 Constructing MLP consisting of symbolic layers:

# In[ ]:


# Construct the network:
model_dict = {
    "type": "MLP",
    "input_size": 4,
    "struct_param": [[2, "Symbolic_Layer", {"symbolic_expression": "[3 * x0 ** 2 + p0 * x1 * x2 + p1 * x3, 5 * x0 ** 2 + p2 * x1 + p3 * x3 * x2]"}],
                     [1, "Symbolic_Layer", {"symbolic_expression": "[3 * x0 ** 2 + p2 * x1]"}], 
                    ],
    # Here the optional "weights" sets up the initial values for the parameters. If not set, will initialize with N(0, 1):
    'weights': [{'p0': -1.3,
                 'p1': 1.0,
                 'p2': 2.3,
                 'p3': -0.4},
                {'p2': -1.5},
               ]
}
net = load_model_dict(model_dict)
pp.pprint(net.model_dict)
print("\nOutput:")
net(torch.rand(100, 4))


# ### 2.2 Simplification of an MLP from Simple_Layer to Symbolic_Layer:

# In[6]:


input_size = 1
struct_param = [
    [2, "Simple_Layer", {"activation": "relu"}],   # (number of neurons in each layer, layer_type, layer settings)
    [10, "Simple_Layer", {"activation": "linear"}],
    [1, "Simple_Layer", {"activation": "relu"}],
]

net = MLP(input_size = input_size,
          struct_param = struct_param,
          settings = {},
         )


# In[ ]:


net.simplify(X=X_train,
             y=y_train,
             mode=['to_symbolic'], 
             # The mode is a list of consecutive simplification methods, choosing from:
             # 'collapse_layers': collapse multiple Simple_Layer with linear activation into a single Simple_Layer; 
             # 'local': greedily try reducing the input dimension by removing input dimension from the beginning
             # 'snap': greedily snap each float parameter into an integer or rational number. Set argument 'snap_mode' == 'integer' or 'rational';
             # 'pair_snap': greedily trying if the ratio of a pair of parameters is an integer or rational number (by setting snap_mode)
             # 'activation_snap': snap the activation;
             # 'to_symbolic': transform the Simple_Layer into Symbolic_layer;
             # 'symbolic_simplification': collapse multiple layers of Symbolic_Layer into a single Symbolic_Layer;
             # 'ramping-L1': increasing L1 regularization for the parameters and train. When some parameter is below a threshold, snap it to 0.
            )


# ## 3. SuperNet:

# ### 3.1 Use SuperNet Layer in your own module:

# In[7]:


layer_dict = {
    "layer_type": "SuperNet_Layer",
    "input_size": 1,
    "output_size": 2,
    "settings": {
                 "W_available": ["dense", "Toeplitz"], # Weight type. Choose subset of "dense", "Toeplitz", "arithmetic-series-in", "arithmetic-series-out", "arithmetic-series-2D-in", "arithmetic-series-2D-out"
                 "b_available": ["dense", "None", "arithmetic-series", "constant"], # Bias type. Choose subsets of "None", "constant", "arithmetic-series", "arithmetic-series-2D"
                 "A_available": ["linear", "relu"], # Activation. Choose subset of "linear", "relu", "leakyRelu", "softplus", "sigmoid", "tanh", "selu", "elu", "softmax"
                }
}
SuperNet_Layer = get_Layer(**layer_dict)


# In[8]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer_0 = SuperNet_Layer
        self.layer_1 = nn.Linear(2,4)
    
    def forward(self, X):
        return self.layer_1(self.layer_0(X))


# In[ ]:


net = Net()
net(X_train)


# ### 3.2 Construct an MLP containing SuperNet layer:

# In[ ]:


input_size = 1
struct_param = [
    [6, "Simple_Layer", {}],   # (number of neurons in each layer, layer_type, layer settings)
    [4, "SuperNet_Layer", {"activation": "relu", # Choose from "linear", "relu", "leakyRelu", "softplus", "sigmoid", "tanh", "selu", "elu", "softmax"
                           "W_available": ["dense", "Toeplitz"],
                           "b_available": ["dense", "None", "arithmetic-series", "constant"],
                           "A_available": ["linear", "relu"],
                          }],
    [1, "Simple_Layer", {"activation": "linear"}],
]
settings = {"activation": "relu"} # Default activation if the activation is not specified in "struct_param" in each layer.
                                    # If the activation is specified, it will overwrite this default settings.

net = MLP(input_size = input_size,
          struct_param = struct_param,
          settings = settings,
         )
net(X_train)


# ## 4. Training using explicit commands:

# In[34]:


# training settings:
batch_size = 128
epochs = 500

# Prepare training set batches:
dataset_train = data_utils.TensorDataset(X_train.data, y_train.data)   #  The data_loader must use the torch Tensor, not Variable. So I use X_train.data to get the Tensor.
train_loader = data_utils.DataLoader(dataset_train, batch_size = batch_size, shuffle = True)

# Set up optimizer:
optimizer = optim.Adam(net.parameters(), lr = 1e-3)
# Get loss function. Choose from "mse" or "huber", etc.
criterion = nn.MSELoss()
# Set up early stopping. If the validation loss does not go down after "patience" number of epochs, early stop.
early_stopping = Early_Stopping(patience = 10) 


# In[ ]:


to_stop = False
for epoch in range(epochs):
    for batch_id, (X_batch, y_batch) in enumerate(train_loader):
        # Every learning step must contain the following 5 steps:
        optimizer.zero_grad()   # Zero out the gradient buffer
        pred = net(Variable(X_batch))   # Obtain network's prediction
        loss_train = criterion(pred, Variable(y_batch))  # Calculate the loss
        loss_train.backward()    # Perform backward step on the loss to calculate the gradient for each parameter
        optimizer.step()         # Use the optimizer to perform one step of parameter update
        
    # Validation at the end of each epoch:
    loss_test = criterion(net(X_test), y_test)
    to_stop = early_stopping.monitor(loss_test.item())
    print("epoch {0} \tbatch {1} \tloss_train: {2:.6f}\tloss_test: {3:.6f}".format(epoch, batch_id, loss_train.item(), loss_test.item()))
    if to_stop:
        print("Early stopping at epoch {0}".format(epoch))
        break


# In[6]:


# Save model:
pickle.dump(net.model_dict, open("net.p", "wb"))

# Load model:
net_loaded = load_model_dict(pickle.load(open("net.p", "rb")))

# Check the loaded net and the original net is identical:
net_loaded(X_train) - net(X_train)


# ## 5. Advanced example: training MNIST using given train() function:

# In[ ]:


from torchvision import datasets
import torch.utils.data as data_utils
from pytorch_net.util import train_test_split, normalize_tensor, to_Variable
is_cuda = torch.cuda.is_available()

struct_param_conv = [
    [64, "Conv2d", {"kernel_size": 3, "padding": 1}],
    [64, "BatchNorm2d", {"activation": "relu"}],
    [None, "MaxPool2d", {"kernel_size": 2}],

    [64, "Conv2d", {"kernel_size": 3, "padding": 1}],
    [64, "BatchNorm2d", {"activation": "relu"}],
    [None, "MaxPool2d", {"kernel_size": 2}],

    [64, "Conv2d", {"kernel_size": 3, "padding": 1}],
    [64, "BatchNorm2d", {"activation": "relu"}],
    [None, "MaxPool2d", {"kernel_size": 2}],

    [64, "Conv2d", {"kernel_size": 3, "padding": 1}],
    [64, "BatchNorm2d", {"activation": "relu"}],
    [None, "MaxPool2d", {"kernel_size": 2}],
    [10, "Simple_Layer", {"layer_input_size": 64}],
]

model = ConvNet(input_channels = 1,
                struct_param = struct_param_conv,
                is_cuda = is_cuda,
               )


# In[ ]:


dataset_name = "MNIST"
batch_size = 128
dataset_raw = getattr(datasets, dataset_name)('datasets/{0}'.format(dataset_name), download = True)
X, y = to_Variable(dataset_raw.train_data.unsqueeze(1).float(), dataset_raw.train_labels, is_cuda = is_cuda)
X = normalize_tensor(X, new_range = (0, 1))
(X_train, y_train), (X_test, y_test) = train_test_split(X, y, test_size = 0.2) # Split into training and testing
dataset_train = data_utils.TensorDataset(X_train, y_train)
train_loader = data_utils.DataLoader(dataset_train, batch_size = batch_size, shuffle = True) # initialize the dataLoader


# In[ ]:


lr = 1e-3
reg_dict = {"weight": 1e-6, "bias": 1e-6}
loss_original, loss_value, data_record = train(model, 
                                               train_loader = train_loader,
                                               validation_data = (X_test, y_test),
                                               criterion = nn.CrossEntropyLoss(),
                                               lr = lr,
                                               reg_dict = reg_dict,
                                               epochs = 1000,
                                               isplot = True,
                                               patience = 40,
                                               scheduler_patience = 40,
                                               inspect_items = ["accuracy"],
                                               record_keys = ["accuracy"],
                                               inspect_interval = 1,
                                               inspect_items_interval = 1,
                                               inspect_loss_precision = 4,
                                              )


# ## 6. An example callback code:

# In[ ]:


if isplot:
    import matplotlib.pylab as plt
    fig = plt.figure()
    if dim == 3:
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig.add_axes([0, 0, 1, 1])
        ax = fig.axes[0]
else:
    ax = None


def visualize(model, X, y, iteration, loss, ax, dim):
    if isplot:
        plt.cla()
        y_pred = model.transform(X)
        y_pred, y = to_np_array(y_pred, y)
        if dim == 3:
            ax.scatter(y[:,0],  y[:,1], y[:,2], color='red', label='ref', s = 1)
            ax.scatter(y_pred[:,0],  y_pred[:,1], y_pred[:,2], color='blue', label='data', s = 1)
            ax.text2D(0.87, 0.92, 'Iteration: {:d}\nError: {:06.4f}'.format(iteration, loss), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
        else:
            ax.scatter(y[:,0], y[:,1], color='red', label='ref')
            ax.scatter(y_pred[:,0], y_pred[:,1], color='blue', label='data')
            plt.text(0.87, 0.92, 'Iteration: {:d}\nError: {:06.4f}'.format(iteration, loss), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
        ax.legend(loc='upper left', fontsize='x-large')
        plt.draw()
        plt.pause(pause)
callback = partial(visualize, ax = ax, dim = dim)

