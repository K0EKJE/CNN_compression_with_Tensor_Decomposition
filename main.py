'''Performing tensor decomposition on pre-trained model with PyTorch.'''

# Import necessary packages
import tensorly as tl
import tensorly

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import os
import time
import collections

# Import self-defined parser, dataset, modules, and helper functions
from models import *
from util.utils import progress_bar,check_folder
from tensor_decomp.decompositions import cp_decomposition_conv_layer, tucker_decomposition_conv_layer
from config import *
from util.dataset import *
from util.parser import *

# Determine the device to use for computation (GPU if available, else CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize the variable to track the best test accuracy
best_acc = 0
# Initialize the starting epoch (either 0 or the last checkpoint epoch)
start_epoch = 0

# If the 'add' argument is True, load a pre-trained model, else use the 'model'
if args.add:
    net = torch.load("/content/decomposed_model")
else:
    net = model

# Move the model to the GPU if 'cuda' is the selected device
net = net.cuda()
# Print the architecture of the neural network
print(net)

# If the device is 'cuda', enable cuDNN benchmarking for faster computation
if device == 'cuda':
    cudnn.benchmark = True

# Define the loss function (Cross-Entropy Loss)
criterion = nn.CrossEntropyLoss()
# Define the optimizer (Stochastic Gradient Descent)
optimizer = optim.SGD(net.parameters(), lr=learning_rate,
                      momentum=0.9, weight_decay=5e-4)
# Define a learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

history = {}
# Training
def train(epoch, decompose=False):
    """
    Trains the neural network for one epoch.

    Args:
        epoch (int): The current epoch number.
        decompose (bool): Whether to use decomposition during training.

    Returns:
        None

    This function trains the neural network for one epoch using the provided data loader.
    It computes the loss, updates the model's weights, and tracks training statistics
    such as loss and accuracy.

    If `decompose` is True, it stores the training history in a separate checkpoint folder.

    """
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    folder_path = 'checkpoint'
    if not os.path.exists(folder_path):
        # Create the checkpoint folder if it doesn't exist
        os.makedirs(folder_path)

    if decompose:
        folder_path = 'checkpoint_decomp'
        if not os.path.exists(folder_path):
            # Create the checkpoint folder if it doesn't exist
            os.makedirs(folder_path)

    # Loop over batches in the training data
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # Move inputs and targets to the GPU (if available)
        inputs, targets = inputs.cuda(), targets.cuda()
        # Zero the gradients to clear the accumulated gradients from the previous batch
        optimizer.zero_grad()
        # Forward pass: Compute the model's predictions
        outputs = net(inputs)
        # Calculate the loss between predictions and actual targets
        loss = criterion(outputs, targets)
        # Backpropagation: Compute gradients of the loss with respect to model parameters
        loss.backward()
        # Update model parameters using the computed gradients
        optimizer.step()

        # Update training statistics
        train_loss += loss.item()  # Accumulate the loss
        _, predicted = outputs.max(1)  # Get the predicted class labels
        total += targets.size(0)  # Update the total number of processed samples
        correct += predicted.eq(targets).sum().item()  # Count correct predictions

        # Display progress during training
        step_time, tot_time = progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    if not os.path.exists(folder_path + '/ckpt.pth'):
        # Create a new checkpoint file if it doesn't exist
        state = {}
        train_history = {}
        train_history['loss'] = []
        train_history['step time'] = []
        train_history['tot time'] = []
        train_history['acc'] = []
        test_history = {}
        test_history['loss'] = []
        test_history['step time'] = []
        test_history['tot time'] = []
        test_history['acc'] = []

        train_history['loss'].append(train_loss/(batch_idx+1))
        train_history['step time'].append(step_time)
        train_history['tot time'].append(tot_time)
        train_history['acc'].append(100.*correct/total)

        state['train_history'] = train_history
        state['test_history'] = test_history
        torch.save(state, folder_path+'/ckpt.pth')
    else:
        # Update the existing checkpoint file with training history
        state = torch.load(folder_path+'/ckpt.pth')  # for training to resume
        state['train_history']['loss'].append(train_loss/(batch_idx+1))
        state['train_history']['step time'].append(step_time)
        state['train_history']['tot time'].append(tot_time)
        state['train_history']['acc'].append(100.*correct/total)
        torch.save(state, folder_path+'/ckpt.pth')


def test1(epoch):
    """
    Evaluates the neural network on the test data for one epoch without fine-tuning.
    Intended to display the accuracy of current model, which has pretrained VGG19 as default.

    Args:
        epoch (int): The current epoch number.

    Returns:
        None

    This function evaluates the neural network's performance on the test dataset for one epoch
    without fine-tuning the network. It computes the loss, accuracy, and displays progress.
    """
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            step_time, tot_time = progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch, decompose=False):
    """
    Evaluates the neural network on the test data.

    Args:
        epoch (int): The current epoch number.
        decompose (bool): Whether to use decomposed model.

    Returns:
        None

    This function evaluates the neural network's performance on the test dataset for one epoch.
    It computes the loss, accuracy, and updates the test history.

    If `decompose` is True, it stores the test history in a separate checkpoint folder.
    """
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    # Evaluate the network on the test dataset
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Display progress during testing
            step_time, tot_time = progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Determine the checkpoint folder based on the 'decompose' argument
    folder_path = 'checkpoint'
    if decompose:
        folder_path = 'checkpoint_decomp'

    # Load the existing checkpoint
    state = torch.load(folder_path+'/ckpt.pth')

    # Update the test history
    state['test_history']['loss'].append(test_loss / (batch_idx + 1))
    state['test_history']['step time'].append(step_time)
    state['test_history']['tot time'].append(tot_time)
    state['test_history']['acc'].append(100. * correct / total)
    state['epoch'] = len(state['test_history']['loss'])

    # Save the updated checkpoint if the accuracy is higher than the best
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state['net'] = net.state_dict()
        state['best_acc'] = acc

        torch.save(state, folder_path+'/ckpt.pth')
        best_acc = acc

if os.path.isdir('checkpoint'):
  weight_path = ('./checkpoint/ckpt.pth')

if __name__ == '__main__':

  if args.train:

    # this will create a checkpoint folder and ckpt.pth file to store state dict
    for epoch in range(start_epoch, start_epoch+240):
        train(epoch)
        test(epoch)
        scheduler.step()

  elif args.resume:
      # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']+1
    for epoch in range(start_epoch, start_epoch+200):
        train(epoch)
        test(epoch)
        scheduler.step()

  elif args.decompose:
      #________Used to address formating issue_________
      if not args.add:

        if len(new_weight_path)>0:
          # Load weights from 'new_weight_path' if it's not empty; otherwise, load weights from 'weight_path'
          checkpoint = torch.load(new_weight_path)
        else: checkpoint = torch.load(weight_path)
        # Create a new ordered dictionary to hold the updated model weights
        new_odict = collections.OrderedDict()
        for key, value in checkpoint['net'].items():
            new_key = key.replace("module.", "") # Remove the "module." prefix from the keys
            new_odict[new_key] = value
        # Load the updated model weights into the 'net' model
        net.load_state_dict(new_odict)
      #----------------------------------------------------
      net.eval()
      net.cpu()

      # Residual structure
      # res = False
      # if res: print("Residual structure enabled")
      # else: print("Residual structure disabled")
      # print(''*100)

      count = 0
      count_d = 0
      # Some model class definition does not have [feature] object, which may require a change in module keys to help algorithm locate layers
      # for i, key in enumerate(net._modules.keys()):
      #     if layer_to_decomp != ['all']:
      #       if i not in layer_to_decomp:# control which layer to decompose
      #         continue
      #     if i in [4]:
      #       for i, key1 in enumerate(net._modules[key]._modules.keys()):
      #           for i, key2 in enumerate(net._modules[key]._modules[key1]._modules.keys()):
      #             if i in [0,2]:
      #               conv_layer = net._modules[key]._modules[key1]._modules[key2]
      #               if args.tucker:
      #
      #                       nparam, npd,ratio, decomposed = tucker_decomposition_conv_layer(conv_layer, tucker_rank_selection_method, target_ratio_)
      #                       count+=nparam
      #                       count_d+=npd
      #                       print("Number of params before: "+str(nparam)+" || after: "+str(npd))
      #               net._modules[key]._modules[key1]._modules[key2] = decomposed
      N = len(net.features._modules.keys())
      # Iterate over the layers in the neural network's features module
      for i, key in enumerate(net.features._modules.keys()):
          # Exit loop if we've reached the last two layers
          if i >= N - 2:
              break

          if i == 0: continue # Ignore the first input convolutional layer
          if layer_to_decomp != 'all':
            # Control which layers to decompose based on 'layer_to_decomp'
            if i not in layer_to_decomp:
              continue
          # Check if the current layer is an instance of convolutional layer
          if isinstance(net.features._modules[key], torch.nn.modules.conv.Conv2d):

              conv_layer = net.features._modules[key]

              print("Decomposing layer " +str(i)+": " +str(net.features._modules[key]))

              if args.tucker:
                # Perform Tucker decomposition on the convolutional layer
                nparam, npd,ratio, decomposed = tucker_decomposition_conv_layer(conv_layer, tucker_rank_selection_method, target_ratio_)
                count+=nparam
                count_d+=npd
                print("Number of params before: "+str(nparam)+" || after: "+str(npd))

              else:
                if rank == 'auto': # Automatically determine the rank for CP decomposition
                  rank_ = max(conv_layer.weight.data.numpy().shape)//3
                elif rank == 'full':
                  rank_ = max(conv_layer.weight.data.numpy().shape)
                else:
                  rank_ = rank[count]
                  count+=1
                print("CP rank = "+ str(rank_))
                # Perform CP decomposition on the convolutional layer
                ratio, decomposed = cp_decomposition_conv_layer(conv_layer, rank_, res)
              # Update the layer with the decomposed version
              net.features._modules[key] = decomposed
              # torch.save(net.state_dict(), './model_data/decomp_weight.pth')

              print("Decomposition of layer "+str(i)+" Completed. Ratio = " + str(ratio))
              print(''*100)

      torch.save(net, model_path)
      if args.tucker:
        if count>0: print("Total param reduction: "+str(count)+" ==> "+str(count_d)+"(X"+str(round(count/count_d,2))+")")

      print('='*100)
      print(''*100)
      print("==> Building decomposed model..")
      print(net)
      print('='*100)
      print(''*100)
      net.cuda()
      print("Accuracy of new model: ")
      test1(1)

  elif args.run_model:
    # Load a pre-trained model from 'model_path'
    net = torch.load(model_path)
    print(net)
    net.eval() # Set the model to evaluation mode
    net.cuda()
    print("Starting learning rate is: "+str(args.lr))
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                momentum=0.9, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    if args.fine_tune:
      if args.resume_d:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint_decomp'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('/content/checkpoint_decomp/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        for epoch in range(start_epoch, start_epoch+5):
            train(epoch, decompose = True, resume = True)
            test(epoch, decompose = True)
            scheduler.step()
      else:
         # Fine-tune the model for the specified number of epochs
        for epoch in range(1, fine_tune_epochs+1):
            train(epoch, decompose = True)
            test(epoch, decompose = True)
            scheduler.step()
            torch.save(net,"fine_tuned_model")

    else:
      print('='*100)
      print(''*100)
      print("Accuracy of new model: ")
      test1(1)
