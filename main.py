'''Performing tensor decomposition on pre-trained model with PyTorch.'''

import tensorly as tl
import tensorly

import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import os

from models import *
from util.utils import progress_bar,check_folder
from tensor_decomp.decompositions import cp_decomposition_conv_layer, tucker_decomposition_conv_layer
from config import *
from util.dataset import *
from util.parser import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


# Model
# print('==> Building model..')

if args.add:
  net = torch.load("/content/decomposed_model")
else: net = model
net = net.cuda()
# print(net)
# print("Layers: ", net._modules.keys())

if device == 'cuda':
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate,
                      momentum=0.9, weight_decay=5e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

history = {}
# Training
def train(epoch, decompose = False, resume = False):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    folder_path = 'checkpoint'
    if decompose:
      folder_path = 'checkpoint_decomp'

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        step_time, tot_time = progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    if not resume: # delete the checkpoint before running an another instance of decomposition
      check_folder(folder_path)
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

      state = torch.load(folder_path+'/ckpt.pth') # for training to resume
      state['train_history']['loss'].append(train_loss/(batch_idx+1))
      state['train_history']['step time'].append(step_time)
      state['train_history']['tot time'].append(tot_time)
      state['train_history']['acc'].append(100.*correct/total)
      torch.save(state, folder_path+'/ckpt.pth')

def test1(epoch): # used to test accuracy without fine tuning
    '''
    Run the net to test accuracy on test set
    '''
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


def test(epoch, decompose = False):
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
    folder_path = 'checkpoint'
    if decompose:
      folder_path = 'checkpoint_decomp'

    state = torch.load(folder_path+'/ckpt.pth')

    state['test_history']['loss'].append(test_loss/(batch_idx+1))
    state['test_history']['step time'].append(step_time)
    state['test_history']['tot time'].append(tot_time)
    state['test_history']['acc'].append(100.*correct/total)
    torch.save(state, folder_path+'/ckpt.pth')

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state['net'] = net.state_dict()
        state['best_acc'] = acc
        state['epoch'] = epoch

        torch.save(state, folder_path+'/ckpt.pth')
        best_acc = acc

import collections

if os.path.isdir('checkpoint'):
  weight_path = ('./checkpoint/ckpt.pth')

if __name__ == '__main__':

  if args.train:
    # this will create a checkpoint folder and ckpt.pth file to store state dict
    for epoch in range(start_epoch, start_epoch+200):
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
    start_epoch = checkpoint['epoch']
    for epoch in range(start_epoch, start_epoch+200):
        train(epoch)
        test(epoch)
        scheduler.step()

  elif args.decompose:
      #________Used to address formating issue_________
      if not args.add:

        if len(new_weight_path)>0:
          checkpoint = torch.load(new_weight_path)
        else: checkpoint = torch.load(weight_path)

        new_odict = collections.OrderedDict()
        for key, value in checkpoint['net'].items():
            new_key = key.replace("module.", "")
            new_odict[new_key] = value
        net.load_state_dict(new_odict)
      #----------------------------------------------------
      net.eval()
      net.cpu()
      N = len(net.features._modules.keys())

      # Residual structure
      # res = False
      if res: print("Residual structure enabled")
      else: print("Residual structure disabled")
      print(''*100)
      #if args.tucker:
      count = 0
      count_d = 0
      for i, key in enumerate(net.features._modules.keys()):
          if i >= N - 2:
              break
          if i == 0: continue #ignore first input convolution layer
          if layer_to_decomp != 'all':
            if i not in layer_to_decomp:# control which layer to decompose
              continue
          # net.features._modules

          if isinstance(net.features._modules[key], torch.nn.modules.conv.Conv2d):

              conv_layer = net.features._modules[key]


              print("Decomposing layer " +str(i)+": " +str(net.features._modules[key]))

              if args.tucker:
                nparam, npd,ratio, decomposed = tucker_decomposition_conv_layer(conv_layer, tucker_rank_selection_method, target_ratio_)
                count+=nparam
                count_d+=npd
                print("Number of params before: "+str(nparam)+" || after: "+str(npd))

              else:
                if rank == 'auto':
                  rank_ = max(conv_layer.weight.data.numpy().shape)//3
                elif rank == 'full':
                  rank_ = max(conv_layer.weight.data.numpy().shape)
                else:
                  rank_ = rank[count]
                  count+=1
                print("CP rank = "+ str(rank_))
                # rank = max(conv_layer.weight.data.numpy().shape)//3
                ratio, decomposed = cp_decomposition_conv_layer(conv_layer, rank_, res)


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
    net = torch.load(model_path)
    print(net)
    net.eval()
    net.cuda()
    print("Starting learning rate is: "+str(args.lr))
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                momentum=0.9, weight_decay=5e-5)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,step_size=5, gamma=0.1, T_max=100)
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
