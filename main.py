'''Performing tensor decomposition on pre-trained model with PyTorch.'''

import tensorly as tl
import tensorly

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
from tensor_decomp.decompositions import cp_decomposition_conv_layer, tucker_decomposition_conv_layer



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument('--lr', default=.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')

parser.add_argument("--decompose", dest="decompose", action="store_true")
parser.add_argument("--fine_tune", dest="fine_tune", action="store_true")
parser.add_argument("--tucker", dest="tucker", action="store_true", \
    help="Use tucker decomposition. uses cp by default")
parser.set_defaults(train=False)
parser.set_defaults(decompose=False)
parser.set_defaults(fine_tune=False)
parser.set_defaults(tucker=False)    


args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')

net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2()
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
print(net)
print('='*100)

net = net.cuda()
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

history = {}
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
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
        

    if not os.path.isdir('checkpoint'):
      os.mkdir('checkpoint')
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
      torch.save(state, './checkpoint/ckpt.pth')
    else: 
      state = torch.load('./checkpoint/ckpt.pth')
      state['train_history']['loss'].append(train_loss/(batch_idx+1))
      state['train_history']['step time'].append(step_time)
      state['train_history']['tot time'].append(tot_time)
      state['train_history']['acc'].append(100.*correct/total)
      torch.save(state, './checkpoint/ckpt.pth')

def test(epoch):
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

    state = torch.load('./checkpoint/ckpt.pth')

    state['test_history']['loss'].append(test_loss/(batch_idx+1))
    state['test_history']['step time'].append(step_time)
    state['test_history']['tot time'].append(tot_time)
    state['test_history']['acc'].append(100.*correct/total)
    torch.save(state, './checkpoint/ckpt.pth')

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        # state = {
        #     'net': net.state_dict(),
        #     'acc': acc,
        #     'epoch': epoch,
        # }
        # if not os.path.isdir('checkpoint'):
        #     os.mkdir('checkpoint')

        state['net'] = net.state_dict()
        state['best_acc'] = acc
        state['epoch'] = epoch

        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

def train_decomp(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer_decomp.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer_decomp.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        step_time, tot_time = progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        
def test_decomp(epoch):

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
import collections


# def load_model(model,weight_path):
    
#     checkpoint = torch.load(weight_path)

#     new_odict = collections.OrderedDict()

#     for key, value in checkpoint['net'].items():
#         new_key = key.replace("module.", "")
#         new_odict[new_key] = value
    
#     model.load_state_dict(new_odict)

if os.path.isdir('checkpoint'):
  weight_path = ('./checkpoint/ckpt.pth')
else: 
  # depends on model
  weight_path = "/content/Tensor-Decompositions-with-PyTorch-/trained_weights/VGG19_240iter_ckpt.pth"

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
      #________Used to solve some formating issue_______
      checkpoint = torch.load(weight_path)
      new_odict = collections.OrderedDict()

      for key, value in checkpoint['net'].items():
          new_key = key.replace("module.", "")
          new_odict[new_key] = value
      
      net.load_state_dict(new_odict)
      #=======================================================
      # net.load_state_dict(checkpoint['net'])
      net.eval()
      net.cpu()
      N = len(net.features._modules.keys())
      if not os.path.isdir('model_data'):
        os.mkdir('model_data')

      # Residual structure 
      res = False
      if res: print("Residual structure enabled")
      else: print("Residual structure disabled")

      for i, key in enumerate(net.features._modules.keys()):

          if i >= N - 2:
              break
          if i>4: # control which layer to decompose
            break 
          if isinstance(net.features._modules[key], torch.nn.modules.conv.Conv2d):
              conv_layer = net.features._modules[key]
              rank = max(conv_layer.weight.data.numpy().shape)//3
              # rank = 3
              print("Decomposing layer " +str(i)+": " +str(net.features._modules[key])+"|| rank = "+ str(rank))
              
              if args.tucker:
                  decomposed = tucker_decomposition_conv_layer(conv_layer)
              else:
                # rank = max(conv_layer.weight.data.numpy().shape)//3
                decomposed = cp_decomposition_conv_layer(conv_layer, rank, res)


              net.features._modules[key] = decomposed
          # torch.save(net.state_dict(), './model_data/decomp_weight.pth')
          #torch.save(net, "decomposed_model")
          # data = {}
          # data['decomp_weights'] = net.state_dict()
          if not res:
            model_scripted = torch.jit.script(net) # Export to TorchScript
            model_scripted.save('model_scripted.pt') 
      print("Decomposition Completed.")

      net = net.cuda()
      optimizer_decomp = optim.SGD(net.parameters(), lr=0.00001,
                      momentum=0.9, weight_decay=5e-5)
      scheduler_decomp = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_decomp, T_max=10)
      for epoch in range(1, 100):
          train_decomp(epoch)
          test_decomp(epoch)
          scheduler_decomp.step()



    

