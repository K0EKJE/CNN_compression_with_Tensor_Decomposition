'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init


import collections

def get_mean_and_std(dataset):
    """
    Compute the mean and standard deviation values of a dataset.

    Args:
        dataset: The dataset for which the mean and std will be computed.

    Returns:
        mean (torch.Tensor): The mean values for each channel.
        std (torch.Tensor): The standard deviation values for each channel.

    This function computes the mean and standard deviation values for each channel
    (e.g., RGB channels for an image dataset) across the entire dataset.
    """
    # Create a DataLoader with a batch size of 1 to iterate over the dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    # Initialize tensors for mean and std for each channel
    mean = torch.zeros(3)
    std = torch.zeros(3)

    print('==> Computing mean and std..')

    # Iterate over the dataset and calculate mean and std
    for inputs, targets in dataloader:
        for i in range(3):  # Assuming 3 channels (e.g., RGB)
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()

    # Divide the sum of mean and std values by the dataset size to get the mean and std
    mean.div_(len(dataset))
    std.div_(len(dataset))

    return mean, std

def init_params(net):
    """
    Initialize the parameters of the layers in a neural network.

    Args:
        net: The neural network for which parameters will be initialized.

    Returns:
        None

    This function iterates over the modules of the neural network and initializes
    the parameters of different types of layers (Conv2d, BatchNorm2d, and Linear).
    """
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            # Initialize Conv2d layer parameters using Kaiming normal initialization
            init.kaiming_normal_(m.weight, mode='fan_out')

            if m.bias is not None:
                # Initialize the bias (if present) with a constant value of 0
                init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            # Initialize BatchNorm2d layer parameters
            init.constant_(m.weight, 1)  # Set weight to 1
            init.constant_(m.bias, 0)    # Set bias to 0

        elif isinstance(m, nn.Linear):
            # Initialize Linear (fully connected) layer parameters
            init.normal_(m.weight, std=1e-3)  # Initialize weight with a small standard deviation

            if m.bias is not None:
                # Initialize the bias (if present) with a constant value of 0
                init.constant_(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

    return step_time, tot_time

def format_time(seconds):
    """
    Convert a duration in seconds into a human-readable string format.

    Args:
        seconds (float): The duration in seconds.

    Returns:
        str: A human-readable string representing the duration.

    This function takes a duration in seconds and converts it into a string format
    that represents the duration in days, hours, minutes, seconds, and milliseconds.
    """
    # Calculate days
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24

    # Calculate hours
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600

    # Calculate minutes
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60

    # Calculate whole seconds
    secondsf = int(seconds)
    seconds = seconds - secondsf

    # Calculate milliseconds
    millis = int(seconds * 1000)

    # Initialize the formatted string
    f = ''
    i = 1

    # Append days, hours, minutes, seconds, and milliseconds to the formatted string
    if days > 0:
        f += str(days) + 'D '
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h '
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm '
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's '
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms '
        i += 1

    # If the formatted string is empty, set it to '0ms'
    if f == '':
        f = '0ms'

    return f


def check_folder(folder_path):
    """
    Create the folder if it doesn't exist, and clear it if it does.

    Args:
        folder_path (str): The path of the folder to check.

    Returns:
        None

    This function checks if the specified folder exists. If it doesn't exist,
    it creates the folder. If the folder already exists, it removes all the files
    inside it, effectively clearing the folder.
    """
    if not os.path.exists(folder_path):
        # Create the folder if it doesn't exist
        os.makedirs(folder_path)

    if os.path.exists(folder_path):
        # If the folder exists
        files = os.listdir(folder_path)
        if len(files) > 0:
            # Iterate over the files and remove them
            for file_name in files:
                os.remove(os.path.join(folder_path, file_name))


def load_model(model, weight_path):
    """
    Load model weights from a checkpoint and assign them to a given model.

    Args:
        model: The target model to which weights will be loaded.
        weight_path (str): The path to the checkpoint file containing the weights.

    Returns:
        None

    This function loads the weights stored in a checkpoint file and assigns them
    to the provided model. It removes the "module." prefix from keys if present in
    the checkpoint.
    """
    # Load the checkpoint file
    checkpoint = torch.load(weight_path)

    # Create a new ordered dictionary to hold the updated model weights
    new_odict = collections.OrderedDict()

    # Iterate over the keys and values in the checkpoint's 'net' dictionary
    for key, value in checkpoint['net'].items():
        # Remove the "module." prefix from the keys (common with DataParallel models)
        new_key = key.replace("module.", "")
        new_odict[new_key] = value

    # Load the updated model weights into the provided model
    model.load_state_dict(new_odict)
