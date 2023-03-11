
from models import *
model = VGG('VGG19')
# Params for training
learning_rate = 0.001
momentum = 0.9

new_weight_path = "/content/CNN_compression_with_Tensor_Decomposition/trained_weights/VGG19_240iter_ckpt.pth"
# new_weight_path = "/content/checkpoint/ckpt.pth" 
layer_to_decomp = [0] # list determining layers to decompose

# rank chosen from 'auto' , "full" or desired number
rank = [3]
# whether to add residual structure to the decomposed model; oroginally designed to account for 
# possible gradient problem, but proven to be not successful.
res = False
fine_tune_epochs = 15 # 

# used pretrained model to test accuracy
model_path = "/content/decomposed_model"
