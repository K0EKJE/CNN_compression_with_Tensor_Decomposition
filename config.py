
from models import *
model = VGG('VGG19')
# Params for training
learning_rate = 0.001
target_ratio_ = 0.79

new_weight_path = "/content/CNN_compression_with_Tensor_Decomposition/trained_weights/VGG19_240iter_ckpt.pth"
# new_weight_path = "/content/checkpoint/ckpt.pth"
layer_to_decomp = ['all']#'all'
#[3,7,10] # list determining layers to decompose
# [3,7,10,14,17,20,23,27]
# rank chosen from 'auto' , "full" or desired number for CP
rank = [3]

# rank selection for Tucker, QR VBMF, SVD
tucker_rank_selection_method = 'VBMF'

# whether to add residual structure to the decomposed model; oroginally designed to account for
# possible gradient problem, but proven to be not successful.
res = False
fine_tune_epochs = 15 #

# used pretrained model to test accuracy
model_path = "/content/decomposed_model"
