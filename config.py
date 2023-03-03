
# Params for training
learning_rate = 0.01
momentum = 0.9

new_weight_path = "/content/CNN_compression_with_Tensor_Decomposition/trained_weights/VGG19_240iter_ckpt.pth"
# new_weight_path = "" 
layer_to_decomp = [3,7] # list determining layers to decompose

# rank chosen from 'auto' or desired number
rank = [32,64]
# whether to add residual structure to the decomposed model; oroginally designed to account for 
# possible gradient problem, but proven to be not successful.
res = False
fine_tune_epochs = 5 # 

# used pretrained model to test accuracy
model_path = "/content/decomposed_model"