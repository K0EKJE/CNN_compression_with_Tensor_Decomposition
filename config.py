
# for training
learning_rate = 0.01
momentum = 0.9

weight_path = "/content/CNN_compression_with_Tensor_Decomposition/trained_weights/VGG19_240iter_ckpt.pth"
layer_to_decomp = [3]

# rank chosen from 'auto' or desired number
rank = 'auto'
#whether to add residual structure to the decomposed model
res = False
fine_tune_epochs = 20

time_disp = True # whether to display running time for each layer
