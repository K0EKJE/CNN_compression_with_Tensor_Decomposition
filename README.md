# Compress CNN With Tensor Decomposition

In this research project I developed a pipline to compress convolutional layers of a CNN model with CP Tensor Decomposition using [PyTorch](http://pytorch.org/) and [Tensorly](http://tensorly.org/). I decomposed the second and third layer of VGG19 with different rank to compare the model performance on the CIFAR10 dataset. 

The training is completely conducted on google colab on NVIDIA Tesla T4, so if you want to train on your local machine, you may need to change some of the path settings.

I am still working to implement more decomposition methods including Tucker and Tensor Train decomposition on different models. 

## Key Takeaways
- The decomposition methods is proven to be working for large trained CNN models.
- The decomposition can almost perfectly achieve the accuracy of original model when CP decomposition rank equals to largest dimension of that layer.
- The accuracy of decomposed model recovers fast after one epoch of fine tuning.
- The rank and accuracy of decomposed model forms a linear relationship for lower values of rank.
- It can easily get into gradient problem or local minima so the choice of learning rate must be careful.
- The accuracy is proportional to recovery ratio.

## Prerequisites
- Python 
- PyTorch 
- Tensorly

## Implementation

### Training
If you don't wish to use the pretrained model weights, you can start a new training with 
``` 
python main.py --train
```
and you can manually resume the training with
```
python main.py --resume
```
I trained on VGG19 for around 250 epochs. The default training params are set in ```config.py``` as follow
```
learning_rate = 0.01
momentum = 0.9
```
A file called ```ckpt.pth``` will be created under a folder named ```./checkpoint/ckpt.pth``` during training. It will be a dictionary with a record of the weight data, step time, total time, loss, and accuracy for both testing and training set. There are already some pretrained data on different layers, with different rank and learning rate, in the repo listed below. Just remember to set ```new_weight_path``` variable in the config file to be ```""``` in order to use the new weights.

Then determine the layers for decomposition by altering ```layer_to_decomp``` and ```rank``` in ```config.py```. They are defined by lists, where ```layer_to_decomp = [3,7], rank = [32,64]``` corresponds to perform rank 32 CP decomposition on the 3rd layer and rank 64 CP decomposition on the 7th layer. Be sure to check whether the layer is a convolutional layer by checking the model structure. 

### Decomposing
Start decomposition with 
```
python main.py --decompose
```
It will save the decomposed model as ```"decomposed_model"``` and output current accuracy of the model.

You can always decompose more layers with ```--add```
```
python main.py --decompose --add
```
Just remember to set ```layer_to_decomp``` and ```rank``` to new layers. 

### Testing and fine-tuning
Lastly ```--run_model``` will automatically load the model named ```"decomposed_model"```, but the can be adjusted in ```config.py``` if you wish to upload a trained model.  It will output the current accuracy of the decomposed model(same as above).
```
python main.py --run_model 
```
Then use ```--fine_tune``` and ```--lr``` to fine tune the decomposed model. 
```
python main.py --run_model --fine_tune --lr = 0.00001
```
Set ```fine_tune_epochs``` in ```config.py``` to control total epochs for fine tuning. 

## Part of result accuracy
| Model             | Decomposed Layer | Rank    |Acc(No FT)|Acc(FT) |Param Size |
| ----------------- | ------------     | ------- |-------   |--------| ----------|
| VGG19             | NA               | NA      |93.11%    |NA      |36864(Layer2)|
| New1              | 2                | 4       |14.71%    |88.75%  |536|
| New1              |         2        | 16      |63.21%    |92.15%  |2144|
| New1              | 2                | 64      |93.02%    |93.02%  |8576|
| New1              | 2                | 128     |93.04%    |93.13%  |17152|

| Model             | Decomposed Layer | Rank    |Acc(No FT)|Acc(FT) |Param Size |
| ----------------- | ------------     | ------- |-------   |--------| ----------|
| VGG19             | NA               | NA      |93.11%    |NA      |73728(layer3)|
| New2              | 3                | 4       |36.67%    |88.11%  |792|
| New2              | 3                | 16      |80.74%    |92.04%  |3168|
| New2              | 3                | 64      |92.49%    |92.99%  |12672|
| New2              | 3                | 128     |93.00%    |93.11%  |25344|

| Model             | Decomposed Layer | Rank    |Acc(No FT)|Acc(FT) |Param Size |
| ----------------- | ------------     | ------- |-------   |--------| ----------|
| VGG19             | NA               | NA      |93.11%    |NA      |110592(total)|
| New23             | 2,3              | 32,64   |91.46%    |92.75%  |16960|
| New23             | 2,3              | 64,128  |92.85%    |93.09%  |33920|


## Pretrained weights 

https://github.com/K0EKJE/CNN_compression_trained_weights


## References

[1] https://github.com/kuangliu/pytorch-cifar

[2] https://github.com/jacobgil/pytorch-tensor-decompositions

[3] [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

[4] [Speeding-up Convolutional Neural Networks Using Fine-tuned CP-Decomposition](https://arxiv.org/abs/1412.6553)

[5][Compression of Deep Convolutional Neural Networks for Fast and Low Power Mobile Applications](https://arxiv.org/abs/1511.06530)

[6][Tensorizing Neural Networks](https://arxiv.org/abs/1509.06569)

[7][Tensor-Train Decomposition](https://epubs.siam.org/doi/10.1137/090752286)
