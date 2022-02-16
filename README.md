PyTorch implementation of ComplexGCN:


# Running a model

     CUDA_VISIBLE_DEVICES=1 python complexgcn.py -model_name complexgcn -dataset FB15k-237 
     CUDA_VISIBLE_DEVICES=1 python complexgcn.py -model_name complexgcn -dataset WN18RR -lr 0.0003 -dim 400 -drop1 0.3 -drop2 0.3 -drop3 0.3

For ablated model on FB15k-237, set -model_name ablated

Unzip data.zip

#Required packages:

    Python     3.7.6
    numpy      1.19.2
    pytorch    1.8.1
    

Acknowledgment: The code is based on https://github.com/tkipf/pygcn, https://github.com/ibalazevic/TuckER, https://github.com/TimDettmers/ConvE
