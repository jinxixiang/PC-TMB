# PC-TMB
 Pytorch implementation of "Pan-cancer computational histopathology reveals tumor mutational burden status through weakly-supervised deep learning"

## Step 1: Prepare WSI

Crop WSI into small patches and then extract low-dimensional farures. Refer to [CLAM](https://github.com/mahmoodlab/CLAM). We use customized self-supervised learning pretrained model rather than the routine ImageNet ptrained model. Weights and code of pretrained model will be releases upon publish.

## Step 2: Construct graph using the embedding vectors
Use *construct_graph.py*

## Step 3: Train MIL model.
Use *train.py*

