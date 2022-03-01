This repo consists of two folders. `ISSA` contains the source code for the NeurIPS 2021 Workshop on Meta-Learning paper **[Few Shot Image Generation via Implicit Autoencoding of Support Sets](https://openreview.net/pdf?id=fem00ckyS8t)**. `ISSA_experimental` contains code for an extended ISSA model outside of the paper. Here, we experiment with a StyleGAN2-based ISSA rather than a DCGAN-based one. 

## ISSA

### Set up Environment

* set up Anaconda environment with Python 3.7.10
* ./setup.sh

### Set up Datasets

Omniglot, to use the same split as DAGAN, we use the numpy Omniglot files from DAGAN repo

* download the numpy format Omniglot dataset from official DAGAN repo: https://github.com/AntreasAntoniou/DAGAN
* place the downloaded files in  `{dataroot}/omniglot` where `{dataroot}` is the directory you plan on storing the dataset and specified as an argparse argument. 

CelebA, crop and resize to 64 x 64 and then use instance selection to retain 80% of images

* use celeba.py to download, crop and resize the images to 64 x 64, specify your own data path in the main function
* (optional) run instance selection on CelebA from https://github.com/uoguelph-mlrg/instance_selection_for_gans and save the instance selected dataset as 'celeba64trainIS.pt' for train images and 'celebalabelstrainIS.pt' for train labels, 'celeba64testIS.pt' for test images and ''celebalabelstestIS.pt' for test labels.
* the original dataset is saved as 'celeba64.pt' for training images and 'celebalabels64.pt' for train labels, 'Test_celeba64.pt' for test images and 'Test_celebalabels64.pt' for test labels
* note that ISSA is trained on an instance selected training set and tested on non-instance selected test domain. ISSA can also work without instance selection as well. 


### Training ISSA
Training on Omniglot:

> python ISSA_omni.py --dataroot {dataset_dir} --output_dir {output_dir}

Training on CelebA:

> python ISSA_celeba.py --dataroot {dataset_dir} --output_dir {output_dir}

where `{dataset_dir}` is the folder where you saved the datasets and `{output_dir}` is the folder where you want to save weights and results to. The rest of the hyperparameters, specified using argparse arguments, are set to default values used within the paper. 


### FID experiment
Omniglot 
* bash omni_fid.sh 1 omni_fid

CelebA
* bash celeba_fid.sh 1 celeba_fid

### identity accuracy experiment
1. Omniglot, train the classifier first 
* train the classifier by using train_classifier() in omni_accuracy.py
* bash omni_accuracy.sh 1 omni_accuracy

2. CelebA, obtain pre-trained face classifier from https://github.com/deepinsight/insightface
* bash celeba_accuracy.sh 1 celeba_accuracy


## ISSA_experimental 

This folder uses the codebase from the [official PyTorch implementation of StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch). Changes are made to convert it to the ISSA training procedure. 


