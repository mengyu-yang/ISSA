This repo consists of two folders. `ISSA` contains the official implementation for the NeurIPS 2021 Workshop on Meta-Learning paper **[Few Shot Image Generation via Implicit Autoencoding of Support Sets](https://openreview.net/pdf?id=fem00ckyS8t)**. `ISSA_experimental` contains code for an extended ISSA model outside of the paper. Here, we experiment with a StyleGAN2-based ISSA rather than a DCGAN-based one. 

If you have any questions, feel free to contact the corresponding author: shenyang.huang@mail.mcgill.ca

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


### FID Experiment
Omniglot:
> bash omni_fid.sh 1 omni_fid

CelebA:
> bash celeba_fid.sh 1 celeba_fid

### Identity Accuracy Experiment
Omniglot:
* Train the classifier first by using train_classifier() in `omni_accuracy.py`
> bash omni_accuracy.sh 1 omni_accuracy

CelebA, obtain pre-trained face classifier from https://github.com/deepinsight/insightface
> bash celeba_accuracy.sh 1 celeba_accuracy


## ISSA_experimental 

This folder uses the codebase from the [official PyTorch implementation of StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch). Changes are made to convert it to the ISSA training procedure. 

### Environment Setup 

`stylegan2.yml` lists the dependencies I used. There may be some additional packages in there that may not be required. Based off the official conda cheatsheet, you can create an environmental from this file using

> conda env create --file stylegan2.yml 

### Data 

The experimental model has been mainly tested on the CelebA dataset. To prepare the data, first download the raw dataset. You should get a folder with all the CelebA images in it (no subfolders). Use `dataset_tool.py` to process the images. Documentation can be found in the official repo linked above. As an example, the command I used is

> python dataset_tool.py --source {dir_to_raw_dataset} --dest {dataset_name}.zip --width 64 --height 64

Note that within `dataset_tool.py`, on line 61, I hardcoded the location of the CelebA json file that the code required to assign labels to each image. The original code didn't seem to work and some changes have been made to make it accomodate labeled data. Take a look at `celeba_train.json` to see what the required format is like. 

### Training 

There are 2 bash `.sh` files that are used to run and train the models. `run_NABirds_baseline.sh` runs a mostly baseline version of ISSA implemented on StyleGAN2 modules. `run_NABirds_iae.sh` has the additional feature of being able to swap in different loss functions and modules (either DCGAN or StyleGAN2) for the encoder, generator, and discriminator. The full list of training parameters can be found as click options in `train.py` and `train_issa_swap.py`.  
