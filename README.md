
# Multi-Object Tracking using FairMOT

This is a PyTorch implementation of [FairMOT](https://arxiv.org/abs/2004.01888) for multi-object tracking. Weights can be downloaded from [here](https://drive.google.com/file/d/1QYvMf1ttsfpkZFCRkHrUANCtn54KInGf/view?usp=sharing) and should be put in a new models/ directory. Create a conda environment using:`conda create --name mot --file requirements.txt`. detect.py runs FairMOT on sample images in /images and track.py on sample frames in /frames. Tracker data association is WIP. Available hyperparameters in train.py, detect.py and detect.py can be explored with --help

