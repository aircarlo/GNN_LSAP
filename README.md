# GNN_LSAP


This repository contains the Python implementation of the framework described in: [Tackling the Linear Sum Assignment Problem with Graph Neural Networks](http://) - _Carlo Aironi, Samuele Cornell, and Stefano Squartini_, where Linear Sum Assignment Problems of different dimensions are faced with a data-driven approach based on Graph Neural Networks, and accuracy is compared against two existing DNN-based frameworks.

To replicate the experiments, type `python main.py` followed by `--help` to know the startup arguments.

## Requirements
- scipy                     1.8.1
- pytorch                   1.11.0
- torch-geometric           2.0.4
- torch-cluster             1.6.0
- torch-scatter             2.0.9
- torch-sparse              0.6.13
