# pytorch_tensorboard_optuna

This repository shows how to use tensorboard and optuna on PyTorch.  
Optuna is an automatic hyperparameter optimization software framework.  
I referred to websites below.  
* https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html  
* https://pytorch.org/docs/stable/tensorboard.html  
* https://github.com/optuna/optuna  

## Requirement
python 3.6  
torch 1.4  
torchvision 0.5  
tensorboard  
future  
optuna  

## How to run training
```
$ python main.py # for training and evaluation
```

## How to watch tensorboard logs
```
$ tensorboard --logdir runs/ # browse "http://localhost:6006/"
```
