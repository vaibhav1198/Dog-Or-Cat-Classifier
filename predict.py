# This second file, predict.py, uses a trained network to predict the class for an input image.

# It also have its function file named inference.py that contains predict image function.
# Take gpu as an argument to run predict.py

import matplotlib.pyplot as plt
import numpy as np
import time
import seaborn as sb
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import argparse
from inference import *
import json

predict_parser = argparse.ArgumentParser(
    description='This project is deveoped by Vaibhav Gupta')
 # User should be able to type python predict.py input checkpoint
    # Non-optional image file input
predict_parser.add_argument('input', action="store", nargs='*', default='C:/Users/VAibhAv GupTA/Artificial_Intelligence/Dog_Cat/data_dir/test/test1/6803.jpg')
    # Non-optional checkpoint
predict_parser.add_argument('checkpoint', action="store", nargs='*', default='C:/Users/VAibhAv GupTA/Artificial_Intelligence/Dog_Cat/checkpoint.pth')
  # Choose processor
predict_parser.add_argument('--processor', action="store", dest="processor", default="GPU")


predict_args = predict_parser.parse_args()
print("Image input: ", predict_args.input, "Checkpoint: ", predict_args.checkpoint, "Category names: Cat , Dog",  "Processor: ", predict_args.processor)
print(predict_args)

def main():

    checkpoint = torch.load(predict_args.checkpoint)
    model = checkpoint['model']
    hidden_units = checkpoint['hidden']
    if model == 'densenet121':
        from collections import OrderedDict
        classifier = nn.Sequential(OrderedDict([
                                 ('fc1', nn.Linear(1024, 512)),
                                 ('relu1', nn.ReLU()),
                                 ('fc2', nn.Linear(512, 2)) ,
                                 ('output', nn.LogSoftmax(dim = 1))]))
        model.classifier = classifier


    elif model == 'alexnet':
        from collections import OrderedDict
        classifier = nn.Sequential(OrderedDict([
                                 ('fc1', nn.Linear(9216, hidden_units)),
                                 ('relu1', nn.ReLU()),
                                 ('fc2', nn.Linear(hidden_units, 2)) ,
                                 ('output', nn.LogSoftmax(dim = 1))]))
        model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['classIndex']
    probes, classes = predict(predict_args.input, model, predict_args.processor)
    p = probes.numpy()
    probes = np.reshape(p, (np.product(p.shape),))


     #Displaying the result along with the topk classes
    result = list(zip(classes,probes))
    print('\n\nThe result for the input image is as follows:')
    print(result)

if __name__ == "__main__":
    main()
