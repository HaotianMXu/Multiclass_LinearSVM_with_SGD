# Multiclass_LinearSVM_with_SGD
A demonstration of how to use PyTorch to implement Support Vector Machine with one-vs.-all hinge loss. Weighted penalty of each class and square hinge loss are also available.

## Requirements
* PyTorch==0.2.0 with GPU support
* Python==3.5

## Approach
* The key idea is to optimize a linear classifier with [one-vs-all Hinge loss](https://en.wikipedia.org/wiki/Hinge_loss) proposed by Dr. Weston and Dr. Watkins.
* For more details, please refer the loss function in the code.
