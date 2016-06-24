# TensorFlow 101

Folders: CIFAR10, MNIST, Cells
-- subsets of the CIFAR10 (https://www.cs.toronto.edu/~kriz/cifar.html),
MNIST (http://yann.lecun.com/exdb/mnist/),
and Mouse Embryo Tracking (http://celltracking.bio.nyu.edu/) databases.

CIFAR10 and MNIST: 10 classes, 1000 samples/class for training, 100 samples/class for testing.

Cells: 4 classes, 500 samples/class for training, 200 samples/class for testing.

Files nn_shallow.py, nn_deep.py, cnn_deep.py:

These Python routines are modified from the
"MNIST For ML Beginners" and "Deep MNIST for Experts" (from https://www.tensorflow.org/versions/0.6.0/tutorials/index.html).
The main diference is that the .py files here contain code to read and build
train/test sets from regular image files, and therefore can be more easily deployed to other databases. 

Notice that the folders are organized
in subfolders Train and Test, and in these subfolders each class is in a separate folder (0, 1, etc).
Therefore, one simple way to deploy {nn_shallow,nn_deep,cnn_deep}.py to your own database is to organize your database in the same hierarchy as MNIST/CIFAR10/Cells in this project, and modify the variable
"path" on the .py routines to point to your dataset. Notice that your dataset doesn't have to
have 10 classes; however, all images in the provided sample datasets are grayscale and have size 28x28,
hence non-trivial modifications to the code should be performed in order to deal with other types of images.

File nn_plot.py:

Toy example using synthetic 2D data. The goal is to visualize the result of classification.

Other sources:
https://www.udacity.com/course/viewer#!/c-ud730 (class on Udacity);
http://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow/ (joke/tutorial).