# TensorFlow 101

This project has three main files (nn_shallow.py, nn_deep.py, and cnn_deep.py), and two sample datasets
(subsets of the MNIST and CIFAR10 databases). The Python routines are modified from the
"MNIST For ML Beginners" and "Deep MNIST for Experts" (from https://www.tensorflow.org/versions/0.6.0/tutorials/index.html).

The main diference is that the .py files here contain code to read and build
train/test sets from regular image files, and therefore can be more easily deployed to other databases (which,
ultimately, is the goal of the user). Notice that the folders MNIST and CIFAR10 are organized
in subfolders Train and Test, and in these subfolders each class has a separate folder (0, 1, etc).

Therefore, one simple way to deploy {nn_shallow,nn_deep,cnn_deep}.py to your own custom database is to organize
your database in the same hierarchy as MNIST and CIFAR10 in this project, and modify the variable
"path" on the .py routines to point to your dataset. Notice that your dataset doesn't have to
have 10 classes; however, all images in the provided sample datasets are grayscale and have size 28x28,
hence non-trivial modifications to the code should be performed in order to deal with other types of images.

Other sources:
https://www.udacity.com/course/viewer#!/c-ud730 (class on Udacity);
http://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow/ (joke/tutorial).