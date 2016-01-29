# TensorFlow 101

There's a sort of "gold rush" between Machine Learning toolkits to grab the attention of developers.
Caffe, Torch, Theano, and now TensorFlow, are only some of the competitors. Which one to choose?

Hard to know for sure. There are the usuall technical trade-offs (see
https://github.com/zer0n/deepframeworks), but for the user, besides technical capabilities,
often times the choice comes down to which one has the best documentation (i.e.,
which one is easier to use).

So far, given the power of it's sponsor, TensorFlow seems to be the one with a more serious
approach to documentation. Still, the MNIST and CNN tutorials could be simpler.

This project has two main files (cnn_shallow.py and cnn_deep.py), and two sample datasets
(subsets of the MNIST and CIFAR10 databases). The Python routines are modified from the
"MNIST For ML Beginners" and "Deep MNIST for Experts" (from https://www.tensorflow.org/versions/0.6.0/tutorials/index.html).

The main diference is that cnn_shallow.py and cnn_deep.py contain code to read and build
train/test sets from regular image files, and therefore can be more easily deployed to other databases (which,
ultimately, is the goal of the user). Notice that the folders MNIST and CIFAR10 are organized
in subfolders Train and Test, and in these subfolders each class has a separate folder (0, 1, etc).

Therefore, one simple way to deploy cnn_shallow.py and cnn_deep.py to your own custom database is to organize
your database in the same hierarchy as MNIST and CIFAR10 in this project, and modify the variable
"path" on the .py routines to point to your dataset. Notice that your dataset doesn't have to
have 10 classes; however, all images in the provided sample datasets are grayscale and have size 28x28,
hence non-trivial modifications to the code should be performed in order to deal with other types of images.