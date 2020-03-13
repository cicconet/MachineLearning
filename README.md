# Machine Learning

Folders: CIFAR10, MNIST, Cells
-- subsets of the CIFAR10 (https://www.cs.toronto.edu/~kriz/cifar.html [1]),
MNIST (http://yann.lecun.com/exdb/mnist/ [2]),
and Mouse Embryo Tracking (http://celltracking.bio.nyu.edu/ [3]) databases.

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

Folder Tutorial: growing complexity, step-by-step tutorial.

&nbsp;  

References:

[1] Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.

[2] MNIST handwritten digit database. Yann LeCun, Corinna Cortes, and CJ Burges. ATT Labs [Online]. Volume 2, 2010.

[3] Label Free Cell-Tracking and Division Detection Based on 2D Time-Lapse Images For Lineage Analysis of Early Embryo Development.
Marcelo Cicconet, Michelle Gutwein, Kristin C Gunsalus, and Davi Geiger. Computers in Biology and Medicine, Volume 51, p. 24-34, 1 Aug. 2014.