# LALG

LALG stands for Lineal Algebra. This repository contains a really really simple implementation of a numpy style library for linear algebra written from scratch in C. The main goal of this project is to learn how to write a library in C and to understand how numpy works under the hood. Also I added a python wrapper to use the library in python using `ctypes`.

This library is contained in just a small file called `lalg.c` and the python wrapper is in `tensor.py`. 

Also I have intentions to add a autograd feature to the library via the python wrapper in order to allow MLP training.

Of course I don't recommend to use this library in production, it's just a toy project. It is simply unefficient, slow, lacking features and probably full of bugs. But it's a good starting point to understand how numpy works under the hood (and autograd in the future). 