# Activity Recognition

## Introduction 

My whole project is based on the [dataset](https://www.kaggle.com/uciml/human-activity-recognition-with-smartphones) provided for a competition one year ago on Kaggle. My aim was to see if I could improved the results given in the kernels through other methods, such as convolutionnal networks. The first problematic was thus the understanding of their protocol, as I wanted to gather the raw signal corresponding to the given features. The whole development on this repository represents my numerous trials, and the relative performances are given along with it.

## Model tested 

Different approaches have been undertaken: From the classic beginning with the Stochastic Gradient Classifier (optimized through hyperband with k-fold cross-validation) to the Multi-Channel Neural Newtorks (whose results are obtained through cross-validation). Some have advantages that others do not have, and the idea was here, in the end, to test new inputs that may end descriptive in the task of activity identification.

## Features

The features being used to feed both the boosted trees and the neural networks are based on the ones given in the dataset, but also my own handcrafted features, based on experience, which I could obtain due to the extracted raw signals.