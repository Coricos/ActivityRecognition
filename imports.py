# Author : DINDIN Meryll
# Date : 01/11/2017

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, tqdm, h5py, pickle, multiprocessing, sys
import tensorflow

from functools import partial

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_curve, auc, f1_score, recall_score, precision_score, accuracy_score
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.utils import shuffle

from keras.utils import np_utils
from keras.models import Sequential, Model, load_model
from keras.layers import Convolution2D, MaxPooling2D, merge, Activation, Dropout, Flatten, Dense
from keras.layers import Conv1D, Input, MaxPooling1D, GlobalAveragePooling1D, LSTM, Bidirectional
from keras.layers import TimeDistributed, BatchNormalization, GlobalAveragePooling2D
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint