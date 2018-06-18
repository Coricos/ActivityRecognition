# Author : DINDIN Meryll
# Date : 01/11/2017

import warnings, sys

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gd

import os, tqdm, h5py, pickle, multiprocessing
import tensorflow, itertools, math, glob

from functools import partial
from scipy.stats import randint, moment, kurtosis, skew
from mpl_toolkits.mplot3d import Axes3D

from sklearn import preprocessing
from sklearn.neighbors import KDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_curve, auc, f1_score, recall_score, precision_score, accuracy_score
from sklearn.metrics import confusion_matrix, mean_squared_error, roc_auc_score
from sklearn.utils import shuffle

from keras import backend as K
from keras import initializers
from keras.engine.topology import Layer
from keras.utils import np_utils
from keras.models import Sequential, Model, load_model
from keras.layers import Convolution2D, MaxPooling2D, merge, Activation, Dropout, Flatten, Dense
from keras.layers import Conv1D, Input, MaxPooling1D, GlobalAveragePooling1D, LSTM, Bidirectional
from keras.layers import TimeDistributed, BatchNormalization, GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling1D, GaussianDropout, Reshape, PReLU
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.layers.merge import concatenate