# Author : DINDIN Meryll
# Date : 12/01/2017

from loading import *
from creator import *

# Main instructions
if __name__ == '__main__' :

    # Create all databases
    # for ana in ['Hips', 'Hand', 'Torso'] :
    #     # Define the loader
    #     loa = SHL_Loader('../data_huawei/Safe_Keeper/dtb_{}.h5'.format(ana), 250, 0.25, ana)
    #     loa.load_signals()
    #     loa.define_test()
    #     del loa

    # Standardize databases
    for ana in ['Hips', 'Hand', 'Torso'] :
        # Define the constructor
        inp = '../data_huawei/Safe_Keeper/dtb_{}.h5'.format(ana)
        out = '../data_huawei/dtb_{}.h5'.format(ana)
        Constructor(inp, out).standardize()

    # Launch the learning tasks
    with open('arguments.pk', 'rb') as raw : args = pickle.load(raw)
    for ana in ['Hips', 'Hand', 'Torso'] :
        # Defines the model and make it learn
        mod = DynamicModel('../data_huawei/dtb_{}.h5'.format(ana), args)
        mod.learn('../clfs_hapt/model_{}.h5'.format(ana), max_epochs=50)
        del mod
