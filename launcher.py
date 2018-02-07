# Author : DINDIN Meryll
# Date : 12/01/2017

# from database import *
from model import *

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
    # for ana in ['Hips', 'Hand', 'Torso'] :
    #     # Define the constructor
    #     inp = '../data_huawei/Safe_Keeper/dtb_{}.h5'.format(ana)
    #     out = '../data_huawei/dtb_{}.h5'.format(ana)
    #     Constructor(inp, out).standardize()

    # Launch the learning tasks
    with open('arguments.pk', 'rb') as raw : args = pickle.load(raw)
    # Defines the model and make it learn
    for ana in ['Hips', 'Hand'] :
        for typ in ['', '_basic', '_transport'] :
            mod = DModel('../data_huawei/dtb_{}.h5'.format(ana), args, msk_labels=[4,5,6,7])
            mod.load_model('../clfs_huawei/clf_{}{}.h5'.format(ana, typ))
            dtf = mod.evaluate()
            dtf.to_pickle('../clfs_huawei/score_{}{}.h5'.format(ana, typ))
            del mod, dtf