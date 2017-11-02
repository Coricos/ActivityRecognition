# Author : DINDIN Meryll
# Date : 02/11/2017

from models import *

if __name__ == '__main__' :

    mod = Models('Conv1D').learn(max_epochs=500, verbose=1)
    mod.save_model()
    del mod

    mod = Models('DeepConv1D').learn(max_epochs=500, verbose=1)
    mod.save_model()
    del mod
