# Author : DINDIN Meryll
# Date : 12/01/2017

# Launching test on performances
from creator import *

# Main instructions
if __name__ == '__main__' : 

	# Load the arguments
	with open('arguments.pk', 'rb') as raw :
		args = pickle.load(raw)
	# Defines the model and make it learn
	mod = DynamicModel('../data_hapt/database.h5', args)
	mod.learn('../clfs_hapt/test.h5', max_epochs=5)
	print(mod.evaluate())