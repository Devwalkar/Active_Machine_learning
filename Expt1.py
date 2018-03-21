import time
import train
import models
import argparse
from keras.optimizers import Adam
from numpy.random import choice
from numpy import amax, argmax, argsort, log2, array, zeros
import random
import numpy as np
import os
from train import load_dataset
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from PIL import Image
from eval.eval import get_predictions, get_auc_score
from keras.callbacks import ModelCheckpoint, EarlyStopping
#from utils.visualization import prepare_img_to_plot, parula_map
#from utils.visualization import get_gaussian_lesion_map as get_lesion_map
from keras.engine import Layer
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import load_model,save_model
from classifier import Classifier
######################################################################################################
Expt_load = 'exp_full.hdf5'
Expt_save = 'exp_full.hdf5'
LOAD_MODEL = False

####################################################################################################3
'''
FUNCTION DEFINITIONS
'''
def train_and_evaluate(batch_size=8, img_dir='mess_cb',
					   out_size=32, f_size=31, horizontal_flip=True, vertical_flip=True,
                       width_shift_range=0.03, height_shift_range=0.03, rotation_range=360,
                       zoom_range=0.03, exp_name='exp0.hdf5', n_blocks=4, context_blocks=1,
                       patience=75):

	global X_train,Y_train,X_val,Y_val				   
	# Initialization
	epochs = 30

	checkpointer = ModelCheckpoint(filepath= exp_name, monitor = 'val_acc', verbose=1, save_best_only=True, save_weights_only=False)
	early = EarlyStopping(monitor = 'val_acc',patience=3, verbose=1)
	callbacks = [checkpointer]
    #####################################################################################################
    # Loading saved model here
	if LOAD_MODEL == True:
		model = load_model(Expt_load)
	else:
		classifier = Classifier()
		model = classifier.prepare_to_init(2e-5)
		model.fit(X_train, Y_train,epochs= 1,batch_size=128,validation_data=(X_val,Y_val),shuffle=True, verbose=1, callbacks=callbacks)

		print 'Model trained with top layers'

		model = classifier.prepare_to_finetune(2e-5)
		model.fit(X_train, Y_train,epochs= epochs,batch_size=128,validation_data=(X_val,Y_val),shuffle=True, verbose=1, callbacks=callbacks)

		print 'Finetuned the model'	
	model.fit(X_train, Y_train,epochs= epochs,batch_size=128,validation_data=(X_val,Y_val),shuffle=True, verbose=1, callbacks=callbacks)
	

###############################################################################################################

if __name__ == '__main__':

	parser = argparse.ArgumentParser(prog='micnn')
	parser.add_argument('--img_dir', nargs='?', help='Directory where the image data is.', dest='img_dir', default='mess_cb')
	args = parser.parse_args()

	out_sizes = {3: 64, 4: 32, 5: 16}
	f_sizes = {3: 15, 4: 31, 5: 63}

	experiments = [[3, 0], [3, 1]]
	for n_blocks in range(3, 6):
		max_context = 5 - n_blocks
		for context_blocks in range(max_context + 1):
			experiments.append([n_blocks, context_blocks])
			
	X_train,Y_train,X_oracle,Y_oracle,X_val,Y_val = train.get_data_iterators(batch_size=1,data_dir='mess_cb',target_size=(512, 512),rescale=1/255,fill_mode='constant',load_train_data=True,color_mode='rgb')
    
	X_train = np.reshape(X_train,(200,3,512,512))
	X_oracle = np.reshape(X_oracle,(568,3,512,512))
	X_val = np.reshape(X_val,(192,3,512,512))

	X_train = np.concatenate((X_train,X_oracle),axis=0)
	Y_train = np.concatenate((Y_train,Y_oracle),axis=0)


	train_and_evaluate(batch_size=1, out_size=out_sizes[n_blocks], f_size=f_sizes[n_blocks],
                           exp_name=Expt_save, n_blocks=n_blocks,
                           context_blocks=context_blocks+1, img_dir=args.img_dir)
    
    
