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
from utils.visualization import prepare_img_to_plot, parula_map
from utils.visualization import get_gaussian_lesion_map as get_lesion_map
from keras.engine import Layer
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import load_model,save_model
from classifier import Classifier
######################################################################################################
Expt_load = 'exp01.hdf5'
Expt_save = 'exp02.hdf5'
LOAD_MODEL = True
####################################################################################################3
'''
FUNCTION DEFINITIONS
'''
def add_file_to_iterator(f):
	global X_train,Y_train,X_oracle,Y_oracle
	Addition = np.reshape(X_oracle[f,:,:,:],(1,3,512,512))
	X_train = np.concatenate((X_train,Addition),axis=0)
	Y_train = np.append(Y_train,Y_oracle[f])
	
	X_oracle = np.delete(X_oracle,f,axis=0)
	Y_oracle = np.delete(Y_oracle,f,axis=0)
	print 'X_train_length:',len(X_train)
	print 'X_oracle length:',len(X_oracle)
	np.save('X_train.npy',X_train)
	np.save('Y_train.npy',Y_train)
	np.save('X_oracle.npy',X_oracle)
	np.save('Y_oracle.npy',Y_oracle)
	np.save('X_val.npy',X_val)
	np.save('Y_val.npy',Y_val)
	print 'Data saved'

##############################################################################################

def active_learning(pred_prob):

	pred_prob = array(pred_prob)
	Entrophy = zeros((pred_prob.shape))

	index = 0
	for prob in pred_prob:
		if prob == 1 or prob == 0:
			Entrophy[index] = 0
		else:
			Entrophy[index] = -prob*log2(prob) - (1-prob)*log2(1-prob)
		index = index + 1

	max_entrophy = amax(Entrophy)
	max_entrophy_index = argmax(Entrophy)

	print('Maximum Entropy = {0}'.format(max_entrophy))

	if max_entrophy > 0.4:
		active_learning_mode = 1
		add_file_to_iterator(max_entrophy_index)

	else:
		active_learning_mode = 0

	return active_learning_mode
###########################################################################################################

def train_and_evaluate(batch_size=8, img_dir='mess_cb',
					   out_size=32, f_size=31, horizontal_flip=True, vertical_flip=True,
                       width_shift_range=0.03, height_shift_range=0.03, rotation_range=360,
                       zoom_range=0.03, exp_name='exp0.hdf5', n_blocks=4, context_blocks=1,
                       patience=75):

	global X_train,Y_train,X_oracle,Y_oracle,X_val,Y_val				   
	# Initialization
	active_learning_mode = 1
	epochs = 30

	checkpointer = ModelCheckpoint(filepath= exp_name, monitor = 'val_acc', verbose=1, save_best_only=True, save_weights_only=False)
	early = EarlyStopping(monitor = 'val_acc',patience=3, verbose=1)
	callbacks = [checkpointer,early]
    #####################################################################################################
    # Loading saved model here
	if LOAD_MODEL == True:
		model = load_model(Expt_load)
	else:
		classifier = Classifier()
		model = classifier.prepare_to_init(2e-4)
		model.fit(X_train, Y_train,epochs= epochs, validation_data=(X_val,Y_val), verbose=1, callbacks=callbacks)

		print 'Model trained with top layers'

		model = classifier.prepare_to_finetune(0.0001)
		model.fit(X_train, Y_train,epochs= epochs, validation_data=(X_val,Y_val), verbose=1, callbacks=callbacks)

		print 'Finetuned the model'
		print 'Saving the model'
		model.save('exp01.hdf5')	
    
	while active_learning_mode:
		model.fit(X_train, Y_train, epochs=epochs, validation_data=(X_val,Y_val), verbose=1, callbacks=callbacks)
		pred_prob = model.predict(X_oracle,batch_size=200,verbose=1)
	
		active_learning_mode = active_learning(pred_prob)

	#print('Train AUC = {0}'.format(get_auc_score(model, train_it, train_len)))
	#print('Validation AUC = {0}'.format(get_auc_score(model, val_it, 192)))
	#print('Test AUC = {0}'.format(get_auc_score(model, test_it, 240)))

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

	if LOAD_MODEL==True:
		X_train = np.load('X_train.npy',encoding='bytes')
		Y_train = np.load('Y_train.npy',encoding='bytes')
		X_oracle = np.load('X_oracle.npy',encoding='bytes')
		Y_oracle = np.load('Y_oracle.npy',encoding='bytes')
		X_val = np.load('X_val.npy',encoding='bytes')
		Y_val = np.load('Y_val.npy',encoding='bytes')
	else:		
		X_train,Y_train,X_oracle,Y_oracle,X_val,Y_val = train.get_data_iterators(batch_size=1,data_dir='mess_cb',target_size=(512, 512),rescale=1/255,fill_mode='constant',load_train_data=True,color_mode='rgb')
    
		X_train = np.reshape(X_train,(200,3,512,512))
		X_oracle = np.reshape(X_oracle,(568,3,512,512))
		X_val = np.reshape(X_val,(192,3,512,512))


	train_and_evaluate(batch_size=1, out_size=out_sizes[n_blocks], f_size=f_sizes[n_blocks],
                           exp_name=Expt_save, n_blocks=n_blocks,
                           context_blocks=context_blocks+1, img_dir=args.img_dir)
    
    
