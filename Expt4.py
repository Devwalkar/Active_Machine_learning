import time
import train
import models
import argparse
from keras.optimizers import Adam
from numpy.random import choice
from numpy import amax, argmax, argsort, log2, array, zeros,amin,argmin
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
Expt_load = 'exp_remove.hdf5'
Expt_save = 'exp_remove.hdf5'
LOAD_MODEL = True
No_images_to_add = 3
No_images_to_remove = 3
####################################################################################################3
'''
FUNCTION DEFINITIONS
'''
def remove_file_from_iterator(f):
	global X_train,Y_train,X_oracle,Y_oracle	

	Addition = np.reshape(X_train[f,:,:,:],(1,3,512,512))
	X_oracle = np.concatenate((X_oracle,Addition),axis=0)
	Y_oracle = np.append(Y_oracle,Y_train[f])

	X_train = np.delete(X_train,f,axis=0)
	Y_train = np.delete(Y_train,f,axis=0)

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

def active_learning(pred_prob,train_prob):
	global No_images_to_add,No_images_to_remove,Y_train
	pred_prob = array(pred_prob)
	Entrophy = zeros((pred_prob.shape))

	train_prob = array(train_prob)
	Train_Entrophy = zeros((train_prob.shape))

	index = 0
	for prob in train_prob:
		if prob == 1 or prob == 0:
			Train_Entrophy[index] = 0
		else:
			Train_Entrophy[index] = -prob*log2(prob) - (1-prob)*log2(1-prob)
		index = index + 1

	index = 0
	for prob in pred_prob:
		if prob == 1 or prob == 0:
			Entrophy[index] = 0
		else:
			Entrophy[index] = -prob*log2(prob) - (1-prob)*log2(1-prob)
			index = index + 1

	for i in range(No_images_to_remove):

		min_entrophy = amin(Train_Entrophy)
		min_entrophy_index = argmin(Train_Entrophy)

		print('Minimum Entropy = {0} for {1}th sample'.format(min_entrophy,i))

		Target_prob = train_prob[min_entrophy_index]
		if ((Target_prob > 0.5) and (Y_train[min_entrophy_index]==1)) or ((Target_prob <= 0.5) and (Y_train[min_entrophy_index]==0)):
			if min_entrophy<=0.4:
				print('Value correctly classified') 	
				remove_file_from_iterator(min_entrophy_index)
				Train_Entrophy = np.delete(Train_Entrophy,min_entrophy_index,axis=0)

	for i in range(No_images_to_add):

		max_entrophy = amax(Entrophy)
		max_entrophy_index = argmax(Entrophy)

		print('Maximum Entropy = {0} for {1}th sample'.format(max_entrophy,i))

		if max_entrophy > 0.4:
			active_learning_mode = 1
			add_file_to_iterator(max_entrophy_index)
			Entrophy = np.delete(Entrophy,max_entrophy_index,axis=0)
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
	epochs = 10

	checkpointer = ModelCheckpoint(filepath= exp_name, monitor = 'val_acc', verbose=1, save_best_only=True, save_weights_only=False)
	early1 = EarlyStopping(monitor = 'val_acc',patience=2, verbose=1)
	early2 = EarlyStopping(monitor = 'val_acc',patience=1, verbose=1)
	callbacks1 = [checkpointer,early1]
	callbacks2 = [checkpointer,early2]
    #####################################################################################################
    # Loading saved model here
	if LOAD_MODEL == True:
		model = load_model(Expt_load)
	else:
		classifier = Classifier()
		model = classifier.prepare_to_init(2e-5)
		model.fit(X_train, Y_train,epochs= 1, validation_data=(X_val,Y_val), verbose=1, callbacks=callbacks1)

		print 'Model trained with top layers'

		model = classifier.prepare_to_finetune(2e-5)
		model.fit(X_train, Y_train,epochs= epochs, validation_data=(X_val,Y_val), verbose=1, callbacks=callbacks2)

		print 'Finetuned the model'
		print 'Saving the model'
		model.save(Expt_save)	
    
	while active_learning_mode:
		model.fit(X_train, Y_train, epochs=epochs, validation_data=(X_val,Y_val),shuffle = True, verbose=1, callbacks=callbacks2)
		pred_prob = model.predict(X_oracle,batch_size=400,verbose=1)
		train_prob = model.predict(X_train,batch_size=64,verbose=1)
		active_learning_mode = active_learning(pred_prob,train_prob)

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
	'''
	if LOAD_MODEL==True:
		X_train = np.load('X_train.npy',encoding='bytes')
		Y_train = np.load('Y_train.npy',encoding='bytes')
		X_oracle = np.load('X_oracle.npy',encoding='bytes')
		Y_oracle = np.load('Y_oracle.npy',encoding='bytes')
		X_val = np.load('X_val.npy',encoding='bytes')
		Y_val = np.load('Y_val.npy',encoding='bytes')
	else:
	'''
	X_train,Y_train,X_oracle,Y_oracle,X_val,Y_val = train.get_data_iterators(batch_size=1,data_dir='mess_cb',target_size=(512, 512),rescale=1/255,fill_mode='constant',load_train_data=True,color_mode='rgb')
    
	X_train = np.reshape(X_train,(200,3,512,512))
	X_oracle = np.reshape(X_oracle,(568,3,512,512))
	X_val = np.reshape(X_val,(192,3,512,512))


	train_and_evaluate(batch_size=1, out_size=out_sizes[n_blocks], f_size=f_sizes[n_blocks],
                           exp_name=Expt_save, n_blocks=n_blocks,
                           context_blocks=context_blocks+1, img_dir=args.img_dir)
    
    
