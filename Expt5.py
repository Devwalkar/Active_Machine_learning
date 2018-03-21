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
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from PIL import Image
from keras.callbacks import ModelCheckpoint, EarlyStopping
from utils.visualization import prepare_img_to_plot, parula_map
from utils.visualization import get_gaussian_lesion_map as get_lesion_map
from keras.engine import Layer
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import load_model,save_model
from classifier import Classifier
from scipy.spatial.distance import cdist
from skimage.feature import corner_peaks,corner_harris,BRIEF
from skimage.color import rgb2gray

######################################################################################################
Expt_load = 'exp_M_BRIEF.hdf5'
Expt_save = 'exp_M_BRIEF.hdf5'
LOAD_MODEL = True
No_images_to_add = 2
Iterations = 10
####################################################################################################3
'''
FUNCTION DEFINITIONS
'''
def add_file_to_iterator(f):
	global X_train,Y_train,X_oracle,Y_oracle,X_oracle_descriptor,X_train_descriptor
	Addition = np.reshape(X_oracle[f,:,:,:],(1,3,512,512))
	X_train = np.concatenate((X_train,Addition),axis=0)
	Y_train = np.append(Y_train,Y_oracle[f])
	
	Addition = np.reshape(X_oracle_descriptor[f,:],(1,256))
	X_train_descriptor = np.concatenate((X_train_descriptor,Addition),axis=0)

	X_oracle = np.delete(X_oracle,f,axis=0)
	Y_oracle = np.delete(Y_oracle,f,axis=0)
	X_oracle_descriptor = np.delete(X_oracle_descriptor,f,axis=0)

	print 'X_train_length:',len(X_train)
	print 'X_oracle length:',len(X_oracle)
	print 'X_train_descriptor length:',len(X_train_descriptor)
	print 'X_oracle_descriptor length:',len(X_oracle_descriptor)

	np.save('X_trainB.npy',X_train)
	np.save('Y_trainB.npy',Y_train)
	np.save('X_oracleB.npy',X_oracle)
	np.save('Y_oracleB.npy',Y_oracle)
	np.save('X_valB.npy',X_val)
	np.save('Y_valB.npy',Y_val)
	np.save('X_train_descriptor.npy',X_train_descriptor)
	np.save('X_oracle_descriptor.npy',X_oracle_descriptor)
	print 'Data saved'

##############################################################################################
def calculate_descriptors(X):
	extractor = BRIEF()
    
	Descriptors = []
	for i in range(len(X)):
		Im = np.asarray(X[i,:,:,:],dtype='float32')
		Max = np.amax(Im)
		Im = Im/Max
		Im = rgb2gray(Im)
		keypoints = corner_peaks(corner_harris(Im), min_distance=5)		
		extractor.extract(Im,keypoints)
		Temp = extractor.descriptors
		Descriptors.append(np.asarray(np.round(np.average(Temp,axis=0)),dtype='int32'))

	Descriptors_matrix = np.zeros([len(X),256])
	for i in range(len(X)):
		Descriptors_matrix[i,:] = Descriptors[i] 
     
	return Descriptors_matrix

##########################################################################################################3
def calculate_distance():
	global X_train_descriptor,X_oracle_descriptor
	Distances = np.zeros([len(X_oracle_descriptor),len(X_train_descriptor)])
	for i in range(len(X_oracle_descriptor)):
		Oracle = np.reshape(X_oracle_descriptor[i,:],(1,256))
		for j in range(len(X_train_descriptor)):
			Train = np.reshape(X_train_descriptor[j,:],(1,256))
			Distances[i,j] = cdist(Train,Oracle,'hamming')

	Distances = np.reshape(np.average(Distances,axis=1),(len(X_oracle_descriptor),))
	Sorted_distances = np.flip(np.sort(Distances,axis=0),axis=0)
	Sorted_indexes = np.flip(np.argsort(Distances,axis=0),axis=0)
	return Sorted_distances,Sorted_indexes

############################################################################################################3
def active_learning():
	global No_images_to_add,X_oracle_descriptor,X_train_descriptor,t
	t=1
	Dist,Index = calculate_distance()   
	Selected_distances = Dist[:No_images_to_add]
	Selected_Index = Index[:No_images_to_add]

	for i in range(No_images_to_add):
		print 'Adding',str(i),'th image with maximum distance:',str(Selected_distances[i])
		add_file_to_iterator(Selected_Index[i])

###########################################################################################################

def train_and_evaluate(batch_size=8, img_dir='mess_cb',
					   out_size=32, f_size=31, horizontal_flip=True, vertical_flip=True,
                       width_shift_range=0.03, height_shift_range=0.03, rotation_range=360,
                       zoom_range=0.03, exp_name='exp0.hdf5', n_blocks=4, context_blocks=1,
                       patience=75):

	global X_train,Y_train,X_oracle,Y_oracle,X_val,Y_val,t				   
	# Initialization
	epochs = 10

	checkpointer = ModelCheckpoint(filepath= exp_name, monitor = 'val_acc', verbose=1, save_best_only=True, save_weights_only=False)
	early1 = EarlyStopping(monitor = 'val_acc',patience=1, verbose=1)
	early2 = EarlyStopping(monitor = 'val_acc',patience=2, verbose=1)
	callbacks1 = [checkpointer,early1]
	callbacks2 = [checkpointer,early2]
    #####################################################################################################
    # Loading saved model here
	
	if LOAD_MODEL == True:
		model = load_model(Expt_load)
	else:
		classifier = Classifier()
		model = classifier.prepare_to_init(2e-3)

		model = classifier.prepare_to_finetune(7e-6)
		model.fit(X_train, Y_train,epochs= 2,shuffle=True, validation_data=(X_val,Y_val), verbose=1, callbacks=callbacks1)

		print 'Finetuned the model'
		print 'Saving the model'
		
		model.save(Expt_save)	
    
	for i in range(Iterations):
		if t==1:
			model.fit(X_train, Y_train, epochs=epochs,shuffle=True,validation_data=(X_val,Y_val),verbose=1, callbacks=callbacks2)	
		active_learning()

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
	t=0
	if LOAD_MODEL==True:
		X_train = np.load('X_trainB.npy',encoding='bytes')
		Y_train = np.load('Y_trainB.npy',encoding='bytes')
		X_oracle = np.load('X_oracleB.npy',encoding='bytes')
		Y_oracle = np.load('Y_oracleB.npy',encoding='bytes')
		X_val = np.load('X_valB.npy',encoding='bytes')
		Y_val = np.load('Y_valB.npy',encoding='bytes')
		X_train_descriptor = np.load('X_train_descriptor.npy',encoding='bytes')
		X_oracle_descriptor = np.load('X_oracle_descriptor.npy',encoding='bytes')
	else:
		X_train,Y_train,X_oracle,Y_oracle,X_val,Y_val = train.get_data_iterators(batch_size=1,data_dir='mess_cb',target_size=(512, 512),rescale=1/255,fill_mode='constant',load_train_data=True,color_mode='rgb')
    
		X_train = np.reshape(X_train,(200,512,512,3))
		X_oracle = np.reshape(X_oracle,(568,512,512,3))
		X_val = np.reshape(X_val,(192,512,512,3))

		X_train_descriptor = calculate_descriptors(X_train)
		X_oracle_descriptor = calculate_descriptors(X_oracle)

		X_train = np.reshape(X_train,(200,3,512,512))
		X_oracle = np.reshape(X_oracle,(568,3,512,512))
		X_val = np.reshape(X_val,(192,3,512,512))		
		
	train_and_evaluate(batch_size=1, out_size=out_sizes[n_blocks], f_size=f_sizes[n_blocks],
                           exp_name=Expt_save, n_blocks=n_blocks,
                           context_blocks=context_blocks+1, img_dir=args.img_dir)
    
    
