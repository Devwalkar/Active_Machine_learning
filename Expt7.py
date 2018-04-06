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
#from eval.eval import get_predictions, get_auc_score
from keras.callbacks import ModelCheckpoint, EarlyStopping
#from utils.visualization import prepare_img_to_plot, parula_map
#from utils.visualization import get_gaussian_lesion_map as get_lesion_map
from keras.engine import Layer
from keras.layers import Input,Conv2D,GlobalAveragePooling2D,Dense,Flatten,Lambda
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import load_model,save_model,Model
from classifier import Classifier
from scipy.spatial.distance import cdist
from skimage.feature import corner_peaks,corner_harris,BRIEF,ORB
from skimage.color import rgb2gray
from skimage.filters import gaussian
from sklearn.utils import shuffle
import tensorflow as tf 
from keras import backend as K

######################################################################################################
Expt_load = 'exp_M_07.hdf5'
Expt_save = 'exp_M_07.hdf5'
LOAD_MODEL = False
No_images_to_add = 5
No_images_to_remove = 5
working_threshold = 4
No_of_images_to_start_with = 100
Iterations= 20

####################################################################################################3
'''
FUNCTION DEFINITIONS
'''
def remove_file_from_iterator(Indices):
	global X_train,Y_train,X_working,Y_working 	

	if len(X_working)==0:
		Addition = np.reshape(np.take(X_train,Indices,axis=0),(len(Indices),3,512,512))
		X_working = Addition
		Addition = np.take(Y_train,Indices,axis=0)
		Y_working = Addition
	else:	
		Addition = np.reshape(np.take(X_train,Indices,axis=0),(len(Indices),3,512,512))
		X_working = np.concatenate((X_working,Addition),axis=0)
		Addition = np.take(Y_train,Indices,axis=0)
		Y_working = np.append(Y_working,Addition)

	X_train = np.delete(X_train,Indices,axis=0)
	Y_train = np.delete(Y_train,Indices,axis=0)
    
	print ('X_train_length:',len(X_train))
	print ('X_working length:',len(X_working))

	np.save('X_train.npy',X_train)
	np.save('Y_train.npy',Y_train)
	np.save('X_working.npy',X_working)
	np.save('Y_working.npy',Y_working)

########################################################################################################333	
def add_file_to_train(Indices):
	global X_train,Y_train,X_oracle,Y_oracle

	Addition = np.reshape(np.take(X_oracle,Indices,axis=0),(len(Indices),3,512,512))
	X_train = np.concatenate((X_train,Addition),axis=0)
	Addition = np.take(Y_oracle,Indices,axis=0)
	Y_train = np.append(Y_train,Addition)

	X_oracle = np.delete(X_oracle,Indices,axis=0)
	Y_oracle = np.delete(Y_oracle,Indices,axis=0)

	print 'X_train_length:',len(X_train)
	print 'X_oracle length:',len(X_oracle)

	np.save('X_train.npy',X_train)
	np.save('Y_train.npy',Y_train)
	np.save('X_oracle.npy',X_oracle)
	np.save('Y_oracle.npy',Y_oracle)
	np.save('X_val.npy',X_val)
	np.save('Y_val.npy',Y_val)

##############################################################################################
def add_file_from_working(Indices):
	global X_train,Y_train,X_working,Y_working

	Addition = np.reshape(np.take(X_working,Indices,axis=0),(len(Indices),3,512,512))
	X_train = np.concatenate((X_train,Addition),axis=0)
	Addition = np.take(Y_working,Indices,axis=0)
	Y_train = np.append(Y_train,Addition)

	X_working = np.delete(X_working,Indices,axis=0)
	Y_working = np.delete(Y_working,Indices,axis=0)

	print 'X_train_length:',len(X_train)
	print 'X_working length:',len(X_working)

	np.save('X_train.npy',X_train)
	np.save('Y_train.npy',Y_train)
	np.save('X_working.npy',X_working)
	np.save('Y_working.npy',Y_working)
	np.save('X_val.npy',X_val)
	np.save('Y_val.npy',Y_val)
	#print 'Data saved'

########################################################################################################3
def calculate_descriptors(X):
	#extractor = BRIEF()
	descriptor_extractor = ORB(n_keypoints=500)
	Descriptors = []
	for i in range(len(X)):
		Im = np.asarray(X[i,:,:,:],dtype='float32')
		Max = np.amax(Im)
		Im = Im/Max
		Im = rgb2gray(Im)
		'''
		keypoints = corner_peaks(corner_harris(Im), min_distance=5)		
		extractor.extract(Im,keypoints)
		Temp = extractor.descriptors
		'''		
		descriptor_extractor.detect_and_extract(Im)
		Temp = descriptor_extractor.descriptors
		Descriptors.append(np.asarray(np.round(np.average(Temp,axis=0)),dtype='int32'))

	Descriptors_matrix = np.zeros([len(X),256])
	for i in range(len(X)):
		Descriptors_matrix[i,:] = Descriptors[i] 
     
	return Descriptors_matrix
#############################################################################################################
def calculate_distance(X_oracle_descriptor,X_train_descriptor):
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

######################################################################################################	
def create_trainset():
	global No_of_images_to_start_with,X_oracle,X_oracle_descriptor,Y_oracle
    
	Initial_sample = np.random.randint(0,len(X_oracle),1)

	X_train_descriptor =  np.reshape(X_oracle_descriptor[Initial_sample,:],(1,256))
	X_train = np.reshape(X_oracle[Initial_sample,:,:,:],(1,512,512,3))
	Y_train = Y_oracle[Initial_sample]

	X_oracle = np.delete(X_oracle,Initial_sample,axis=0)
	X_oracle_descriptor = np.delete(X_oracle_descriptor,Initial_sample,axis=0)
	Y_oracle = np.delete(Y_oracle,Initial_sample,axis=0)

	#print 'Creating training dataset:'
	for i in range(No_of_images_to_start_with-1):
		Dist,Index = calculate_distance(X_oracle_descriptor, X_train_descriptor)

		Selected_Index = Index[0]
        
		Addition = np.reshape(X_oracle[Selected_Index,:,:,:],(1,512,512,3))
		X_train = np.concatenate((X_train,Addition),axis=0)
	
		Addition = np.reshape(X_oracle_descriptor[Selected_Index,:],(1,256))
		X_train_descriptor = np.concatenate((X_train_descriptor,Addition),axis=0)

		X_oracle = np.delete(X_oracle,Selected_Index,axis=0)
		X_oracle_descriptor = np.delete(X_oracle_descriptor,Selected_Index,axis=0)

		Addition = np.reshape(Y_oracle[Selected_Index],(1,))
		Y_train = np.concatenate((Y_train,Addition),axis=0)
		Y_oracle = np.delete(Y_oracle,Selected_Index,axis=0)
   
	X_train,Y_train = shuffle(X_train,Y_train,random_state=62)

	return X_train,Y_train

#########################################################################################################33
def active_learning_working(working_prob):
	global working_threshold,X_working,Y_working

	working_prob = array(working_prob)
	working_loss = zeros((working_prob.shape))

	for i,prob in enumerate(working_prob):
		if prob == 1 or prob == 0:
			working_loss[i] = 0
		else:
			working_loss[i] = -Y_working[i]*log2(prob) - (1-Y_working[i])*log2(1-prob)

	Sorted_entropy = np.flip(np.sort(working_loss,axis=0),axis=0)
	Sorted_indexes = np.flip(np.argsort(working_loss,axis=0),axis=0)

	add_indices = []
	for i in range(len(X_working)):
		if Sorted_entropy[i]>=working_threshold:
			add_indices.append(Sorted_indexes[i])

	Sorted_entropy = np.take(Sorted_entropy,add_indices,axis=0)
	
	for i in range(len(Sorted_entropy)):

		print('Adding {0}th sample from working set having crossentropy: {1}'.format(i,Sorted_entropy[i]))

	add_file_from_working(add_indices)

#################################################################################################################333

def active_learning(oracle_gradients,train_prob):
	global No_images_to_add,No_images_to_remove, t,Y_train

	oracle_gradients = array(oracle_gradients)
	train_prob = array(train_prob)
	Train_Entrophy = zeros((train_prob.shape))

	for i,prob in enumerate(train_prob):
		if prob == 1 or prob == 0:
			Train_Entrophy[i] = 0
		else:
			Train_Entrophy[i] = -Y_train[i]*log2(prob) - (1-Y_train[i])*log2(1-prob)

	Sorted_entropy = np.sort(Train_Entrophy,axis=0)[:No_images_to_remove]
	Sorted_indexes = np.argsort(Train_Entrophy,axis=0)[:No_images_to_remove]

	remove_indices = []
	for i in range(No_images_to_remove):
		if ((train_prob[Sorted_indexes[i]]>0.5) and (Y_train[Sorted_indexes[i]]==0)) or ((train_prob[Sorted_indexes[i]]<0.5) and (Y_train[Sorted_indexes[i]]==1)):
			remove_indices.append(i)

	Sorted_entropy = np.delete(Sorted_entropy,remove_indices,axis=0)
	Sorted_indexes = np.delete(Sorted_indexes,remove_indices,axis=0)

	for i in range(len(Sorted_entropy)):

		print('Removing {0}th sample having crossentropy: {1}'.format(i,Sorted_entropy[i]))

	remove_file_from_iterator(Sorted_indexes)

	Sorted_gradients = np.flip(np.sort(oracle_gradients,axis=0),axis=0)[:No_images_to_add]
	Sorted_indexes = np.flip(np.argsort(oracle_gradients,axis=0),axis=0)[:No_images_to_add]

	for i in range(No_images_to_add):

		print('Adding {0}th sample having jacobian norm: {1}'.format(i,Sorted_gradients[i]))

	add_file_to_train(Sorted_indexes)
	t=1
###########################################################################################################

def train_and_evaluate(batch_size=8, img_dir='mess_cb',
					   out_size=32, f_size=31, horizontal_flip=True, vertical_flip=True,
                       width_shift_range=0.03, height_shift_range=0.03, rotation_range=360,
                       zoom_range=0.03, exp_name='exp0.hdf5', n_blocks=4, context_blocks=1,
                       patience=75):

	global X_train,Y_train,X_oracle,Y_oracle,X_val,Y_val,t,Iterations,X_working,Y_working			   

	checkpointer = ModelCheckpoint(filepath= exp_name, monitor = 'val_acc', verbose=1, save_best_only=True, save_weights_only=True)
	callbacks2 = [checkpointer]
    #####################################################################################################
    # Loading saved model here
	if LOAD_MODEL == True:
		classifier = Classifier()
		model = classifier.prepare_to_finetune(7e-5)
		model.load_weights(Expt_load)
	else:
		classifier = Classifier()
		model = classifier.prepare_to_finetune(7e-5)
		model.fit(X_train,Y_train,epochs=1,shuffle=True,batch_size=32, validation_data=(X_val,Y_val), verbose=1, callbacks=callbacks2)

#################################################################################################################
    # Defining jacobian computation here

	sess = K.get_session()
	model_grad = tf.gradients(model.output,model.input)
	#model_grad = K.gradients(K.dot(model.layers[-1].input,model.layers[-1].kernel)+model.layers[-1].bias, model.input)
	
	for i in range(Iterations):
		model.load_weights(Expt_load)
		
		if t==1:
			model.fit(X_train,Y_train,epochs=10,shuffle=True,batch_size=32,validation_data=(X_val,Y_val),verbose=1, callbacks=callbacks2)			
		
		W1 = np.squeeze(array(sess.run(model_grad,feed_dict={model.input:X_oracle[:int(len(X_oracle)/4),:,:,:]}),dtype=np.float32),axis=0)
		print 'W1 done'
		W2 = np.squeeze(array(sess.run(model_grad,feed_dict={model.input:X_oracle[int(len(X_oracle)/4):int(len(X_oracle)/2),:,:,:]}),dtype=np.float32),axis=0)
		print 'W2 done'
		W3 = np.squeeze(array(sess.run(model_grad,feed_dict={model.input:X_oracle[int(len(X_oracle)/2):int(0.75*len(X_oracle)),:,:,:]}),dtype=np.float32),axis=0)
		print 'W3 done'
		W4 = np.squeeze(array(sess.run(model_grad,feed_dict={model.input:X_oracle[int(0.75*len(X_oracle)):,:,:,:]}),dtype=np.float32),axis=0)
		print 'W4 done'

		Jacobian_oracle1 = np.sqrt(np.sum(np.sum(np.sum(np.square(W1),axis=-1),axis=-1),axis=-1))
		Jacobian_oracle2 = np.sqrt(np.sum(np.sum(np.sum(np.square(W2),axis=-1),axis=-1),axis=-1))
		Jacobian_oracle3 = np.sqrt(np.sum(np.sum(np.sum(np.square(W3),axis=-1),axis=-1),axis=-1))
		Jacobian_oracle4 = np.sqrt(np.sum(np.sum(np.sum(np.square(W4),axis=-1),axis=-1),axis=-1))

		Jacobian_oracle = np.concatenate((Jacobian_oracle1,Jacobian_oracle2,Jacobian_oracle3,Jacobian_oracle4),axis=0)

		train_prob = model.predict(X_train,verbose=1)			
												
		active_learning(Jacobian_oracle,train_prob)
		if len(X_working)!=0:
			working_prob = model.predict(X_working,verbose=1)
			active_learning_working(working_prob)
		
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
		X_working = np.load('X_working.npy',encoding='bytes') 
		Y_working = np.load('Y_working.npy',encoding='bytes')
	else:		
		X_train,Y_train,X_oracle,Y_oracle,X_val,Y_val = train.get_data_iterators(batch_size=1,data_dir='mess_cb',target_size=(512, 512),rescale=1/255,fill_mode='constant',load_train_data=True,color_mode='rgb')
    
		X_train = np.reshape(X_train,(200,512,512,3))
		X_oracle = np.reshape(X_oracle,(568,512,512,3))
		X_val = np.reshape(X_val,(192,3,512,512))

		X_oracle = np.concatenate((X_train,X_oracle),axis=0)
		Y_oracle = np.concatenate((Y_train,Y_oracle),axis=0)

		X_oracle,Y_oracle = shuffle(X_oracle,Y_oracle,random_state=42)

		X_oracle_descriptor = calculate_descriptors(X_oracle)

		X_train,Y_train = create_trainset()

		X_train = np.reshape(X_train,(No_of_images_to_start_with,3,512,512))
		X_oracle = np.reshape(X_oracle,(768-No_of_images_to_start_with,3,512,512))

		X_working = array([])
		Y_working = array([])

		np.save('X_train.npy',X_train)
		np.save('Y_train.npy',Y_train)
		np.save('X_oracle.npy',X_oracle)
		np.save('Y_oracle.npy',Y_oracle)
		np.save('X_val.npy',X_val)
		np.save('Y_val.npy',Y_val)

		t=0

	train_and_evaluate(batch_size=1, out_size=out_sizes[n_blocks], f_size=f_sizes[n_blocks],
                           exp_name=Expt_save, n_blocks=n_blocks,
                           context_blocks=context_blocks+1, img_dir=args.img_dir)
    
'''
		remove_indices = []
		for i in range(len(oracle_prob)):
			if ((oracle_prob[i]>0.5) and (Y_oracle[i]==0)) or ((oracle_prob[i]<0.5) and (Y_oracle[i]==1)):
				remove_indices.append(i)

		incorrect_Jacobian_oracle = np.take(Jacobian_oracle,remove_indices,axis=0)
		incorrect_oracle_prob = np.take(oracle_prob,remove_indices,axis=0)

		incorrect_Loss = []
		for i in range(len(incorrect_oracle_prob)):
			if (incorrect_oracle_prob[i]>0.5):
				incorrect_Loss.append(-1*log2(incorrect_oracle_prob[i]))
			elif (incorrect_oracle_prob[i]<0.5):
				incorrect_Loss.append(-1*log2(1-incorrect_oracle_prob[i]))

		incorrect_Loss = array(incorrect_Loss)	

		np.save('IL.npy',incorrect_Loss)
		np.save('IJN.npy',incorrect_Jacobian_oracle)

		oracle_prob = np.delete(oracle_prob,remove_indices,axis=0)
		Jacobian_oracle = np.delete(Jacobian_oracle,remove_indices,axis=0)

		Loss = []
		for i in range(len(oracle_prob)):
			if (oracle_prob[i]>0.5):
				Loss.append(-1*log2(oracle_prob[i]))
			elif (oracle_prob[i]<0.5):
				Loss.append(-1*log2(1-oracle_prob[i]))

		Loss = array(Loss)					

		np.save('L.npy',Loss)
		np.save('JN.npy',Jacobian_oracle)
		print 'arrays saved'
'''		