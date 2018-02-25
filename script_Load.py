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
from keras.models import load_model
#from keras.utils.np_utils import probas_to_classes

from classifier import Classifier
'''
FUNCTION DEFINITIONS
'''


def add_file_to_iterator(f, train_it, oracle_it):
	oracle_file = os.path.join('..', 'oracle', f)
	c, _ = os.path.split(f)
	if c == 'pathological':
		yi = 1
	else:
		yi = 0

	train_it.filenames.append(oracle_file)
	train_it.classes = np.append(train_it.classes, yi)
	train_it.nb_sample = len(train_it.filenames)
	train_it.n = len(train_it.filenames)
	train_it.batch_index = 0
	train_it.total_batches_seen = 0
	#train_it.index_generator = train_it._flow_index(train_it.n, train_it.batch_size, train_it.shuffle, None)
	#train_it.index_generator = train_it._flow_index()

	idx = oracle_it.filenames.index(f)
	oracle_it.filenames.remove(f)
	oracle_it.classes = np.delete(oracle_it.classes, idx)
	oracle_it.nb_sample = len(oracle_it.filenames)
	oracle_it.n = len(oracle_it.filenames)
	oracle_it.batch_index = 0
	oracle_it.total_batches_seen = 0
	#oracle_it.index_generator = oracle_it._flow_index(oracle_it.n, oracle_it.batch_size, oracle_it.shuffle, None)
	#oracle_it.index_generator = oracle_it._flow_index()

	train_it, oracle_it, val_it, test_it = train.get_data_iterators(batch_size=batch_size, horizontal_flip=horizontal_flip,
																	vertical_flip=vertical_flip, width_shift_range=width_shift_range,
																	height_shift_range=height_shift_range, rotation_range=rotation_range,
																	zoom_range=zoom_range, data_dir=img_dir, target_size=(512, 512),
																	rescale=1/255., fill_mode='constant', load_train_data=False, color_mode='rgb')


'''
active learning : 	active learning is implemented based on uncertainity sampling metrics
					Inputs taken : predicted probabilities along with their filenames,Directory path for messidor image and lesions
					Check the entrophy values of the predicted values.
					Add the image with high uncertainity/entrophy value from oracle directory to train dataset.
					return mode as 1.
					If there are images all have low entrophy values, then stop adding->mode =0
					path1 = current_file , path2 = destination_file
					os.rename(path1,path2)
'''
##############################################################################################

def active_learning(pred_prob, oracle_it, train_it, img_dir):

	names_dict = {}
	file_index = 0

	for file in oracle_it.filenames:
		names_dict[file_index] = file
		file_index = file_index + 1

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
		add_file_to_iterator(names_dict[max_entrophy_index], train_it, oracle_it)

	else:
		active_learning_mode = 0

	num_imgs_added = 1

	return [active_learning_mode, num_imgs_added]
###########################################################################################################

def train_and_evaluate(batch_size=8, img_dir='mess_cb',
					   out_size=32, f_size=31, horizontal_flip=True, vertical_flip=True,
                       width_shift_range=0.03, height_shift_range=0.03, rotation_range=360,
                       zoom_range=0.03, exp_name='exp0.hdf5', n_blocks=4, context_blocks=1,
                       patience=75):
	# Initialization
	active_learning_mode = 1
	epochs = 30
	oracle_len = 568    #71       #568
	best_loss = 10
	train_len = 200   #25            #200
	#test_len = 200

	train_it, oracle_it, val_it, test_it = train.get_data_iterators(batch_size=batch_size, horizontal_flip=horizontal_flip,
																	vertical_flip=vertical_flip, width_shift_range=width_shift_range,
																	height_shift_range=height_shift_range, rotation_range=rotation_range,
																	zoom_range=zoom_range, data_dir=img_dir, target_size=(512, 512),
																	rescale=1/255., fill_mode='constant', load_train_data=False, color_mode='rgb')

    checkpointer = ModelCheckpoint(filepath= exp_name, monitor = 'val_acc', verbose=1, save_best_only=True, save_weights_only=False)
	early = EarlyStopping(monitor = 'val_acc',patience=2, verbose=1)
	callbacks = [checkpointer,early]
    #####################################################################################################
    # Loading saved model here

	model = load_model('exp0.hdf5')

	for layer in model.layers:
		layer.trainable = True

	opt = Adam(lr=2e-4)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc'])	

	while active_learning_mode:
		print('Evaluating model {0}:'.format(exp_name))
		#model.load_weights(exp6.hdf5)
    	#model.load_weights(exp_name)		

		model.fit_generator(train_it, train_len, epochs, validation_data=val_it, validation_steps=192, verbose=2, callbacks=callbacks2)

	    # Make sure that you are not shuffling the prediction data in val_it
		pred_prob = model.predict_generator(oracle_it, oracle_len)
	
		active_learning_mode, num_imgs_added = active_learning(pred_prob, oracle_it, train_it, img_dir)
		print('Added {0} images.'.format(num_imgs_added))
		oracle_len = oracle_len - num_imgs_added
		train_len = train_len + num_imgs_added

	print('Train AUC = {0}'.format(get_auc_score(model, train_it, train_len)))
	print('Validation AUC = {0}'.format(get_auc_score(model, val_it, 192)))
	print('Test AUC = {0}'.format(get_auc_score(model, test_it, 240)))

'''
Main Program :  using argparse, add positional arguments. 1st directory should be of lesion data and second
                directory should be that of image data.
                Calculate the number of experiments to be performed. And design the outsize and f_size w.r.t experiment number
                [3,0],[3,1],[3,0],[3,1],[3,2],[4,0],[4,1],[4,3],[5,0],[5,1],[5,2],[5,3]
                There are 12 experiments overall. each experiment create a model weight exp{exp_no}.hdf5
                xtrain,xoracle,xvalid - shape (no of images,3,512,512)
'''
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
    for i, exp in enumerate(experiments):
        n_blocks, context_blocks = exp
     
	
    	target_size = (512, 512)
    	classes = ['normal', 'pathological']
    	color_mode = 'rgb'
    '''
    train_and_evaluate(out_size=out_sizes[n_blocks], f_size=f_sizes[n_blocks],
                           exp_name='exp6.hdf5', n_blocks=n_blocks,
                           context_blocks=context_blocks+1, img_dir=args.img_dir)
    
    
    
