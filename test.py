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
#from keras.utils.np_utils import probas_to_classes
######################################################################################################
Expt_load = 'exp01.hdf5'
Expt_save = 'exp03.hdf5'

from classifier import Classifier
'''
FUNCTION DEFINITIONS
'''

def add_file_to_iterator(f):
	global train_it,oracle_it,val_it,test_it
	oracle_file = os.path.join('..', 'oracle', f)
	c, _ = os.path.split(f)
	if c == 'pathological':
		yi = 1
	else:
		yi = 0

	train_it.filenames.append(oracle_file)
	train_it.classes = np.append(train_it.classes, yi)
	train_it.samples = len(train_it.filenames)
	train_it.n = len(train_it.filenames)
	assert len(train_it.filenames) == len(train_it.classes)
	train_it.batch_index = 0
	train_it.total_batches_seen = 0
	#train_it.index_generator = train_it._flow_index(train_it.n, train_it.batch_size, train_it.shuffle, None)
	#train_it.index_generator = train_it._flow_index()

	idx = oracle_it.filenames.index(f)
	oracle_it.filenames.remove(f)
	oracle_it.classes = np.delete(oracle_it.classes, idx)
	oracle_it.samples = len(oracle_it.filenames)
	assert len(oracle_it.filenames) == len(oracle_it.classes)
	oracle_it.n = len(oracle_it.filenames)
	oracle_it.batch_index = 0
	oracle_it.total_batches_seen = 0
	#oracle_it.index_generator = oracle_it._flow_index(oracle_it.n, oracle_it.batch_size, oracle_it.shuffle, None)
	#oracle_it.index_generator = oracle_it._flow_index()


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

train_it,oracle_it,val_it,test_it = train.get_data_iterators(batch_size=1,data_dir='mess_cb',target_size=(512, 512),rescale=1/255,fill_mode='constant',load_train_data=False,color_mode='rgb')

names_dict = {}
file_index = 0

for file in oracle_it.filenames:
		names_dict[file_index] = file
		file_index = file_index + 1

model = load_model(Expt_load)
W = [200,547]
#add_file_to_iterator(names_dict[0])
for i in range(2):
       pred_prob = model.predict_generator(oracle_it,verbose =1)
       add_file_to_iterator(names_dict[W[i]])
       print len(oracle_it)
       print len(train_it)
       print len(oracle_it.classes)
       print len(train_it.classes)
   
