import os
import datetime
import numpy as np
from glob import glob
from scipy.io import loadmat
import tensorflow as tf
from loss import *
from get_data import *

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, concatenate

import sys
sys.path.append('../')
from swin_layers import *
from transform_layers import *
from loss import *
from stacking import *

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=1.0)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session()

np.random.seed(42)

IMAGE_SIZE = 256
n_labels = 1 

DATA_DIR_TRAIN = '/work2/pa21/ptzouv/gnanos/images_256/dataset_256/train/'
DATA_DIR_VAL = '/work2/pa21/ptzouv/gnanos/images_256/dataset_256/val/'

def train(epochs, BATCH_SIZE, model_save_dir):
	batch_size = BATCH_SIZE
	train_images = sorted(glob(os.path.join(DATA_DIR_TRAIN, "images/*")))
	train_masks = sorted(glob(os.path.join(DATA_DIR_TRAIN, "masks/*")))

	val_images = sorted(glob(os.path.join(DATA_DIR_VAL, "images/*")))
	val_masks = sorted(glob(os.path.join(DATA_DIR_VAL, "masks/*")))


	train_dataset = data_generator(train_images, train_masks, batch_size)
	val_dataset = data_generator(val_images, val_masks, batch_size)

	print('-'*20,'Number of Training images ',len(train_images), '-'*20,)
	print('-'*20,'Number of Training masks ',len(train_masks), '-'*20,)

	print('-'*20,'Number of Validation images ',len(val_images), '-'*20,)
	print('-'*20,'Number of Validation masks ',len(val_masks), '-'*20,)

	print('-'*20,'Number of Training Batches:', len(list(train_dataset)), '-'*20)
	print('-'*20,'Number of Validation Batches:', len(list(val_dataset)), '-'*20)

	filter_num_begin = 256     # number of channels in the first downsampling block; it is also the number of embedded dimensions
	depth = 4                  # the depth of SwinUNET; depth=4 means three down/upsampling levels and a bottom level 
	stack_num_down = 2         # number of Swin Transformers per downsampling level
	stack_num_up = 2           # number of Swin Transformers per upsampling level
	patch_size = (4, 4)        # Extract 4-by-4 patches from the input image. Height and width of the patch must be equal.
	num_heads = [4, 8, 8, 8]   # number of attention heads per down/upsampling level
	window_size = [4, 2, 2, 2] # the size of attention window per down/upsampling level
	num_mlp = 512              # number of MLP nodes within the Transformer
	shift_window=True          # Apply window shifting, i.e., Swin-MSA


	# Input section
	input_size = (IMAGE_SIZE, IMAGE_SIZE, 3)
	IN = Input(input_size)

	# Base architecture
	X = swin_unet_2d_base(IN, filter_num_begin, depth, stack_num_down, stack_num_up, 
		              patch_size, num_heads, window_size, num_mlp, 
		              shift_window=shift_window, name='swin_unet')

	# Output section
	mean_iou = MeanIoU(2, 0.5)
	OUT = Conv2D(n_labels, kernel_size=1, use_bias=False, activation='softmax')(X) ### change this to sigmoid

	# Model configuration
	model = Model(inputs=[IN,], outputs=[OUT,])

	# Optimization
	# <---- !!! gradient clipping is important
	opt = keras.optimizers.Adam(learning_rate=1e-4, clipvalue=0.5)

	loss = keras.losses.BinaryCrossentropy(from_logits=False)
	if not os.path.exists('/work2/pa21/ptzouv/gnanos/images_256/monitor_val_loss/swintransformer/best_model'):
	    os.makedirs('/work2/pa21/ptzouv/gnanos/images_256/monitor_val_loss/swintransformer/best_model')
	checkpoint_filepath = '/work2/pa21/ptzouv/gnanos/images_256/monitor_val_loss/swintransformer/best_model/best_weights'
	model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    		filepath=checkpoint_filepath,
    		save_weights_only=True,
    		monitor='val_loss',
    		mode='auto',
    		save_best_only=True)
	model.compile(
	    optimizer= opt,
	    loss=loss,
	    metrics = [iou_loss, recall2, recall3, precision2, precision3, dice_loss, dice_loss2]	
	)
	print('-'*20, 'Begin Training', '-'*20)
	history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, callbacks=[model_checkpoint_callback], verbose=100)
	print('-'*20, 'Finish Training', '-'*20)
	
	model.save_weights('/work2/pa21/ptzouv/gnanos/images_256/monitor_val_loss/swintransformer/weights')
	print('Epochs run: ', len(history.history['loss']))
model_save_dir = '/work2/pa21/ptzouv/gnanos/images_256/monitor_val_loss/swintransformer/'
train(200, 8, model_save_dir)


