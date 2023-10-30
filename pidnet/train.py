import os
import datetime
import numpy as np
from glob import glob
from scipy.io import loadmat

import tensorflow as tf

from loss import *
from model import *
from get_data import *

IMAGE_SIZE = 256
NUM_CLASSES = 1

DATA_DIR_TRAIN = '/work2/pa21/ptzouv/gnanos/images_256/dataset_256/train/'
DATA_DIR_VAL = '/work2/pa21/ptzouv/gnanos/images_256/dataset_256/val/'

INPUT_SHAPE = [256, 256, 3]
OUTPUT_CHANNELS = 1

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



	mean_iou = MeanIoU(2, 0.5)
	optimizer = keras.optimizers.Adam(learning_rate=0.001)
	loss = keras.losses.BinaryCrossentropy(from_logits=False)
	model = get_pred_model("pidnet_m", INPUT_SHAPE, OUTPUT_CHANNELS)
	if not os.path.exists(model_save_dir):
	    os.makedirs(model_save_dir)
	checkpoint_filepath = os.path.join(model_save_dir, 'best_weights')

	model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
		filepath=checkpoint_filepath,
		save_weights_only=True,
		monitor='val_loss',
		mode='auto',
		save_best_only=True)

	model.compile(
	    optimizer=optimizer,
	    loss=loss,
	    metrics = [iou_loss, recall2, recall3, precision2, precision3, dice_loss, dice_loss2],

	)
	print('-'*20, 'Begin Training', '-'*20)
	history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, callbacks=[model_checkpoint_callback])
	model.save_weights('/work2/pa21/ptzouv/gnanos/images_256/monitor_val_loss/pidnet/weights')

	print('Epochs run: ', len(history.history['loss']))

model_save_dir = '/work2/pa21/ptzouv/gnanos/images_256/monitor_val_loss/pidnet/best_model'
train(200, 8, model_save_dir)
###


