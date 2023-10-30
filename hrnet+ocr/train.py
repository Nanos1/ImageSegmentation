import os
import datetime
import numpy as np
from glob import glob
from scipy.io import loadmat
import tensorflow as tf

from loss import *
from model import *
from get_data import *

NUM_CLASSES = 1

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

#=============================================
	# Modelcheckpoint callback
	if not os.path.exists('/work2/pa21/ptzouv/gnanos/images_256/monitor_val_loss/hrnet+ocr/best_model'):
		os.makedirs('/work2/pa21/ptzouv/gnanos/images_256/monitor_val_loss/hrnet+ocr/best_model')

	if not os.path.exists('/work2/pa21/ptzouv/gnanos/images_256/monitor_val_loss/hrnet+ocr/model_save'):
	    os.makedirs('/work2/pa21/ptzouv/gnanos/images_256/monitor_val_loss/hrnet+ocr/model_save')
	filepath="/work2/pa21/ptzouv/gnanos/images_256/monitor_val_loss/hrnet+ocr/model_save/weights-{epoch:04d}.hdf5"
	checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=filepath,
		                               save_best_only=False,
		                               mode='auto', 
		                               monitor='val_loss')

	# Tensorboard Callback
	log_dir = os.path.join("/work2/pa21/ptzouv/gnanos/images_256/monitor_val_loss/hrnet+ocr/model", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
		                                              histogram_freq=1, 
		                                              write_graph=True)

#============================================

	mean_iou = MeanIoU(2, 0.5)
	callbacks_list = [checkpoint, tensorboard_callback]#,  tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=40)]
	

	x = tf.keras.layers.Input(shape = (256, 256, 3))
	out = hrnet18_v2(x, n_class = 1, include_top = True, mode = "ocr") #mode = seg > hrnet v2 + semantic segmentation, clsf > hrnet v2 + classifier, ocr > hrnet v2 + ocr + semantic segmentation
	out = tf.keras.layers.UpSampling2D((4, 4))(out)
	model = tf.keras.Model(x, out)
	
	loss = keras.losses.BinaryCrossentropy(from_logits=False)
	opt=keras.optimizers.Adam(learning_rate=0.001)
					
	metrics=[iou_loss, recall2, recall3, precision2, precision3, dice_loss, dice_loss2]
	model.compile(loss = loss, optimizer = opt, metrics = metrics)

	history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, callbacks=callbacks_list, verbose=10)
	# Best model	
	hrnet_ocr_model = tf.keras.models.load_model('/work2/pa21/ptzouv/gnanos/images_256/monitor_val_loss/hrnet+ocr/model_save/' + sorted(os.listdir('/work2/pa21/ptzouv/gnanos/images_256/monitor_val_loss/hrnet+ocr/model_save/'))[-1], custom_objects={"iou_loss": iou_loss, "recall2": recall2, "recall3": recall3, "precision2": precision2,"precision3": precision3, "dice_loss": dice_loss, "dice_loss2": dice_loss2})

	# save model
	hrnet_ocr_model.save("/work2/pa21/ptzouv/gnanos/images_256/monitor_val_loss/hrnet+ocr/best_model/hrnet+ocr.h5")

	hrnet_ocr_model.save_weights('/work2/pa21/ptzouv/gnanos/images_256/monitor_val_loss/hrnet+ocr/weights')
	print('Epochs run: ', len(history.history['loss']))
model_save_dir = '/work2/pa21/ptzouv/gnanos/images_256/monitor_val_loss/hrnet+ocr/best_model'
train(200, 8, model_save_dir)
