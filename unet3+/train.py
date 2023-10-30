import os
import datetime
import numpy as np
from glob import glob
from scipy.io import loadmat
import tensorflow as tf

from loss import *
from model import *
from get_data import *

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=1.0)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session()

IMAGE_SIZE = 128
NUM_CLASSES = 1

DATA_DIR_TRAIN = '/work2/pa21/ptzouv/gnanos/images_128/dataset_128/train/'
DATA_DIR_VAL = '/work2/pa21/ptzouv/gnanos/images_128/dataset_128/val/'

INPUT_SHAPE = [128, 128, 3]
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

#=============================================
        # Modelcheckpoint callback
	if not os.path.exists('/work2/pa21/ptzouv/gnanos/images_128/monitor_val_loss/unet3+/best_model'):
		os.makedirs('/work2/pa21/ptzouv/gnanos/images_128/monitor_val_loss/unet3+/best_model')

	if not os.path.exists('/work2/pa21/ptzouv/gnanos/images_128/monitor_val_loss/unet3+/model_save'):
		os.makedirs('/work2/pa21/ptzouv/gnanos/images_128/monitor_val_loss/unet3+/model_save')
	filepath="/work2/pa21/ptzouv/gnanos/images_128/monitor_val_loss/unet3+/model_save/weights-{epoch:04d}.hdf5"
	checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=filepath,
                                               save_best_only=False,
                                               mode='auto',
                                               monitor='val_loss')

        # Tensorboard Callback
	log_dir = os.path.join("/work2/pa21/ptzouv/gnanos/images_128/monitor_val_loss/unet3+/model", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                              histogram_freq=1,
                                                              write_graph=True)

#============================================

	callbacks_list = [checkpoint, tensorboard_callback]#, tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=40)]
	mean_iou = MeanIoU(2, 0.5)
	optimizer = keras.optimizers.Adam(learning_rate=0.001)
	loss = keras.losses.BinaryCrossentropy(from_logits=False)
	model = UNet_3Plus(INPUT_SHAPE, OUTPUT_CHANNELS)

	model.compile(
	    optimizer=optimizer,
	    loss=loss,
	metrics = [iou_loss, recall2, recall3, precision2, precision3, dice_loss, dice_loss2, unet3p_hybrid_loss, focal_loss, ssim_loss]
	)
	print('-'*20, 'Begin Training', '-'*20)
	history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, callbacks=callbacks_list, verbose=10)
	print('-'*20, 'Finish Training', '-'*20)
        # Best model    
	model = tf.keras.models.load_model('/work2/pa21/ptzouv/gnanos/images_128/monitor_val_loss/unet3+/model_save/' + sorted(os.listdir('/work2/pa21/ptzouv/gnanos/images_128/monitor_val_loss/unet3+/model_save'))[-1], custom_objects={"iou_loss": iou_loss, "recall2": recall2, "recall3": recall3, "precision2": precision2,"precision3": precision3, "dice_loss": dice_loss, "dice_loss2": dice_loss2, "unet3p_hybrid_loss": unet3p_hybrid_loss, "focal_loss": focal_loss, "ssim_loss": ssim_loss})
        # save model
	model.save("/work2/pa21/ptzouv/gnanos/images_128/monitor_val_loss/unet3+/best_model/unet3+.h5")

	model.save_weights('/work2/pa21/ptzouv/gnanos/images_128/monitor_val_loss/unet3+/weights')
	print('Epochs run: ', len(history.history['loss']))

model_save_dir = '/work2/pa21/ptzouv/gnanos/images_128/monitor_val_loss/unet3+/best_model'
train(200, 8, model_save_dir)
##


