import os
from PIL import Image
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import itertools
import warnings
from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm
from math import sqrt, ceil
import tifffile as tif

from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.color import rgb2gray

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input,Average,Conv2DTranspose, SeparableConv2D,dot, UpSampling2D, Add, Flatten,Concatenate,Multiply,Conv2D, MaxPooling2D, Activation,AveragePooling2D, ZeroPadding2D,GlobalAveragePooling2D,multiply,DepthwiseConv2D,ZeroPadding2D, BatchNormalization, LeakyReLU
from tensorflow.keras import backend as K
from tensorflow.keras.layers import concatenate, Lambda
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam,RMSprop
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import *

from loss import *
from model import *
from get_data import *

np.random.seed(123)

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=1.0)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session()

IMAGE_SIZE = 256
NUM_CLASSES = 1
INPUT_SHAPE = [256, 256, 3]
OUTPUT_CHANNELS = 1
weights_list = glob('/work2/pa21/ptzouv/gnanos/images_256/monitor_val_loss/emanet/model_save/*')
def test(BATCH_SIZE, weight):
	batch_size = BATCH_SIZE
	DATA_DIR_TEST = '/work2/pa21/ptzouv/gnanos/images_256/dataset_256/test/'

	test_img_list = glob("/work2/pa21/ptzouv/gnanos/images_256/dataset_256/test/images/*")
	test_mask_list = glob('/work2/pa21/ptzouv/gnanos/images_256/dataset_256/test/masks/*')

	print("test images ",len(test_img_list))
	print("test masks ",len(test_mask_list))


	test_images = sorted(glob(os.path.join(DATA_DIR_TEST, "images/*")))
	test_masks = sorted(glob(os.path.join(DATA_DIR_TEST, "masks/*")))

	test_dataset = data_generator(test_images, test_masks, batch_size)

	print('-'*20,'Number of Test Batches:', len(list(test_dataset)), '-'*20)

	optimizer = keras.optimizers.Adam(learning_rate=0.001)
	loss = keras.losses.BinaryCrossentropy(from_logits=False)
	input_shape = INPUT_SHAPE
	model = EMANet(INPUT_SHAPE)
	model.load_weights(weight)#.expect_partial()
	print('loading model weights')
	model.compile(
	    optimizer=optimizer,
	    loss=loss,
	    metrics=[MeanIoU(num_classes=2), iou_loss, Precision(), precision2, precision3, Recall(), recall2, recall3, dice_loss, dice_loss2, dice_loss3],
	
	)

	print("Evaluate")
	result = model.evaluate(test_dataset, batch_size = batch_size, verbose=0) 
	dict(zip(model.metrics_names, result))
	print('-'*50)
	print("Final metrics are: ", result)
	print('-'*50)
	for i in range(1, len(model.metrics_names)):
	  print("%s: %.5f%%\n" % (model.metrics_names[i], result[i]*100))
	
	#=== create some figures
	plt.rcParams.update({'figure.max_open_warning': 0})
	if not os.path.exists('emanet/figures'):
	    os.makedirs('emanet/figures')
	r = 0
	for image, mask in test_dataset.take(4):
	  r +=1
	  for i in range(batch_size):
	    pred_mask_h  = model.predict(image[i][np.newaxis,:,:,:])
	    fig = plt.figure(figsize=(14,10))
	    ax1 = fig.add_subplot(141)
	    ax1.title.set_text('Original Image')
	    ax1.imshow(image[i])
	    
	    ax2 = fig.add_subplot(142)
	    ax2.title.set_text('Ground Truth')
	    ax2.imshow(mask[i][:,:,0], cmap='gray')
	    
	    ax3 = fig.add_subplot(143)
	    ax3.title.set_text('EMANet Prediction')
	    ax3.imshow(pred_mask_h[0,:,:,0], cmap='gray')

	    plt.savefig(os.path.join('emanet/figures', 'prediction_fig_' + str(i)+ '_' + str(r) + '.png'))

	return(result[1], result[4], result[6], result[9])

test(4, '/work2/pa21/ptzouv/gnanos/images_256/monitor_val_loss/emanet/model_save/weights')
