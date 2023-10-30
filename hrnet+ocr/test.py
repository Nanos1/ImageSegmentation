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

weights_list = glob('/work2/pa21/ptzouv/gnanos/images_256/monitor_val_loss/hrnet+ocr/model_save/*')
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

	mean_iou = MeanIoU(2, 0.5)

	x = tf.keras.layers.Input(shape = (256, 256, 3))
	out = hrnet18_v2(x, n_class = 1, include_top = True, mode = "ocr") #mode = seg > hrnet v2 + semantic segmentation, clsf > hrnet v2 + classifier, ocr > hrnet v2 + ocr + semantic segmentation
	out = tf.keras.layers.UpSampling2D((4, 4))(out)
	model = tf.keras.Model(x, out)
	
	loss = keras.losses.BinaryCrossentropy(from_logits=False)

	opt=keras.optimizers.Adam(learning_rate=0.001)
					
	metrics= [MeanIoU(num_classes=2), iou_loss, Precision(), precision2, precision3, Recall(), recall2, recall3, dice_loss, dice_loss2, dice_loss3],
	model.load_weights(weight)#.expect_partial()
	print('loading model weights')
	model.compile(
	    optimizer=keras.optimizers.Adam(learning_rate=0.001),
	    loss=loss,
	    metrics=metrics,
	
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
	if not os.path.exists('hrnet+ocr/figures'):
		os.makedirs('hrnet+ocr/figures')
	r = 0
	for image, mask in test_dataset.take(2):
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
	    ax3.title.set_text('HRNet + OCR Prediction')
	    ax3.imshow(pred_mask_h[0,:,:,0], cmap='gray')

	    plt.savefig(os.path.join('hrnet+ocr/figures', 'prediction_fig_' + str(i)+ '_' + str(r) + '.png'))

	
	return(result[1], result[4], result[6], result[11])

test(64, '/work2/pa21/ptzouv/gnanos/images_256/monitor_val_loss/hrnet+ocr/model_save/weights')
