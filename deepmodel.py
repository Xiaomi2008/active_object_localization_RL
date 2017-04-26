#kdd_deep_models
from keras.layers import merge, Dropout, Dense, Lambda, Flatten, Activation
from keras.layers.pooling import GlobalMaxPooling2D
from keras.layers.convolutional import MaxPooling1D,Convolution1D,MaxPooling2D, Convolution3D, Convolution2D, SeparableConv2D, AveragePooling2D, ZeroPadding2D, ZeroPadding3D, UpSampling2D, Deconvolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Reshape, Permute, Dense, Dropout
from keras.layers.recurrent import LSTM,GRU
from keras.layers.embeddings import Embedding
from keras.layers import Input
from keras import regularizers
# from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras import backend as K
from keras.models import Model
from keras.layers.merge import add, multiply, concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
ACTIONS =9

def res_vgg_model(input_shape_1,input_shape_2,output_shape):
	ip1 = Input(shape=input_shape_1,name = 'warp_image')
	ip2=  Input(shape=input_shape_2,name = 'action_history')
	conv1_1=Convolution2D(48, (3, 3), activation='relu', padding='same')(ip1)
	conv1_1=BatchNormalization()(conv1_1)
	conv1_2=Convolution2D(48, (3, 3), activation='relu', padding='same')(conv1_1)
	conv1_2=BatchNormalization()(conv1_2)
	
	pool1  =MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv1_2)
	conv2_branch=Convolution2D(96, (1, 1), padding='same')(pool1)
	conv2_1=Convolution2D(96, (3, 3), activation='relu', padding='same')(pool1)
	conv2_1=BatchNormalization()(conv2_1)
	conv2_2=Convolution2D(96, (3, 3), activation='relu', padding='same')(conv2_1)
	conv2_2=BatchNormalization()(conv2_2)
	conv2_3=Convolution2D(96, (3, 3),  padding='same')(conv2_2)
	conv2_merge=add([conv2_3,conv2_branch])
	conv2_merge=Activation('relu')(conv2_merge)
	
	pool2  =MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv2_merge)
	conv3_branch=Convolution2D(196, (1, 1), padding='same')(pool2)
	conv3_1=Convolution2D(196, (3, 3), activation='relu', padding='same')(pool2)
	conv3_1=BatchNormalization()(conv3_1)
	conv3_2=Convolution2D(196, (3, 3), activation='relu', padding='same')(conv3_1)
	conv3_2=BatchNormalization()(conv3_2)
	conv3_3=Convolution2D(196, (3, 3), padding='same')(conv3_2)
	conv3_merge=add([conv3_3,conv3_branch])
	conv3_merge=Activation('relu')(conv3_merge)
	
	pool3  =MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv3_merge)
	conv4_branch=Convolution2D(378, (1, 1), padding='same')(pool3)
	conv4_1=Convolution2D(378, (3, 3), activation='relu', padding='same')(pool3)
	conv4_1=BatchNormalization()(conv4_1)
	conv4_2=Convolution2D(378, (3, 3), activation='relu', padding='same')(conv4_1)
	conv4_2=BatchNormalization()(conv4_2)
	conv4_3=Convolution2D(378, (3, 3),  padding='same')(conv4_2)
	conv4_merge=add([conv4_3,conv4_branch])
	conv4_merge=Activation('relu')(conv4_merge)

	ft =Flatten()(conv4_merge)
	d1=Dense(2048, activation='relu')(ft)
	d1=Dropout(0.5)(d1)
	d1=concatenate([d1,ip2])
	d2=Dense(2048, activation='relu')(d1)
	d2=Dropout(0.5)(d2)
	d3=Dense(1024, activation='relu')(d2)
	d3=Dropout(0.5)(d3)
	out=Dense(ACTIONS)(d3)
	return (ip1,ip2), out
	

    # adam = Adam(lr=LEARNING_RATE)
    # vgg16_model.compile(loss='mse',optimizer=adam)
    # print("We finish building the model")
    # return vgg16_model