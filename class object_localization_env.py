import h5py
import numpy as np
import random
import os.path
import ipdb
 class object_localization_env(object):
 	def __init__(self):
 		pass
 	def getName(self):
 		return 'root_env'
 	def localization_step(self,action):


class neuron_object_env(object_localization_env):
	def __init__(self):
		pass
	def getName(self):
		return 'neuron_localization_env'
	def localization_step(self,action):

class nature_obj_env(object_localization_env):
	def __init__(self):
		pass
	def getName(self):
		return 'neuron_localization_env'
	def localization_step(self,action):


class RL_data(object):
	def getName(self):
 		return 'root_data'
 	def get_one_image(self):
 		pass:
class RL_neuron_data(RL_data):
	def __init__(self):
		self.data_file='./data/snemi3d_train_full_stacks_added_segLabel_v1.h5'
		h5f=h5py.File(self.data_file,'r')
		self.images=h5f['data']
		self.seg_labels=h5f['seg_label']
		self.x_size=self.images.shape[0]
		self.y_size=self.images.shape[1]
		self.z_size=self.images.shape[2]
	def getName(self):
 		return 'neuron_data'
 	def get_one_image(self,slice_idx):
 		image=self.images[:,:,slice_idx]
 		seg_label=self.seg_labels[:,:,slice_idx]
 		return image, seg_label
 	def get_bounding_box(self,seg_label):
 		obj_bbox=[]
 		unique_objs=np.unique(seg_label)
 		unique_objs=unique_objs[unique_obj] # remove label 0 --neuron boundary
 		for each_seg_lb in unique_objs:
 			# each_obj_idx=seg_label==each_seg_lb
 			seg_index = np.where(seg_label==each_obj_idx)
 			left  	= min(seg_index[0])
 			right 	= max(seg_index[0])
 			top   	= min(seg_index[1])
 			bottom 	= max(seg_index[1])
 			obj_bbox.append([(left,top),(right,bottom)])
 		return obj_bbox




