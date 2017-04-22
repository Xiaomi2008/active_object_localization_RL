import h5py
import numpy as np
import random
import os.path
import ipdb
def bb_intersection_over_union(boxA, boxB):
 		# code from http://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
		# determine the (x, y)-coordinates of the intersection rectangle
		# ipdb.set_trace()
		xA = max(boxA[0], boxB[0])
		yA = max(boxA[1], boxB[1])
		xB = min(boxA[2], boxB[2])
		yB = min(boxA[3], boxB[3])
	 
		# compute the area of intersection rectangle
		interArea = (xB - xA + 1) * (yB - yA + 1)
	 
		# compute the area of both the prediction and ground-truth
		# rectangles
		boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
		boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	 
		# compute the intersection over union by taking the intersection
		# area and dividing it by the sum of prediction + ground-truth
		# areas - the interesection area
		iou = interArea / float(boxAArea + boxBArea - interArea)
	 
		# return the intersection over union value
		return iou
def index_of_largest_IOU(input_bbox,all_bbox):
	max_iou=0
	i_bbox=[input_bbox[0][0],input_bbox[0][1],input_bbox[1][0],input_bbox[1][1]]
	for i,each_box in enumerate(all_bbox):
		# ipdb.set_trace()
		box=[each_box[0][0],each_box[0][1],each_box[1][0],each_box[1][1]]
		iou=bb_intersection_over_union(i_bbox,box)
		if max_iou< iou:
			max_iou=iou
			index = i
	return i
def bbox_to_4_scaler_list(bbox):
	return [bbox[0][0],bbox[0][1],bbox[1][0],bbox[1][1]]
def deform_bbox(bbox,x_size,y_size,dx1,dy1,dx2,dy2):
	print('dx1={}  dx2={}'.format(dx1,dx2))
	print('dy1={}  dy2={}'.format(dy1,dy2))
 	assert(abs(dx1)==abs(dx2))
 	assert(abs(dy1)==abs(dy2))
 	if bbox[0][0]+dx1 <0:
 		if dx1==dx2:
 			dx2=-bbox[0][0]
 		dx1=-bbox[0][0]
 	if bbox[1][0]+dx2>x_size-1:
 		if dx1==dx2:
 			dx1=x_size-bbox[1][0]-1
 		dx2==x_size-bbox[1][0]-1
 	if bbox[0][1]+dy1 <0:
 		if dy1==dy2:
 			dy2=-bbox[0][1]
 		dy1=-bbox[0][1]
 	if bbox[1][1]+dy2>y_size-1:
 		if dy1==dy2:
 			dy1=y_size-bbox[1][1]-1
 		dy2==y_size-bbox[1][1]-1 

 	new_bbox=[[bbox[0][0]+dx1,bbox[0][1]+dy1],[bbox[0][1]+dx2,bbox[1][1]+dy2]]
 	return new_bbox
def warp_image(image, bbox):
	x_size=image.shape[0]
	y_size=image.shape[1]
	assert(bbox[1][0]<x_size)
	assert(bbox[1][1]<y_size)
	warp_im = image[bbox[0][0]:bbox[1][0],bbox[0][1]:bbox[1][1]]
	return warp_im

class object_localization_env(object):
 	def __init__(self):
 		self.data_obj =None
 		self.init_box_x_range = [50,100]
 		self.init_box_y_range = [50,100]
 		self.cur_bbox =None
		self.objective_bbox =None
		self.action_alpha=0.2
		self.previous_IOU = None
		self.tal =  0.6
		self.eta =  3.0 
 	def getName(self):
 		return 'base_env'
 	def localization_step(self,action):
 		if action != 8:
 			# ipdb.set_trace()
 			self.update_current_bbox_with_action(action)
 			# ipdb.set_trace()
 			reward = self.get_reward()
 		else:
 			# for trigger action
 			reward = self.eta \
 			if bb_intersection_over_union(bbox_to_4_scaler_list(self.objective_bbox),bbox_to_4_scaler_list(self.cur_bbox)) > self.tal else -1*self.eta
 		bbox_warp_image =self.get_cur_bbox_warp_image()
 		terminal = True if action ==8 else False
 		return bbox_warp_image,reward, terminal
 		# warp_image(image, bbox):
 	# def get_cur_bbox_warp_image():
 	# 	pass

 	def get_reward(self):
 		cur_iou = bb_intersection_over_union(bbox_to_4_scaler_list(self.objective_bbox),bbox_to_4_scaler_list(self.cur_bbox))
 		reward = 1 if cur_iou -self.previous_IOU else -1
 		self.previous_IOU =cur_iou
 		return reward

 	def update_current_bbox_with_action(self,action):
 		alpha_w = int(round(self.action_alpha *(self.cur_bbox[1][0]-self.cur_bbox[0][0])))
 		alpha_h = int(round(self.action_alpha *(self.cur_bbox[1][1]-self.cur_bbox[1][0])))
 		if action == 0:
 			# move left
 			# self.cur_bbox = [[int(round(self.cur_bbox[0][0]-alpha_w)),self.cur_bbox[1][0]],
 			# 				[int(round(self.cur_bbox[1][0]-alpha_w)),self.cur_bbox[1][0]]]
 			dx1=-alpha_w
 			dy1=0
 			dx2=-alpha_w
 			dy2=0
 		elif action ==1:
 			#move right
 			# self.cur_bbox = [[int(round(self.cur_bbox[0][0]+alpha_w)),self.cur_bbox[1][0]],
 							# [self.cur_bbox[1][0]+alpha_w,self.cur_bbox[1][0]]]
 			dx1=alpha_w
 			dy1=0
 			dx2=alpha_w
 			dy2=0
 		elif action ==2:
 			#move up
 			# self.cur_bbox = [[self.cur_bbox[0][0],int(round(self.cur_bbox[0][1]-alpha_h))],
 			# 				[self.cur_bbox[1][0],int(round(self.cur_bbox[1][1]-alpha_h))]]
 			dx1=0
 			dy1=-alpha_h
 			dx2=0
 			dy2=-alpha_h
 		elif action ==3:
 			#move_down
 			# self.cur_bbox = [[self.cur_bbox[0][0],int(round(self.cur_bbox[0][1]+alpha_h))],
 			# 				[self.cur_bbox[1][0],int(round(self.cur_bbox[1][1]+alpha_h))]]
 			dx1=0
 			dy1=alpha_h
 			dx2=0
 			dy2=alpha_h
 		elif action ==4:
 			#bigger
 			# self.cur_bbox =[[int(round(self.cur_bbox[0][0]-alpha_w/2)),int(round(self.cur_bbox[1][0]-alpha_h/2))],
 			# 				[int(round(self.cur_bbox[1][0]+alpha_w/2)),int(round(self.cur_bbox[1][0]+alpha_h/2))]]
 			dx1=int(round(-alpha_w/2.0))
 			dy1=int(round(-alpha_h/2.0))
 			dx2=int(round(alpha_w/2.0))
 			dy2=int(round(alpha_h/2.0))
 		elif action ==5:
 			#smaller
 			# self.cur_bbox =[[int(round(self.cur_bbox[0][0]+alpha_w/2)),int(round(self.cur_bbox[1][0]+alpha_h/2))],
 			# 				[int(round(self.cur_bbox[1][0]-alpha_w/2)),int(round(self.cur_bbox[1][0]-alpha_h/2))]]
 			dx1=int(round(alpha_w/2.0))
 			dy1=int(round(alpha_h/2.0))
 			dx2=int(round(-alpha_w/2.0))
 			dy2=int(round(-alpha_h/2.0))
 		elif action ==6:
 			# fatter 
 			# self.cur_bbox =[[self.cur_bbox[0][0],int(round(self.cur_bbox[1][0]+alpha_h/2))],
 							# [self.cur_bbox[1][0],int(round(self.cur_bbox[1][0]-alpha_h/2))]]
 			dx1=0
 			dy1=int(round(alpha_h/2))
 			dx2=0
 			dy2=-dy1
 		elif action ==7:
 			# Taller
 			# self.cur_bbox =[[int(round(self.cur_bbox[0][0]+alpha_w/2)),self.cur_bbox[1][0]+alpha_h/2],
 							# [int(round(self.cur_bbox[1][0]-alpha_w/2)),self.cur_bbox[1][0]-alpha_h/2]]
 			dx1=int(round(alpha_w/2))
 			dy1=0
 			dx2=-dx1
 			dy2=0
 		elif action==8:
 			# trigger
 			dx1=0
 			dy1=0
 			dx2=0
 			dy2=0
 		self.cur_bbox=deform_bbox(self.cur_bbox,self.x_size,self.y_size,dx1,dy1,dx2,dy2)


class nature_object_env(object_localization_env):
	def __init__(self):
		pass
	def getName(self):
		return 'neuron_localization_env'
	def localization_step(self,action):
		pass

class neuron_object_env(object_localization_env):
	def __init__(self):
		super(neuron_object_env,self).__init__()
		self.data_obj=RL_neuron_data()
		self.init_starting_box()
	def getName(self):
		return 'neuron_localization_env'
	def init_starting_box(self):
		slice_idx =2
		image,all_bbox  =	self.data_obj.get_image_with_boundingBox(slice_idx)
		self.x_size 	=	image.shape[0]
		self.y_size 	=	image.shape[1]
		x_left 			=	random.randint(0,self.x_size-self.init_box_x_range[1])
		x_right 		= 	x_left+random.randint(self.init_box_x_range[0], self.init_box_x_range[1])
		y_top 			=	random.randint(0,self.y_size-self.init_box_y_range[1])
		y_bottom 		= 	y_top+random.randint(self.init_box_y_range[0], self.init_box_y_range[1])
		ipdb.set_trace()
		self.cur_image  =   image
		self.cur_bbox 	=	[[x_left,y_top],[x_right,y_bottom]]
		ipdb.set_trace()
		self.objective_bbox =all_bbox[index_of_largest_IOU(self.cur_bbox,all_bbox)]
		# ipdb.set_trace()
		self.previous_IOU   = bb_intersection_over_union(bbox_to_4_scaler_list(self.cur_bbox),
														bbox_to_4_scaler_list(self.objective_bbox))
	# def localization_step(self,action):
		# return 
	def get_cur_bbox_warp_image(self):
		return  warp_image(self.cur_image,self.cur_bbox)




class RL_data(object):
	def getName(self):
 		return 'root_data'
 	def get_one_image(self):
 		pass
class RL_neuron_data(RL_data):
	def __init__(self):
		super(RL_neuron_data,self).__init__()
		self.data_file='./data/snemi3d_train_full_stacks_added_segLabel_v1.h5'
		h5f=h5py.File(self.data_file,'r')
		self.images=h5f['data']
		self.seg_labels=h5f['seg_label']
		self.x_size=self.images.shape[0]
		self.y_size=self.images.shape[1]
		self.z_size=self.images.shape[2]
	def getName(self):
 		return 'neuron_data'
 	def get_image_with_boundingBox(self,slice_idx):
 		image,seg_label=self.get_one_image(slice_idx)
 		bbox =self.get_bounding_box(seg_label)
 		return image,bbox
 	def get_one_image(self,slice_idx):
 		image=self.images[:,:,slice_idx]
 		seg_label=self.seg_labels[:,:,slice_idx]
 		return image, seg_label
 	def get_bounding_box(self,seg_label):
 		obj_bbox=[]
 		unique_objs=np.unique(seg_label)
 		# ipdb.set_trace()
 		unique_objs=unique_objs[np.nonzero(unique_objs)] # remove label 0 --neuron boundary
 		for each_seg_lb in unique_objs:
 			# each_obj_idx=seg_label==each_seg_lb
 			seg_index = np.where(seg_label==each_seg_lb)
 			left  	= min(seg_index[0])
 			right 	= max(seg_index[0])
 			top   	= min(seg_index[1])
 			bottom 	= max(seg_index[1])
 			obj_bbox.append([(left,top),(right,bottom)])
 		return obj_bbox
if __name__ == "__main__":
	# test class and functions:
	import matplotlib.pyplot as plt
	env = neuron_object_env()
	for i in range(100):
		action=random.randint(0,7)
		warp_x, reward, terminal=env.localization_step(action)
		print (warp_x.shape)
		plt.imshow(warp_x,cmap='gray')
		plt.draw()
		plt.pause(0.1)




