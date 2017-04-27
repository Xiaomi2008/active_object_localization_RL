from __future__ import print_function

import argparse
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
import sys
import random
import numpy as np
from collections import deque

import json
# from keras import initializations
# from keras.initializations import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD , Adam

from object_localization_env import nature_object_env, neuron_object_env
import deepmodel
import ipdb
from keras.layers import Input
from keras.callbacks  import CSVLogger, ModelCheckpoint, TensorBoard, EarlyStopping
from keras.models import Model
from keras import backend as K
from keras.optimizers import RMSprop, SGD,Adam
import matplotlib.pyplot as plt
from keras import metrics
from keras import losses

CONFIG = 'nothreshold'
ACTIONS = 9 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 3000. # timesteps to observe before training
EXPLORE = 3000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.18 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-5
ACTION_ALPHA  = 0.15
action_discription={0:'left',1:'right',2:'up',3:'bottom',4:'bigger',5:'smaller',6:'fatter',7:'toller',8:'triger'}
netwrok_model={}
netwrok_model['res_vgg_model']=deepmodel.res_vgg_model
#Convert image into Black and white
img_channels = 1
def get_model_file(args):
    print(args)
    file_name =args.cnn_model+'_'+args.data+'_netinput_size'+str(args.cnn_input_size)+ '_act_alpha-'+str(ACTION_ALPHA)+'.h5'
    return file_name
def trainNetwork(model,args):
    # open up a game state to communicate with emulator
    # game_state = game.GameState()
    img_rows =img_cols = args.cnn_input_size
    print(img_rows)
    # ipdb.set_trace()
    if args.data =='neuron':
        object_loc_env=neuron_object_env()
    elif args.data =='nature':
        object_loc_env=nature_object_env()
    else:
        raise NameError(args.data  + ' :  No such env !' )
    object_loc_env.action_alpha=ACTION_ALPHA

    model_file =get_model_file(args)
    # store the previous observations in replay memory
    D =deque()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    # sipdb.set_trace()
    x_t, r_0, terminal = object_loc_env.localization_step(0)

    # x_t = skimage.color.rgb2gray(x_t)
    x_t = skimage.transform.resize(x_t,(img_rows,img_rows))
    x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))
    s_t=x_t
    action_history_st =[np.zeros([ACTIONS]) for i in range(10)]
    action_history_st_1 =[np.zeros([ACTIONS]) for i in range(10)]
    # ipdb.set_trace()
    # s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    #print (s_t.shape)

    #In Keras, need to reshape
    channels = 1 if len(s_t.shape) <=2 else s_t.shape[2]
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], channels)  #1*80*80*1 or 3
    s_at=np.array(action_history_st)
    s_at=np.reshape(s_at,(1,-1))
    S_Ts=[s_t,s_at]
    # ipdb.set_trace()

    

    if args.mode == 'Run':
        OBSERVE = 999999999    #We keep observe, never train
        epsilon = FINAL_EPSILON
        print ("Now we load weight")
        # model.load_weights("model.h5")
        model.load_weights(model_file)
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss='mse',optimizer=adam)
        print ("Weight load successfully")    
    else:                       #We go to training mode
        OBSERVE = OBSERVATION
        epsilon = INITIAL_EPSILON
        try:
            model.load_weights(model_file)
        except:
            print('can not find saved model file . {}'.format(model_file))
            pass
        # model.load_weights("model.h5")

    t = 0
    
    while (True):
        loss = 0
        Q_sa = 0
        action_index = 0
        r_t = 0
        a_t = np.zeros([ACTIONS])
        #choose an action epsilon greedy
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
            else:
                # q = model.predict(s_t)       #input a stack of 4 images, get the prediction
                q= model.predict(S_Ts)
                max_Q = np.argmax(q)
                action_index = max_Q
                a_t[max_Q] = 1
                # a_t= action_index
        #We reduced the epsilon gradually
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        #run the selected action and observed next state and reward
        x_t1, r_t, terminal = object_loc_env.localization_step(action_index)

        # x_t1 = skimage.color.rgb2gray(x_t1_colored)
        x_t1 = skimage.transform.resize(x_t1,(img_rows,img_rows))
        x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))
        channels = 1 if len(x_t1.shape) <=2 else x_t1.shape[2]
        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1],channels) #1x80x80x1
        s_t1=x_t1
        
        # s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

        # store the transition in D
        s_at1=np.array(action_history_st_1)
        s_at1=np.reshape(s_at1,(1,-1))
        S_Ts1=[s_t1,s_at1]
        D.append((S_Ts, action_index, r_t, S_Ts1,terminal))
        # D.append((s_t, action_index, r_t, s_t1, terminal))
        action_history_st.pop(0)
        action_history_st.append(a_t)

        if len(D) > REPLAY_MEMORY:
            D.popleft()

        #only train if done observing
        if t > OBSERVE:
            # object_loc_env.show_step()
            #sample a minibatch to train on
            minibatch = random.sample(D, BATCH)
            # ipdb.set_trace()
            # channels = 1 if len(s_t.shape) <=2 else s_t.shape[2]
            inputs = [np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[2])),\
                        np.zeros((BATCH, s_at1.shape[1]))]   #32, 80, 80, 1
            # print (inputs.shape)
            targets = np.zeros((inputs[0].shape[0], ACTIONS))                         #32, 9

            #Now we do the experience replay
            for i in range(0, len(minibatch)):
                state_t = minibatch[i][0]
                action_t = minibatch[i][1]   #This is action index
                reward_t = minibatch[i][2]
                state_t1 = minibatch[i][3]
                terminal = minibatch[i][4]
                # if terminated, only equals reward

                inputs[0][i:i + 1] = state_t[0]    #I saved down s_t
                inputs[1][i:i + 1] = state_t[1]

                targets[i] = model.predict(state_t)   # Hitting each buttom probability
                Q_sa = model.predict(state_t1)

                if terminal:
                    targets[i, action_t] = reward_t
                else:
                    targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)

            # targets2 = normalize(targets)
            loss += model.train_on_batch(inputs, targets)

        s_t = s_t1
        S_Ts = S_Ts1
        t = t + 1

        # save progress every 10000 iterations
        if t % 3000 == 0:
            print("Now we save model")
            # model.save_weights("model.h5", overwrite=True)
            # ipdb.set_trace()
            model.save_weights(model_file, overwrite=True)
            # with open("model.json", "w") as outfile:
            with open(model_file[:-4]+'.json',"w") as outfile:
                json.dump(model.to_json(), outfile)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state, \
            "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
            "/ Q_MAX " , np.max(Q_sa), "/ Loss ", loss)

    print("Episode finished!")
    print("************************")

def start(args):
    print(args)
    img_rows =args.cnn_input_size
    img_cols  = img_rows
    img_channels =3 if args.data =='nature' else 1
    input_shape=(img_rows,img_cols,img_channels)
    action_history_shape =(9*10,)
    output_shape =(ACTIONS,)
    # ips,out =deepmodel.res_vgg_model(input_shape, action_history_shape,output_shape)
    ips,out=netwrok_model[args.cnn_model](input_shape, action_history_shape,output_shape)
    model=Model(ips,out)
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse',optimizer=adam)
    model.summary()
    trainNetwork(model,args)


def main():
    parser = argparse.ArgumentParser(description='object detection ')
    parser.add_argument('-m','--mode', help='Train / Run', required=True)
    parser.add_argument('-d','--data',help='neuron/nature',default='neuron')
    parser.add_argument('-s','--cnn_input_size',help='network input size',type=int,default=80)
    parser.add_argument('-c','--cnn_model',help='deep q-net model',default='res_vgg_model')
    args = parser.parse_args()
    print(args)
    start(args)

if __name__ == "__main__":
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)
    # from keras import backend as K
    # K.set_session(sess)
    main()
