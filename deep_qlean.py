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
import ipdb

CONFIG = 'nothreshold'
ACTIONS = 9 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 3200. # timesteps to observe before training
EXPLORE = 3000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4

img_rows , img_cols = 80, 80
#Convert image into Black and white
img_channels = 1
def vgg_model():
    vgg16_model = Sequential()
    vgg16_model.add(ZeroPadding2D((1, 1), input_shape=(img_rows,img_cols,img_channels)))
    vgg16_model.add(Convolution2D(64, 3, 3, activation='relu'))
    vgg16_model.add(ZeroPadding2D((1, 1)))
    vgg16_model.add(Convolution2D(64, 3, 3, activation='relu'))
    vgg16_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    vgg16_model.add(ZeroPadding2D((1, 1)))
    vgg16_model.add(Convolution2D(128, 3, 3, activation='relu'))
    vgg16_model.add(ZeroPadding2D((1, 1)))
    vgg16_model.add(Convolution2D(128, 3, 3, activation='relu'))
    vgg16_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    vgg16_model.add(ZeroPadding2D((1, 1)))
    vgg16_model.add(Convolution2D(256, 3, 3, activation='relu'))
    vgg16_model.add(ZeroPadding2D((1, 1)))
    vgg16_model.add(Convolution2D(256, 3, 3, activation='relu'))
    vgg16_model.add(ZeroPadding2D((1, 1)))
    vgg16_model.add(Convolution2D(256, 3, 3, activation='relu'))
    vgg16_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    vgg16_model.add(ZeroPadding2D((1, 1)))
    vgg16_model.add(Convolution2D(512, 3, 3, activation='relu'))
    vgg16_model.add(ZeroPadding2D((1, 1)))
    vgg16_model.add(Convolution2D(512, 3, 3, activation='relu'))
    vgg16_model.add(ZeroPadding2D((1, 1)))
    vgg16_model.add(Convolution2D(512, 3, 3, activation='relu'))
    # vgg16_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # vgg16_model.add(ZeroPadding2D((1, 1)))
    # vgg16_model.add(Convolution2D(512, 3, 3, activation='relu'))
    # vgg16_model.add(ZeroPadding2D((1, 1)))
    # vgg16_model.add(Convolution2D(512, 3, 3, activation='relu'))
    # vgg16_model.add(ZeroPadding2D((1, 1)))
    # vgg16_model.add(Convolution2D(512, 3, 3, activation='relu'))
    # vgg16_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    vgg16_model.add(Flatten())
    vgg16_model.add(Dense(4096, activation='relu'))
    vgg16_model.add(Dropout(0.5))
    vgg16_model.add(Dense(4096, activation='relu'))
    vgg16_model.add(Dropout(0.5))
    vgg16_model.add(Dense(ACTIONS))
    adam = Adam(lr=LEARNING_RATE)
    vgg16_model.compile(loss='mse',optimizer=adam)
    print("We finish building the model")
    return vgg16_model
def buildmodel():
    print("Now we build the model")
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same',input_shape=(img_rows,img_cols,img_channels)))  #80*80*4
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(ACTIONS))
   
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse',optimizer=adam)
    print("We finish building the model")
    return model

def trainNetwork(model,args):
    # open up a game state to communicate with emulator
    # game_state = game.GameState()
    object_loc_env=neuron_object_env()

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

    # s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    #print (s_t.shape)

    #In Keras, need to reshape
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], 1)  #1*80*80*1

    

    if args['mode'] == 'Run':
        OBSERVE = 999999999    #We keep observe, never train
        epsilon = FINAL_EPSILON
        print ("Now we load weight")
        model.load_weights("model.h5")
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss='mse',optimizer=adam)
        print ("Weight load successfully")    
    else:                       #We go to training mode
        OBSERVE = OBSERVATION
        epsilon = INITIAL_EPSILON
        model.load_weights("model.h5")

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
                # a_t[action_index] = 1
                a_t= action_index
            else:
                q = model.predict(s_t)       #input a stack of 4 images, get the prediction
                max_Q = np.argmax(q)
                action_index = max_Q
                # a_t[max_Q] = 1
                a_t= action_index

        #We reduced the epsilon gradually
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        #run the selected action and observed next state and reward
        x_t1, r_t, terminal = object_loc_env.localization_step(a_t)

        # x_t1 = skimage.color.rgb2gray(x_t1_colored)
        x_t1 = skimage.transform.resize(x_t1,(80,80))
        x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))

        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1) #1x80x80x1
        s_t1=x_t1
        # s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

        # store the transition in D
        D.append((s_t, action_index, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        #only train if done observing
        if t > OBSERVE:
            object_loc_env.show_step()
            #sample a minibatch to train on
            minibatch = random.sample(D, BATCH)



            inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], 1))   #32, 80, 80, 1
            print (inputs.shape)
            targets = np.zeros((inputs.shape[0], ACTIONS))                         #32, 2

            #Now we do the experience replay
            for i in range(0, len(minibatch)):
                state_t = minibatch[i][0]
                action_t = minibatch[i][1]   #This is action index
                reward_t = minibatch[i][2]
                state_t1 = minibatch[i][3]
                terminal = minibatch[i][4]
                # if terminated, only equals reward

                inputs[i:i + 1] = state_t    #I saved down s_t

                targets[i] = model.predict(state_t)  # Hitting each buttom probability
                Q_sa = model.predict(state_t1)

                if terminal:
                    targets[i, action_t] = reward_t
                else:
                    targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)

            # targets2 = normalize(targets)
            loss += model.train_on_batch(inputs, targets)

        s_t = s_t1
        t = t + 1

        # save progress every 10000 iterations
        if t % 4000 == 0:
            print("Now we save model")
            model.save_weights("model.h5", overwrite=True)
            with open("model.json", "w") as outfile:
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
    # model = buildmodel()
    model =vgg_model()
    trainNetwork(model,args)

def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-m','--mode', help='Train / Run', required=True)
    args = vars(parser.parse_args())
    start(args)

if __name__ == "__main__":
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)
    # from keras import backend as K
    # K.set_session(sess)
    main()
