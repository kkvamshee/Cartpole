#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 19:35:04 2018

@author: kkvamshee
"""
import os
import time

import gym
import universe
import numpy as np
import cv2

env = gym.make('CartPole-v0')
obs = env.reset()

score = 0
done = False
action=0
while not done:
    env.render()
    cv2.imshow('1', np.zeros((5, 4), dtype='uint8'))
    key = cv2.waitKey(0) & 0xff
    if key==81:
        action=0
    elif key==83:
        action=1
    obs, reward, done, _ = env.step(action)
    print(key, action)
    score += reward
print(score)
env.close()
cv2.destroyAllWindows()