#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 16:58:29 2018

@author: kkvamshee
"""
import os
import time

import numpy as np

import gym
import universe

env = gym.make('CartPole-v0')

score = 0
obs = env.reset()
while True:
    env.render()
    if obs[3]>0:
        action = 1
    else:
        action = 0
    print(score, action)
    obs, reward, done, _ = env.step(action)
    score += reward
    if done:
        break
print(score)
env.close()