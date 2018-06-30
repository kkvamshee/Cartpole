#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 17:18:52 2018

@author: kkvamshee
"""

import numpy as np

def discount_rewards(rewards, discount_rate):
    rewards = np.array(rewards)
    after_effects = np.zeros(rewards.size)
    after_effects[-2] = rewards[-1] * discount_rate
    for step in np.arange(rewards.size-2)[::-1]:
        after_effects[step] = (rewards[step+1] + after_effects[step+1]) * discount_rate
    discounted_rewards = rewards + after_effects
    return discounted_rewards

def normalize_rewards(rewards):
    flat = np.concatenate(rewards)
    avg = flat.mean()
    dev = flat.std()
    return [(reward-avg)/dev for reward in rewards]
