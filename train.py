#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 17:06:21 2018

@author: kkvamshee
"""
import time

import numpy as np
import gym
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, xavier_initializer

from utils import *




n_inputs = 4
n_hidden_1 = 4
n_hidden_2 = 4
n_outputs = 1

initializer = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape=[None, n_inputs], name='input_layer')
hidden_1 = fully_connected(X, n_hidden_1,weights_initializer=initializer,
                           activation_fn=tf.nn.elu, scope='hidden_layer_1')
logits = fully_connected(hidden_1, n_outputs, scope='final_layer')

logits = tf.reshape(logits, [-1])

outputs = tf.nn.sigmoid(logits, name='prediction_layer')
action = tf.round(outputs)

learning_rate = 0.01
optimizer = tf.train.AdamOptimizer(learning_rate)
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=action, logits=logits)
grads_and_vars = optimizer.compute_gradients(loss)
grads = [grad for grad,var in grads_and_vars]
gradient_placeholders = []
grads_and_vars_feed = []
for grad, variable in grads_and_vars:
	gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
	gradient_placeholders.append(gradient_placeholder)
	grads_and_vars_feed.append((gradient_placeholder, variable))
training_op = optimizer.apply_gradients(grads_and_vars_feed)

init = tf.global_variables_initializer()
saver = tf.train.Saver()



#execution phase
start = time.time()

n_iterations = 600
n_max_steps = 1000
#because we dont want our algorithm to get stuck in an infinite game
n_games_per_update = 100
discount_rate = 2**(-1/10)

env = gym.make('CartPole-v0')
env.reset()

with tf.Session() as sess:
	init.run()
	for iteration in range(n_iterations):
		iter_rewards = []
		iter_gradients = []
		for game in range(n_games_per_update):
			game_rewards = []
			game_gradients = []
			obs = env.reset()
			for step in range(n_max_steps):
				act, gradient = sess.run([action, grads],
					feed_dict={X:obs.reshape(1, n_inputs)})
				obs, reward, done, _ = env.step(int(act[0]))
				game_rewards.append(reward)
				game_gradients.append(gradient)

				if done:
					break
			#completion of an episode/game

			iter_rewards.append(game_rewards)
			iter_gradients.append(game_gradients)

		#We now collected rewards and gradients from each episode 
		#for 'n_games_per_update' episodes and lets apply these 
		# gradients and rewards to train our model.

		discounted_rewards = [discount_rewards(rewards, discount_rate) for rewards in iter_rewards]
		normalized_rewards = normalize_rewards(discounted_rewards)
		grad_dict = {}

		for var_idx, grad_placeholder in enumerate(gradient_placeholders):
			mean_gradients = np.mean([reward*iter_gradients[game_index][step][var_idx]
				for game_index, rewards in enumerate(normalized_rewards)
				for step, reward in enumerate(rewards)], axis=0)
			grad_dict[grad_placeholder] = mean_gradients
		sess.run(training_op, feed_dict=grad_dict)

		#if iteration % 150 == 0:
			#saver.save(sess, './checkpoints/net_iter_{0}.ckpt'.format(iteration))
		if iteration % n_games_per_update == 0:
			avg_score = np.concatenate(iter_rewards).sum()//n_games_per_update
			print('avg score after {0} iterations : {1:.1f}'.format(iteration, avg_score))
			#print('time for iteration = {0:.3f}s'.format(iter_end - iter_start))
print('total train time : {0}s'.format(time.time()-start))