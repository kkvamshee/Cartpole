#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 17:06:21 2018

@author: kkvamshee
"""
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, xavier_initializer

from utils import *




n_inputs = 4
n_hidden_1 = 4
n_hidden_2 = 4
n_outputs = 1

initializer = xavier_initializer()

X = tf.placeholder(tf.float32, (None, 4), name='input layer')
hidden_1 = fully_connected(n_inputs, n_hidden_1,weights_initializer=initializer,
                           activation_fn=tf.nn.relu6, name='hidden layer 1')
hidden_2 = fully_connected(n_hidden_1, n_hidden_2, weights_initializer=initializer,
                           activation_fn=tf.nn.relu6, name='hidden layer 2')
logits = fully_connected(n_hidden_2, n_outputs, name='final layer')

outputs = tf.nn.sigmoid(logits, name='prediction layer')
predict_proba = tf.concat(values=[1-outputs, outputs])
actions = tf.multinomial(np.log(predict_proba), num_samples=1)

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
saver = tf.train.saver()



#execution phase


n_iterations = 1
n_max_steps = 500
#because we dont want our algorithm to get stuck in an infinite game
n_games_per_update = 15
discount_rate = 2**(-1/10)

env = gym.make('CartPole-v0')

with tf.Session as sess:
	init.run()
	for iteration in range(n_iterations):
		iter_start = time.time()
		iter_rewards = []
		iter_gradients = []
		for game in range(n_games_per_update):
			game_rewards = []
			game_gradients = []
			obs = env.reset()
			for step in range(n_max_steps):
				action, gradient = sess.run([actions, grads],
					feed_dict={X:obs.reshape(1, n_inputs)})
				obs, reward, done, _ = obs.step(action)
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

		discounted_rewards = discount_rewards(iter_rewards, discount_rate)
		normalized_rewards = normalized_rewards(dicounted_rewards)
		grad_dict = {}

		for var_idx, grad_placeholder in enumerate(gradient_placeholders):
			mean_gradients = np.mean([reward*iter_gradients[game_index][step][var_idx]
				for game_index, rewards in enumerate(normalized_rewards)
				for step, reward in enumerate(rewards)], axis=0)
			grad_dict[grad_placeholder] = mean_gradients
		sess.run(training_op, feed_dict=grad_dict)

		if iteration % 1 == 0:
			saver.save(sess, './checkpoints/net_iter_{0}.ckpt'.format(iteration))

		iter_end = time.time()
		print('time for iteration = {0}s'.format(iter_end - iter_start))




