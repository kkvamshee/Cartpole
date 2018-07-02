import tensorflow as tf
import numpy as np
import gym


env = gym.make('CartPole-v0')
obs = env.reset()


with tf.Session() as sess:
	saver = tf.train.import_meta_graph('./checkpoints/net_iter_150.meta')
	saver.restore(sess, tf.train.latest_checkpoint('./checkpoints/'))

	graph = tf.get_default_graph()
	X = graph.get_tensor_by_name('input_layer:0')
	output = graph.get_tensor_by_name('prediction_layer:0')

	scores = []

	for i in range(100):
		score=0
		done = False
		env.reset()
		while not done:
			#env.render()
			out = sess.run(output,
				feed_dict={X:obs.reshape(1, 4)})
			action = int(np.round(out)[0])
			obs, reward, done, info = env.step(action)
			score+=reward
		print(score)
		if score==200:
			print(obs)
		scores.append(score)
scores = np.array(scores)
print('average_score = {0:.1f} +/- {1:.2f}'.format(scores.mean(), scores.std()))

env.close()