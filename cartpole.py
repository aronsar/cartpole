import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

NUM_EPISODES = 200
EPISODE_LENGTH = 200
LEARNING_RATE = 1e-3

def main():
  # Set learning parameters
  y = .99
  e = 0.1
  tList = []
  rList = []
  
  graph = tf.Graph()
  
  with graph.as_default():
    #These lines establish the feed-forward part of the network used to choose actions 
    observation = tf.placeholder(dtype=tf.float32, shape=[4])
    W = tf.Variable(tf.random_uniform([4,2],0,0.01))
    Q_out = tf.matmul(tf.expand_dims(observation,0),W)
    predicted_action = tf.argmax(Q_out,1)
    
    # do the env.step(action) in a py_func so that not so many sess.run calls are made
    
    #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
    next_Q = tf.placeholder(shape=[1,2],dtype=tf.float32)
    loss = tf.reduce_sum(tf.square(next_Q - Q_out))
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
    
  with tf.Session(graph=graph) as sess:
    # creates window for graphics
    env = gym.make('CartPole-v0')
    
    # Initialize the model parameters
    tf.global_variables_initializer().run()
    
    for episode in range(NUM_EPISODES):
      old_obs = env.reset()
      rAll = 0
      done = False
      
      for t in range(EPISODE_LENGTH):
        # clean this up if possible, it's messy
        env.render()
        action, old_Q = sess.run([predicted_action, Q_out], feed_dict={observation:old_obs})
        
        # random action necessary???
        if np.random.rand(1) < e:
          action[0] = env.action_space.sample()

        new_obs, reward, done, info = env.step(action[0])

        new_Q = sess.run([Q_out], feed_dict={observation:new_obs})
        target_Q = old_Q
        target_Q[0, action[0]] = reward + y*np.max(new_Q)
        
        _ = sess.run([optimizer], feed_dict={observation:old_obs, next_Q:target_Q})
        
        old_obs = new_obs
        rAll += reward
        
        # reduce chance of choosing random action as training progresses
        if done == True:
          e = 1./((episode/50) + 10)
          tList.append(t)
          rList.append(rAll)
          break
    
    print("Percent of succesful episodes: " + str(sum(rList)/NUM_EPISODES) + "%")
  
  import pdb; pdb.set_trace()
  plt.figure(figsize=(8,8))
  plt.plot(rList)
  plt.show()
  
  plt.figure(figsize=(8,8))
  plt.plot(tList)
  plt.show()

if __name__ == '__main__':
  main()