import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

NUM_EPISODES = 20
EPISODE_LENGTH = 200
LEARNING_RATE = 1e-3

# creates window for graphics
env = gym.make('CartPole-v0')

def step(e, action):
  # random action necessary???
  #import pdb; pdb.set_trace()
  if np.random.rand(1) < e:
    action = env.action_space.sample()
  
  new_obs, reward, done, info = env.step(action)
  new_obs = [o.astype(np.float32) for o in new_obs]
  return [new_obs, np.array([reward], dtype=np.float32), done]

def forwardPass(obs):
  W1 = tf.Variable(tf.random_normal(shape=[4,10]))
  B1 = tf.Variable(tf.random_normal(shape=[10]))
  W2 = tf.Variable(tf.random_normal(shape=[10,2]))
  B2 = tf.Variable(tf.random_normal(shape=[2]))
  
  H1 = tf.add(tf.matmul(tf.expand_dims(obs,0),W1), B1)
  H1 = tf.nn.relu(H1)
  
  Q = tf.add(tf.matmul(H1,W2), B2)
  
  return Q
  
def main():
  # Set learning parameters
  y = .99
  ee = 0.1
  tList = []
  rList = []
  
  graph = tf.Graph()
  
  with graph.as_default():
    observation = tf.placeholder(dtype=tf.float32, shape=[4])
    e = tf.placeholder(dtype=tf.float32, shape=[])
    old_Q = forwardPass(observation)
    predicted_action = tf.argmax(old_Q,1)
    
    [new_obs, reward, done] = tf.py_func(step, [e, predicted_action[0]], [tf.float32, tf.float32, tf.bool], name='step')
    
    new_Q = forwardPass(new_obs)
    loss = tf.square(reward + tf.multiply(y,tf.reduce_max(new_Q)) - tf.gather(old_Q, tf.argmax(new_Q)))
    #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
    
  with tf.Session(graph=graph) as sess:

    
    # Initialize the model parameters
    tf.global_variables_initializer().run()
    
    for episode in range(NUM_EPISODES):
      obs = env.reset()
      rAll = 0
      don = False
      
      for t in range(EPISODE_LENGTH):
        #if not episode % 50:
          #env.render()
        new_ob, loss_val, reward_val, don, _ = sess.run([new_obs, loss, reward, done, optimizer], feed_dict={observation:obs, e:ee})
        
        obs = new_ob
        rAll += reward_val
        
        # reduce chance of choosing random action as training progresses
        if don == True:
          ee = 1./((episode/200) + 10)
          tList.append(t)
          rList.append(rAll)
          break
    
    print("Percent of succesful episodes: " + str(sum(rList)/NUM_EPISODES) + "%")
  
  #import pdb; pdb.set_trace()
  plt.figure(1)
  plt.subplot(211)
  plt.plot(rList)
  plt.title('rList')
  plt.xlabel('episode number')
  plt.ylabel('reward?')
  
  plt.subplot(212)
  plt.plot(tList)
  plt.title('tList')
  plt.xlabel('episode number')
  plt.ylabel('episode length')
  
  plt.show()

if __name__ == '__main__':
  main()