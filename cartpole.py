import gym
import tensorflow as tf

NUM_EPISODES = 1000
EPISODE_LENGTH = 100



env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
            
            
def main():
  graph = tf.Graph()
  
  with graph.as_default():
    observation = tf.placeholder(dtype=tf.float32, shape=[4,4])
    
    
  with tf.Session(graph=graph) as sess:
    for episode in range(NUM_EPISODES):
      observation_ = env.reset()
      
      for t in range(EPISODE_LENGTH):
        env.render()
        feed_dict = {observation:observation_}
        action_, loss_, _ = sess.run([action, loss, optimizer], feed_dict=feed_dict)
        