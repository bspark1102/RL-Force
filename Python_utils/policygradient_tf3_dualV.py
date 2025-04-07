import os
import tensorflow as tf
import numpy as np

class PolicyGradientAgentVent():
    def __init__(self, ALPHA, GAMMA=0.95, n_actions=8,
                 layer1_size=100, layer2_size=100, layer3_size=100, input_dims=20,
                 chkpt_dir='tmp/checkpoints',ID='model'):
        self.lr = ALPHA
        self.gamma = GAMMA
        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.layer3_size = layer3_size
        self.input_dims = input_dims
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.sess = tf.Session()
        # self.sess = tf.compat.v1.Session()
        self.build_net()
        self.sess.run(tf.global_variables_initializer())
        # self.sess.run(tf.compat.v1.global_variables_initializer())
        # self.saver = tf.compat.v1.train.Saver()
        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(chkpt_dir,'policy_network.ckpt')
            
        
        self.path = './{0}'.format(chkpt_dir)
        self.ID = ID
        

        
    def build_net(self):
        with tf.variable_scope('parametersv', reuse=tf.AUTO_REUSE):
            self.input = tf.placeholder(tf.float32,
                                        shape=[None, self.input_dims], name='input')
            self.label = tf.placeholder(tf.int32,
                                        shape=[None, ], name='label')
            self.G = tf.placeholder(tf.float32, shape=[None,], name='G')

        with tf.variable_scope('layer1v', reuse=tf.AUTO_REUSE):
            l1 = tf.layers.dense(inputs=self.input, units=self.layer1_size,
                                 activation=tf.nn.relu,
                      kernel_initializer=tf.contrib.layers.xavier_initializer())

        with tf.variable_scope('layer2v', reuse=tf.AUTO_REUSE):
            l2 = tf.layers.dense(inputs=l1, units=self.layer2_size,
                                 activation=tf.nn.relu,
                      kernel_initializer=tf.contrib.layers.xavier_initializer())
            
        with tf.variable_scope('layer3v', reuse=tf.AUTO_REUSE):
            l3 = tf.layers.dense(inputs=l2, units=self.layer3_size,
                                 activation=tf.nn.relu,
                      kernel_initializer=tf.contrib.layers.xavier_initializer())

        with tf.variable_scope('layer4v', reuse=tf.AUTO_REUSE):
            l4 = tf.layers.dense(inputs=l3, units=self.n_actions,
                                 activation=None,
                      kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.actions = tf.nn.softmax(l4, name='actions')

        with tf.variable_scope('lossv', reuse=tf.AUTO_REUSE):
            negative_log_probability = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                    logits=l4, labels=self.label)

            loss = negative_log_probability * self.G

        with tf.variable_scope('trainv', reuse=tf.AUTO_REUSE):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, observation):
#         observation = observation[np.newaxis, :]
        probabilities = self.sess.run(self.actions, feed_dict={self.input: observation})[0]
        action = np.random.choice(self.action_space, p = probabilities )

        return action, probabilities

    def store_transition(self, observation, action, reward):
        self.state_memory.append(observation)
        self.action_memory.append(action)
        self.reward_memory.append(reward)

    def learn(self):
        state_memory = np.array(self.state_memory)
        state_memory = np.reshape(state_memory,[-1,state_memory.shape[2]])
        action_memory = np.array(self.action_memory)
        reward_memory = np.array(self.reward_memory)

        G = np.zeros_like(reward_memory)
        for t in range(len(reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(reward_memory)):
                G_sum += reward_memory[k] * discount
                discount *= self.gamma
            G[t] = G_sum
        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        G = (G - mean) / std

        _ = self.sess.run(self.train_op,
                            feed_dict={self.input: state_memory,
                                       self.label: action_memory,
                                       self.G: G})
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

    def load_checkpoint(self, epoch):
        print("...Loading checkpoint...")
        self.saver.restore(self.sess, '{0}/{1}/{2}_{3}'.format(self.path, 'checkpoint', self.ID, epoch))

    def save_checkpoint(self, epoch):
        
#         saver = tf.train.Saver()
        self.saver.save(self.sess, '{0}/{1}/{2}_{3}'.format(self.path, 'checkpoint', self.ID, epoch))
        print('Model saved in file: {}'.format(epoch))
