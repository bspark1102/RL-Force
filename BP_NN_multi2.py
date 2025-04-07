import os
import tensorflow as tf
import numpy as np

class NN_V():
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
        self.state_batch = []
        self.action_batch = []
        self.reward_memory = []
        self.loss = 0
#         self.sess = tf.Session()
        self.sess = tf.compat.v1.Session()
        self.build_net()
#         self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.saver = tf.compat.v1.train.Saver()
#         self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(chkpt_dir,'policy_network.ckpt')
            
        self.path = './{0}'.format(chkpt_dir)
        self.ID = ID
        
    def build_net(self):
        with tf.compat.v1.variable_scope('parametersv', reuse=tf.compat.v1.AUTO_REUSE):
            self.input = tf.compat.v1.placeholder(tf.float32,
                                        shape=[None, self.input_dims], name='input')
            self.label = tf.compat.v1.placeholder(tf.int32,
                                        shape=[None, self.n_actions], name='label')
                                        
        with tf.compat.v1.variable_scope('layer1v', reuse=tf.compat.v1.AUTO_REUSE):
            l1 = tf.layers.dense(inputs=self.input, units=self.layer1_size,
                                 activation=tf.nn.relu,
                      kernel_initializer=tf.contrib.layers.xavier_initializer())

        with tf.compat.v1.variable_scope('layer2v', reuse=tf.compat.v1.AUTO_REUSE):
            l2 = tf.layers.dense(inputs=l1, units=self.layer2_size,
                                 activation=tf.nn.relu,
                      kernel_initializer=tf.contrib.layers.xavier_initializer())
            
        with tf.compat.v1.variable_scope('layer3v', reuse=tf.compat.v1.AUTO_REUSE):
            l3 = tf.layers.dense(inputs=l2, units=self.layer3_size,
                                 activation=tf.nn.relu,
                      kernel_initializer=tf.contrib.layers.xavier_initializer())

        with tf.compat.v1.variable_scope('layer4v', reuse=tf.compat.v1.AUTO_REUSE):
            l4 = tf.layers.dense(inputs=l3, units=self.n_actions,
                                 activation=None,
                      kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.actions = tf.nn.softmax(l4, name='actions')

        with tf.compat.v1.variable_scope('lossv', reuse=tf.compat.v1.AUTO_REUSE):
            self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=l4, labels=self.label)
            self.loss = tf.reduce_mean(self.loss)

        with tf.compat.v1.variable_scope('trainv', reuse=tf.compat.v1.AUTO_REUSE):
#             self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
            self.train_op = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(self.loss)
    
    def compute_loss(self,x,y):    
        my_pred = self.choose_action(x)
        test_y = np.argmax(y, axis = 1)

        accr = np.mean(np.equal(my_pred, test_y))
        return accr

        
    def choose_action(self, observation):
        observation = np.reshape(observation, [-1,self.input_dims])
        probabilities = self.sess.run(self.actions, feed_dict={self.input: observation})
        action = np.argmax(probabilities, axis = 1)
        return action
    
    def get_prob(self, observation):
        observation = np.reshape(observation, [-1,self.input_dims])
        probabilities = self.sess.run(self.actions, feed_dict={self.input: observation})
        return probabilities

    def store_transition(self, observation, action):
        self.state_batch.append(observation)
        self.action_batch.append(action)
    
    def learn(self,state_batch_, action_batch_):
        state_batch_ = np.array(state_batch_)
        action_batch_ = np.array(action_batch_)

        _ = self.sess.run(self.train_op, feed_dict={self.input: state_batch_,
                                       self.label: action_batch_})

    def load_checkpoint(self, epoch):
        print("...Loading checkpoint...")
        self.saver.restore(self.sess, '{0}/{1}/{2}_{3}'.format(self.path, 'checkpoint', self.ID, epoch))

    def save_checkpoint(self, epoch):
        
#         saver = tf.train.Saver()
        self.saver.save(self.sess, '{0}/{1}/{2}_{3}'.format(self.path, 'checkpoint', self.ID, epoch))
        print('Model saved in file: {}'.format(epoch))
        
# #         print("...Saving checkpoint...")
#         self.saver.save(self.sess, self.checkpoint_file)


class NN_I():
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
        self.state_batch = []
        self.action_batch = []
        self.reward_memory = []
        self.loss = 0
#         self.sess = tf.Session()
        self.sess = tf.compat.v1.Session()
        self.build_net()
#         self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.saver = tf.compat.v1.train.Saver()
#         self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(chkpt_dir,'policy_network.ckpt')
            
        self.path = './{0}'.format(chkpt_dir)
        self.ID = ID
        
    def build_net(self):
        with tf.compat.v1.variable_scope('parametersi', reuse=tf.compat.v1.AUTO_REUSE):
            self.input = tf.compat.v1.placeholder(tf.float32,
                                        shape=[None, self.input_dims], name='input')
            self.label = tf.compat.v1.placeholder(tf.int32,
                                        shape=[None, self.n_actions], name='label')
                                        
        with tf.compat.v1.variable_scope('layer1i', reuse=tf.compat.v1.AUTO_REUSE):
            l1 = tf.layers.dense(inputs=self.input, units=self.layer1_size,
                                 activation=tf.nn.relu,
                      kernel_initializer=tf.contrib.layers.xavier_initializer())

        with tf.compat.v1.variable_scope('layer2i', reuse=tf.compat.v1.AUTO_REUSE):
            l2 = tf.layers.dense(inputs=l1, units=self.layer2_size,
                                 activation=tf.nn.relu,
                      kernel_initializer=tf.contrib.layers.xavier_initializer())
            
        with tf.compat.v1.variable_scope('layer3i', reuse=tf.compat.v1.AUTO_REUSE):
            l3 = tf.layers.dense(inputs=l2, units=self.layer3_size,
                                 activation=tf.nn.relu,
                      kernel_initializer=tf.contrib.layers.xavier_initializer())

        with tf.compat.v1.variable_scope('layer4i', reuse=tf.compat.v1.AUTO_REUSE):
            l4 = tf.layers.dense(inputs=l3, units=self.n_actions,
                                 activation=None,
                      kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.actions = tf.nn.softmax(l4, name='actions')

        with tf.compat.v1.variable_scope('lossi', reuse=tf.compat.v1.AUTO_REUSE):
            self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=l4, labels=self.label)
            self.loss = tf.reduce_mean(self.loss)

        with tf.compat.v1.variable_scope('traini', reuse=tf.compat.v1.AUTO_REUSE):
#             self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
            self.train_op = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(self.loss)

    def compute_loss(self,x,y):    
        my_pred = self.choose_action(x)
        test_y = np.argmax(y, axis = 1)

        accr = np.mean(np.equal(my_pred, test_y))
        return accr

        
    def choose_action(self, observation):
        observation = np.reshape(observation, [-1,self.input_dims])
        probabilities = self.sess.run(self.actions, feed_dict={self.input: observation})
        action = np.argmax(probabilities, axis = 1)
        return action
    
    def get_prob(self, observation):
        observation = np.reshape(observation, [-1,self.input_dims])
        probabilities = self.sess.run(self.actions, feed_dict={self.input: observation})
        return probabilities

    def store_transition(self, observation, action):
        self.state_batch.append(observation)
        self.action_batch.append(action)
    
    def learn(self,state_batch_, action_batch_):
        state_batch_ = np.array(state_batch_)
        action_batch_ = np.array(action_batch_)

        _ = self.sess.run(self.train_op, feed_dict={self.input: state_batch_,
                                       self.label: action_batch_})

    def load_checkpoint(self, epoch):
        print("...Loading checkpoint...")
        self.saver.restore(self.sess, '{0}/{1}/{2}_{3}'.format(self.path, 'checkpoint', self.ID, epoch))

    def save_checkpoint(self, epoch):
        
#         saver = tf.train.Saver()
        self.saver.save(self.sess, '{0}/{1}/{2}_{3}'.format(self.path, 'checkpoint', self.ID, epoch))
        print('Model saved in file: {}'.format(epoch))
        
# #         print("...Saving checkpoint...")
#         self.saver.save(self.sess, self.checkpoint_file)