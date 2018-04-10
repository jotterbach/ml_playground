import numpy as np
import tensorflow as tf
from collections import deque, namedtuple
from typing import Tuple
import random

Transition = namedtuple('Transition',
                        ('actions', 'rewards', 'gradients', 'data', 'targets'))

logs_path = '/tmp/tensorflow_logs/example/'


class ReplayMemory(object):
    """
    Simple convenience class to store relevant training traces and efficiently sample from them.

    :param int capacity: Size of the replay memory
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = deque([], maxlen=self.capacity)
        self.position = 0

    def push(self, transition: Transition):
        """
        Adds a new observation to the memory

        :param transition: Transition object
        :return: none
        """
        self.memory.append(transition)

    def sample(self, batchsize: int):
        """
        Samples 'batchsize'd number of samples from the memory
        :param batchsize:
        :return: sample of Transition objects
        """
        return random.sample(self.memory, batchsize)

    def __len__(self):
        return len(self.memory)


class Agent(object):
    """
    Agent that learns the update rule for a logistic regression.

    :param sess: Tensorflow session.
    """
    def __init__(self, sess: tf.Session):
        self.memory = ReplayMemory(5000)
        self.batch_size = 30
        self.sess = sess
        self.mode = tf.estimator.ModeKeys.TRAIN

        self._init_graph()
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        self.summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    def _init_graph(self):

        self.data = tf.placeholder(name='data',
                                   dtype=tf.float32,
                                   shape=[100, None])
        self.targets = tf.placeholder(name='target',
                                      dtype=tf.float32,
                                      shape=[100, None])

        self.actions = tf.placeholder(name='action',
                                      shape=[None, 3 * 25],
                                      dtype=tf.float32)
        self.reward = tf.placeholder(name='reward',
                                     shape=[None, 25],
                                     dtype=tf.float32)
        self.grads = tf.placeholder(name='gradients',
                                    shape=[None, 3 * 25],
                                    dtype=tf.float32)
        self.last_action = tf.placeholder(name='last_action',
                                          shape=[None, 3],
                                          dtype=tf.float32)

        # We encode the bias by adding a third column to the data filled with 1's
        self.weights = tf.Variable(initial_value=[[1e-5, 1e-4, 0]],
                                   name="logit_weights",
                                   dtype=tf.float32,
                                   expected_shape=[1, 3],
                                   trainable=True)

        with tf.name_scope("policy"):
            # Subgraph to define the policy network. We construct the input from
            # atomic observation objects
            self.input_layer = tf.concat([self.actions, self.reward, self.grads],
                                         axis=1,
                                         name="state_concatenation")

            self.dense = tf.layers.dense(inputs=self.input_layer,
                                         units=50,
                                         activation=tf.nn.softmax,
                                         name='dense_1')

            self.dropout = tf.layers.dropout(inputs=self.dense,
                                             rate=0.4,
                                             training=(self.mode == tf.estimator.ModeKeys.TRAIN),
                                             name='dropout')

            self.policy = tf.layers.dense(inputs=self.dropout,
                                          units=3,
                                          name='output_layer')

        with tf.name_scope('update_weights'):
            # We update the weights variables using the policy output and the weights from the
            # previous transaction
            self.weights = self.last_action - self.policy

        with tf.name_scope("meta_loss"):
            # The meta-loss is constructed by varying the input data of a logit and then generally
            # trying to find the right weights:
            self.logits = tf.log(tf.nn.sigmoid(tf.matmul(self.data, self.weights, transpose_b=True)))
            self.loss = -1. * tf.reduce_mean(tf.matmul(self.targets, self.logits, transpose_a=True))

    def _train_minibatch(self):
        """
        Samples from the ReplayMemory and trains the policy on the sampled observations

        :return: None
        """
        batch: Tuple[Transition] = self.memory.sample(self.batch_size)
        for obs in batch:
            action = obs.actions
            reward = obs.rewards
            grad = obs.gradients
            data = obs.data
            targets = obs.targets
            self.sess.run(self.optimizer.minimize(self.loss),
                          feed_dict={
                              self.actions: np.array(action).flatten().reshape(-1, 75),
                              self.reward: np.array(reward).flatten().reshape(-1, 25),
                              self.grads: np.array(grad).flatten().reshape(-1, 75),
                              self.last_action: np.array(action[-1]).flatten().reshape(-1, 3),
                              self.data: data,
                              self.targets: np.array(targets).reshape(100, -1)
                          })

    def _run_single_round(self, x0: list):
        """
        Runs a single optimization round on a fixed dataset to create new memories to train on.

        :param x0: Initial value for the weights.
        :return: None
        """
        # initialize round with new data
        mean0 = [.1, .1]
        cov0 = [[1, .01], [.01, 1]]
        mean1 = [-.1, -.1]
        cov1 = [[1, .02], [.02, 1]]
        data, targets = create_data_for_metaloss(mean0, mean1, cov0, cov1)
        # augment the data with a constant np.ones field to incorporate bias term
        data = np.concatenate([data, np.ones(data.shape[0]).reshape(data.shape[0], 1)], axis=1)

        # Initialize finite state space with a maximum FIFO queue
        action = deque([], maxlen=25)
        reward = deque([], maxlen=25)
        grad = deque([], maxlen=25)
        for _ in range(25):
            action.append([0, 0, 0])  # 2 weights + 1 bias
            reward.append(0)
            grad.append([0, 0, 0])  # updates to the actions, a.k.a. logit weights

        action.append(x0)
        rew = 0
        reward.append(rew)
        grad.append(len(x0) * [0.0])

        # Run a single event by doing 100 iterations of the update rule.
        for idx in range(101):
            rew, grad_update, weight = self.sess.run(
                [self.loss, self.policy, self.weights],
                feed_dict={
                    self.actions: np.array(action).flatten().reshape(-1, 75),
                    self.reward: np.array(reward).flatten().reshape(-1, 25),
                    self.grads: np.array(grad).flatten().reshape(-1, 75),
                    self.last_action: np.array(action[-1]).flatten().reshape(-1, 3),
                    self.data: data,
                    self.targets: np.array(targets).reshape(100, -1)
                })
            if idx % 20 == 0:
                print(rew, weight)

            # adjust tensorflow output and push it to the ReplayMemory as observation
            action.append(weight.squeeze().flatten().tolist())
            reward.append(rew.flatten().tolist()[0])
            grad.append(grad_update.squeeze().flatten().tolist())

            obs = Transition(actions=action,
                             gradients=grad,
                             rewards=reward,
                             data=data,
                             targets=targets)
            self.memory.push(obs)

    def learn(self):

        for _ in range(500):
            self._run_single_round(list(np.random.normal(0, 1, size=3)))
            if len(self.memory) >= self.batch_size:
                self._train_minibatch()


def create_data_for_metaloss(mean0, mean1, cov0, cov1):
    data0 = np.random.multivariate_normal(mean0, cov0, size=50)
    data1 = np.random.multivariate_normal(mean1, cov1, size=50)
    data = np.vstack([data0, data1])

    target0 = np.zeros(shape=50)
    target1 = np.ones(shape=50)
    targets = np.hstack([target0, target1])

    return data, targets


if __name__ == "__main__":
    sess = tf.Session()
    agent = Agent(sess)
    agent.learn()

