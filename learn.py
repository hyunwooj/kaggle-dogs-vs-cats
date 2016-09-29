import numpy as np
import tensorflow as tf
from load_data import load_data

class ConvNet:
    def __init__(self):
        self.train, self.test = load_data()
        self.image_size = 28
        self.num_labels = 2
        self.num_steps = 250
        self.batch_size = 50

    def weight_variable(self, shape, name=None):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial, name=name)

    def bias_variable(self, shape, name=None):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial, name=name)

    def conv2d(self, x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')

    def learn(self):
        # placeholders for inputs and labels
        x = tf.placeholder(tf.float32,
                           shape=[None, self.image_size, self.image_size, 3])
        y = tf.placeholder(tf.float32, shape=[None, 2])

        # first conv layer
        W_conv1 = self.weight_variable([5, 5, 3, 32], name='W_conv1')
        b_conv1 = self.bias_variable([32], name='b_conv1')

        h_conv1 = tf.nn.relu(self.conv2d(x, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        # second conv layer
        W_conv2 = self.weight_variable([5, 5, 32, 64], name='W_conv2')
        b_conv2 = self.bias_variable([64], name='b_conv2')

        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)

        # fully connected layer
        W_fc1 = self.weight_variable([7 * 7 * 64, 1024], name='W_fc1')
        b_fc1 = self.bias_variable([1024], name='b_fc1')

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # dropout
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # readout layer
        W_fc2 = self.weight_variable([1024, 2], name='W_fc2')
        b_fc2 = self.bias_variable([2], name='b_fc2')

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        # loss function
        y_prob_dog, y_prob_cat = y_conv[:, 0], y_conv[:, 1]
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y))

        # optimizer
        optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

        saver = tf.train.Saver({
            'W_conv1': W_conv1,
            'b_conv1': b_conv1,
            'W_conv2': W_conv2,
            'b_conv2': b_conv2,
            'W_fc1': W_fc1,
            'b_fc1': b_fc1,
            'W_fc2': W_fc2,
            'b_fc2': b_fc2
        })

        with tf.Session() as session:
            session.run(tf.initialize_all_variables())
            print('Initialized')

            for step in range(self.num_steps):
                offset = step * self.batch_size

                # dogs
                inputs_dog = self.train[0][offset:offset+self.batch_size, :, :, :]
                labels_dog = np.array([[1, 0] for _ in range(self.batch_size)])

                # cats
                inputs_cat = self.train[1][offset:offset+self.batch_size, :, :, :]
                labels_cat = np.array([[0, 1] for _ in range(self.batch_size)])

                batch_x = np.vstack([inputs_dog, inputs_cat])
                batch_y = np.vstack([labels_dog, labels_cat])

                feed_dict = {x: batch_x/255, y: batch_y, keep_prob: 0.5}

                _, l = session.run([optimizer, loss], feed_dict=feed_dict)
                if (step % self.batch_size == 0):
                    print('Step: %d' % (step))
                    print('Loss: %f' % (l))

            # loss for all
            # dogs
            inputs_dog = self.train[0]
            labels_dog = np.array([[1, 0] for _ in range(len(inputs_dog))])

            # cats
            inputs_cat = self.train[1]
            labels_cat = np.array([[0, 1] for _ in range(len(inputs_cat))])

            batch_x = np.vstack([inputs_dog, inputs_cat])
            batch_y = np.vstack([labels_dog, labels_cat])

            feed_dict = {x: batch_x/255, y: batch_y, keep_prob: 0.5}

            l = session.run(loss, feed_dict=feed_dict)
            print('Loss for all: %f' % (l))

            path = saver.save(session, '%s.ckpt' % type(self).__name__)
            print('saved in file: %s' % path)
            print('Done')

    def predict(self):
        print('Pridicting...')

        # placeholders for inputs
        x = tf.constant(self.test)

        # first conv layer
        W_conv1 = self.weight_variable([5, 5, 3, 32], name='W_conv1')
        b_conv1 = self.bias_variable([32], name='b_conv1')

        h_conv1 = tf.nn.relu(self.conv2d(x, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        # second conv layer
        W_conv2 = self.weight_variable([5, 5, 32, 64], name='W_conv2')
        b_conv2 = self.bias_variable([64], name='b_conv2')

        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)

        # fully connected layer
        W_fc1 = self.weight_variable([7 * 7 * 64, 1024], name='W_fc1')
        b_fc1 = self.bias_variable([1024], name='b_fc1')

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # dropout
        keep_prob = tf.constant(1.0)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # readout layer
        W_fc2 = self.weight_variable([1024, 2], name='W_fc2')
        b_fc2 = self.bias_variable([2], name='b_fc2')

        y_prob = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        saver = tf.train.Saver({
            'W_conv1': W_conv1,
            'b_conv1': b_conv1,
            'W_conv2': W_conv2,
            'b_conv2': b_conv2,
            'W_fc1': W_fc1,
            'b_fc1': b_fc1,
            'W_fc2': W_fc2,
            'b_fc2': b_fc2
        })

        with tf.Session() as session:
            try:
                saver.restore(session, '%s.ckpt' % type(self).__name__)
                print('Restored')
            except:
                print('Learn')
                self.learn()
                saver.restore(session, '%s.ckpt' % type(self).__name__)

            prob = session.run(y_prob)[:, 0]
            indexed = np.hstack([np.arange(1, len(prob)+1)[:, None],
                                prob[:, None]])
            np.savetxt('predictions.csv', indexed, fmt=['%d', '%.3f'],
                       header='Id,Label', delimiter=',', comments='')
            print('Done')
            return prob

convnet = ConvNet()
y = convnet.predict()
