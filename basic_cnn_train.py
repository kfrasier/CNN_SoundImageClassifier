from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tensorflow as tf
import numpy as np
import os
import pickle
import scipy.io as sio

# import urb_sound

def deep_cnn(x):
  # if tf.gfile.Exists(log_dir): 
  #   tf.gfile.DeleteRecursively(log_dir)
  # tf.gfile.MakeDirs(log_dir)
  frames = 360
  bands = 1001

  # feature_size = 2460 #60x41

  kernel_size = 5 # filter size aka kernel size
  stride_size = 3
  depth = 20 # number of channels output by convolution layer
  num_hidden = 2000 # number of neurons in hidden layer
  num_labels = 10
  num_channels = 1

  # define model flow
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([kernel_size, kernel_size, num_channels, depth])
    b_conv1 = bias_variable([depth])
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1,stride_size) + b_conv1)

  with tf.name_scope('pool1'):
    h_pool1 = max_pool(h_conv1,kernel_size,stride_size)

  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([kernel_size, kernel_size, depth, depth*2])
    b_conv2 = bias_variable([depth*2])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2,stride_size) + b_conv2)

  with tf.name_scope('pool2'):
    h_pool2 = max_pool(h_conv2,kernel_size,stride_size)

  # Fully connected layer 1 -- after 2 rounds of downsampling
  with tf.name_scope('full_connect'):
    shape = h_pool2.get_shape().as_list()
    h_pool2_flat = tf.reshape(h_pool2, [-1, shape[1] * shape[2] * shape[3]])
    W_fc1 = weight_variable([shape[1] * shape[2] *shape[3], num_hidden])
    b_fc1 = bias_variable([num_hidden])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    _activation_summary(h_fc1,'full_connect')

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  with tf.name_scope('mapping'):
    # Map the 1024 features to 10 classes, one for each digit
    W_fc2 = weight_variable([num_hidden, num_labels])
    b_fc2 = bias_variable([num_labels])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  
  return y_conv, keep_prob

def _activation_summary(x,tensor_name):
  # Helper to create summaries for activations.
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',tf.nn.zero_fraction(x))

def weight_variable(shape):
  # Create a weight variable with of a given shape.
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  # Create a bias (constant) variable with of a given shape.
  initial = tf.constant(0.1,shape = shape)
  return tf.Variable(initial)

def conv2d(x, W, stride_size):
  #  returns a 2d convolution layer with full stride.
  return tf.nn.conv2d(x, W, strides=[1, stride_size, stride_size,   1], padding='SAME')

def max_pool(x,kernel_size,stride_size):
  # max_pool downsamples a feature map by stride_size.
  return tf.nn.max_pool(x, ksize=[1, kernel_size, kernel_size, 1],
                        strides=[1, stride_size, stride_size, 1], padding='SAME')



def main(_):

  os.chdir("E:/Data/CNN_images/half_hr_windows/single_label")
  # tr_features = pickle.load(open("tr_features.pkl","rb"))
  # ts_features = pickle.load(open("ts_features.pkl","rb"))
  # tr_labels = pickle.load(open("tr_labels.pkl","rb"))
  # ts_labels = pickle.load(open("ts_labels.pkl","rb"))

  
  tr_feature_contents = sio.loadmat('tr_features.mat')
  tr_features = np.expand_dims(tr_feature_contents['imageStack'],3)
  tr_label_contents = sio.loadmat('tr_labels.mat')
  tr_labels = tr_label_contents['labelList']

  ts_feature_contents = sio.loadmat('ts_features.mat')
  ts_features = np.expand_dims(ts_feature_contents['imageStack'],3)
  ts_label_contents = sio.loadmat('ts_labels.mat')
  ts_labels = ts_label_contents['labelList']


  learning_rate = 0.001
  total_iterations = 1000
  batch_size=20
  frames = 360
  bands = 1001
  num_labels = 10
  num_channels = 1 # number of input channels
  log_dir = "E:/Data/CNN_images/half_hr_windows/single_label/logs"

  g = tf.Graph()
  with g.as_default():
    sess = tf.InteractiveSession()

    # define nn structure
    with tf.name_scope('input'):
      # Create the model
      x = tf.placeholder(tf.float32, shape=[None,bands,frames,num_channels])
      # Define loss and optimizer
      y_ = tf.placeholder(tf.float32,[None,num_labels])

    y_conv,keep_prob = deep_cnn(x)

    with tf.name_scope('training'):
      cross_entropy = tf.reduce_mean(
     	tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
      train_step = tf.train.AdamOptimizer(
        learning_rate = learning_rate).minimize(cross_entropy)
      correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      tf.summary.scalar('accuracy', accuracy)

    # Merge all the summaries and write them out to ??
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(log_dir + '/test')

    tf.global_variables_initializer().run()

    cost_history = np.empty(shape=[1],dtype=float)

    for itr in range(total_iterations):  
      # select random subsets of training and testing sets
      tr_batch = np.random.randint(low = 0, high = tr_features.shape[0],size = batch_size)
      batch_x = tr_features[tr_batch, :, :, :]
      batch_y = tr_labels[tr_batch, :]
      ts_batch = np.random.randint(low = 0, high = ts_features.shape[0],size = batch_size)
      batch_ts_x = ts_features[ts_batch, :, :, :]
      batch_ts_y = ts_labels[ts_batch, :]

      # offset = (itr * batch_size) % (tr_labels.shape[0] - batch_size)
      # batch_x = tr_features[offset:(offset + batch_size), :, :, :]
      # batch_y = tr_labels[offset:(offset + batch_size), :]
      # offset_ts = (itr * batch_size) % (ts_labels.shape[0] - batch_size)
      # batch_ts_x = ts_features[offset_ts:(offset_ts + batch_size), :, :, :]
      # batch_ts_y = ts_labels[offset_ts:(offset_ts + batch_size), :]
      if itr % 10 == 0:  
        # TEST: Record summaries and test-set accuracy
        summary, acc = sess.run([merged, accuracy],
          feed_dict= {x:batch_ts_x, y_:batch_ts_y,keep_prob:1})
        test_writer.add_summary(summary, itr)
        print('Accuracy at step %s: %s' % (itr, acc))
      else:
        # TRAIN
        if itr % 100 == 99:
          run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
          run_metadata = tf.RunMetadata()

          summary, c = sess.run([merged,train_step],
              feed_dict= {x:batch_x, y_:batch_y, keep_prob:.5},
              options=run_options,
              run_metadata=run_metadata)

          cost_history = np.append(cost_history,c)
          train_writer.add_run_metadata(run_metadata, 'step%03d' % itr)
          train_writer.add_summary(summary, itr)

        else:  # Record a summary
          summary, c = sess.run([merged,train_step],
          	  feed_dict= {x:batch_x, y_:batch_y,keep_prob:.5})
          train_writer.add_summary(summary, itr)

  # y_pred = sess.run(tf.argmax(y_,1),feed_dict={x: batch_x})
  # y_true = sess.run(tf.argmax(ts_labels,1))
  train_writer.close()
  test_writer.close()


if __name__ == '__main__':
  tf.app.run(main=main)
