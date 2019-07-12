import tensorflow as tf
import tensorflow.contrib.slim as slim
from tflearn.layers.conv import global_avg_pool
import numpy as np

'''
Resface20 and Resface36 proposed in sphereface and applied in Additive Margin Softmax paper
Notice:
batch norm is used in line 111. to cancel batch norm, simply commend out line 111 and use line 112

'''
reduction_ratio = 16

def prelu(x):
  alphas = tf.get_variable(name='prelu_alphas', dtype=tf.float32, initializer=tf.constant_initializer(value=0.25),shape=[x.get_shape()[-1]])
  pos = tf.nn.relu(x)
  neg = alphas * (x - abs(x)) * 0.5
  return pos + neg

def Global_Average_Pooling(net):
    return global_avg_pool(net, name='Global_avg_pooling')

def resface_block(lower_input,output_channels,stride,dim_match=True,scope=None):
    with tf.variable_scope(scope):
        net = slim.batch_norm(lower_input, activation_fn=None,scope='bn1')
        net = slim.conv2d(net, output_channels)
        net = slim.batch_norm(net,scope='bn2')
        net = slim.conv2d(net, output_channels,stride=stride)
        net = slim.batch_norm(net, activation_fn=None,scope='bn3')

        if dim_match==True:
            short_cut = lower_input
        else:
            short_cut = slim.conv2d(lower_input, output_channels, stride=2, kernel_size=1)
            short_cut = slim.batch_norm(short_cut, activation_fn=None,scope='shortcut_bn')

        channel = int(np.shape(net)[-1])
        net = Squeeze_excitation_layer(net, out_dim=channel, ratio=reduction_ratio, layer_name='SE_A')
        return short_cut + net

def Squeeze_excitation_layer(net, out_dim, ratio, layer_name):
    with tf.variable_scope(layer_name) :
        squeeze = Global_Average_Pooling(net)
        excitation = slim.fully_connected(squeeze, out_dim // ratio, scope=layer_name+'_fully_connected1')
        excitation = tf.nn.relu(excitation)
        excitation = slim.fully_connected(excitation, out_dim, scope=layer_name+'_fully_connected2')
        excitation = tf.nn.sigmoid(excitation)
        excitation = tf.reshape(excitation, [-1,1,1,out_dim])
        scale = net * excitation
        return scale


def LResnet50E_IR(images, keep_probability, 
             phase_train=True, bottleneck_layer_size=512, 
             weight_decay=0.0, reuse=None):
    '''
    conv name
    conv[conv_layer]_[block_index]_[block_layer_index]
    
    for resnet50 n_units=[3,4,14,3], consider one unit is dim_reduction_layer
    repeat n_units=[2,3,13,2]
    '''
    with tf.variable_scope('Conv1'):
        net = slim.conv2d(images,64,scope='Conv1_pre')
        net = slim.batch_norm(net,scope='Conv1_bn')
    with tf.variable_scope('Conv2'):
        net = resface_block(net,64,stride=2,dim_match=False,scope='Conv2_pre')
        net = slim.repeat(net,2,resface_block,64,1,True,scope='Conv2_main')
    with tf.variable_scope('Conv3'):
        net = resface_block(net,128,stride=2,dim_match=False,scope='Conv3_pre')
        net = slim.repeat(net,3,resface_block,128,1,True,scope='Conv3_main')
    with tf.variable_scope('Conv4'):
        net = resface_block(net,256,stride=2,dim_match=False,scope='Conv4_pre')
        net = slim.repeat(net,5,resface_block,256,1,True,scope='Conv4_main')
    with tf.variable_scope('Conv5'):
        net = resface_block(net,512,stride=2,dim_match=False,scope='Conv5_pre')
        net = slim.repeat(net,2,resface_block,512,1,True,scope='Conv5_main')

    with tf.variable_scope('Logits'):
        net = slim.batch_norm(net,activation_fn=None,scope='bn1')
        net = slim.dropout(net, keep_probability, is_training=phase_train,scope='Dropout')        
        net = slim.flatten(net)
    
    net = slim.fully_connected(net, bottleneck_layer_size, biases_initializer=tf.contrib.layers.xavier_initializer(), scope='fc1')
    net = slim.batch_norm(net, activation_fn=None, scope='Bottleneck')

    return net,''

def inference(image_batch, keep_probability, 
              phase_train=True, bottleneck_layer_size=512, 
              weight_decay=0.0):
    with tf.variable_scope('LResnetE_IR'):
        with slim.arg_scope([slim.conv2d, slim.fully_connected], 
                             weights_initializer=tf.contrib.layers.xavier_initializer(), 
                             weights_regularizer=slim.l2_regularizer(weight_decay), 
                             biases_initializer=None, #default no biases
                             activation_fn=None,
                             normalizer_fn=None
                             ):
            with slim.arg_scope([slim.conv2d], kernel_size=3):
                with slim.arg_scope([slim.batch_norm],
                                    decay=0.995,
                                    epsilon=1e-3,
                                    scale=True,
                                    is_training=phase_train,
                                    activation_fn=prelu,
                                    updates_collections=None,
                                    variables_collections=[ tf.GraphKeys.TRAINABLE_VARIABLES ]
                                   ):
                    return LResnet50E_IR(images=image_batch, 
                                    keep_probability=keep_probability, 
                                    phase_train=phase_train, 
                                    bottleneck_layer_size=bottleneck_layer_size, 
                                    reuse=None)
