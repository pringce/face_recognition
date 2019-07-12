import tensorflow as tf
import math

def arcface_logits_compute(embeddings, label_batch, args, nrof_classes):
    
    '''
    embeddings : normalized embedding layer of Facenet, it's normalized value of output of resface
    label_batch : ground truth label of current training batch
    args:         arguments from cmd line
    nrof_classes: number of classes
    '''
    m = 0.5
    # m = 0.0
    s = 64.0

    cos_m = math.cos(m)
    sin_m = math.sin(m)
    mm = sin_m * m  # issue 1
    threshold = math.cos(math.pi - m)
    with tf.name_scope('arcface_loss'):
        # weights norm
        embedding_weights = tf.get_variable(name='embedding_weights', shape=(args.embedding_size, nrof_classes),
                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False), dtype=tf.float32)
        weights_norm = tf.nn.l2_normalize(embedding_weights, 0, 1e-10, name='weights_norm')
        # cos(theta+m)
        cos_t = tf.matmul(embeddings, weights_norm, name='cos_t')
        cos_t2 = tf.square(cos_t, name='cos_2')
        sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
        sin_t = tf.sqrt(sin_t2, name='sin_t')
        cos_mt = s * tf.subtract(tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')
        
        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_t - threshold
        cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)

        keep_val = s*(cos_t - mm)
        cos_mt_temp = tf.where(cond, cos_mt, keep_val)

        mask = tf.one_hot(label_batch, depth=nrof_classes, name='one_hot_mask')
        # mask = tf.squeeze(mask, 1)
        inv_mask = tf.subtract(1., mask, name='inverse_mask')

        s_cos_t = tf.multiply(s, cos_t, name='scalar_cos_t')

        output = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='arcface_loss_output')
        return output
