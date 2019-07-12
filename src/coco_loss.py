import tensorflow as tf
import math

def coco_logits_compute(embeddings, args, nrof_classes, alpha=8.0):
    
    with tf.name_scope('coco_loss'):
        embeddings_dim = args.embedding_size
        embeddings_norm = tf.nn.l2_normalize(embeddings, 1, 1e-10, name='embeddings')
        coco_centers = tf.get_variable(name='coco_centers', shape=(nrof_classes, embeddings_dim), initializer=tf.contrib.layers.xavier_initializer(uniform=False), dtype=tf.float32, trainable=True)
        snembedding = alpha*embeddings_norm
        norm_centers = tf.nn.l2_normalize(coco_centers, 1, 1e-10, name='centers_norm')
        logits = tf.matmul(snembedding, tf.transpose(norm_centers,[1,0]))
        return logits, embeddings_norm

