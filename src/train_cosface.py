from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import sys
import random
import tensorflow as tf
import numpy as np
import importlib
import argparse
import lfw
import facenet
import h5py
import math
import tensorflow.contrib.slim as slim
from AM_softmax import AM_logits_compute
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

def main(args):
  
    network = importlib.import_module(args.model_def)
    image_size = (args.image_size, args.image_size)

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir) #日志保存地址
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

    # Write arguments to a text file
    facenet.write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))
        
    # Store some git revision info in a text file in the log directory
    # os.path.realpath(__file__)代表返回当前模块真实路径
    # os.path.split(os.path.realpath(__file__))代表返回路径的目录和文件名(元组形式)
    src_path,_ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, log_dir, ' '.join(sys.argv))

    np.random.seed(seed=args.seed)
    random.seed(args.seed)
    # dataset是列表，列表元素为每一个类的ImageClass对象，定义见facenet.py文件
    dataset = facenet.get_dataset(args.data_dir)
    if args.filter_filename:
        dataset = filter_dataset(dataset, os.path.expanduser(args.filter_filename), 
            args.filter_percentile, args.filter_min_nrof_images_per_class)
    
    # 划分训练集和测试集，train_set和val_set的形式和dataset一样
    if args.validation_set_split_ratio>0.0:
        train_set, val_set = facenet.split_dataset(dataset, args.validation_set_split_ratio, args.min_nrof_val_images_per_class, 'SPLIT_IMAGES')
    else:
        train_set, val_set = dataset, []
        
    nrof_classes = len(train_set) #类目数(即有多少人)
    
    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)
    pretrained_model = None
    if args.pretrained_model:
        pretrained_model = os.path.expanduser(args.pretrained_model)
        print('Pre-trained model: %s' % pretrained_model)
    
    if args.lfw_dir:
        print('LFW directory: %s' % args.lfw_dir)
        # Read the file containing the pairs used for testing
        # pairs也是列表，子元素也是列表，每个子列表包含pairs.txt的每一行
        pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))
        # Get the paths for the corresponding images
        #lfw_paths为两两比较的列表,列表的元素为有2个元素的元祖,actual_issame为列表,表示每个元素是否为同一个人
        lfw_paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs)
    
    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)

        # global_step对应的是全局批的个数，根据这个参数可以更新学习率
        global_step = tf.Variable(0, trainable=False)
        
        # Get a list of image paths and their labels
        image_list, label_list = facenet.get_image_paths_and_labels(train_set)
        assert len(image_list)>0, 'The training set should not be empty'
        
        val_image_list, val_label_list = facenet.get_image_paths_and_labels(val_set)

        # Create a queue that produces indices into the image_list and label_list 
        labels = ops.convert_to_tensor(label_list, dtype=tf.int32)
        # range_size的大小和样本个数一样
        range_size = array_ops.shape(labels)[0]
        #QueueRunner：保存的是队列中的入列操作，保存在一个list当中，其中每个enqueue运行在一个线程当中
        #range_input_producer：返回的是一个队列，队列中有打乱的整数，范围是从0到range size 
        #range_size大小和总的样本个数一样
        #并将一个QueueRunner添加到当前图的QUEUE_RUNNER集合中
        index_queue = tf.train.range_input_producer(range_size, num_epochs=None,
                             shuffle=True, seed=None, capacity=32) #capacity代表队列容量
        
        # 返回的是一个出列操作，每次出列一个epoch需要用到的样本个数
        index_dequeue_op = index_queue.dequeue_many(args.batch_size*args.epoch_size, 'index_dequeue')
        
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        image_paths_placeholder = tf.placeholder(tf.string, shape=(None,1), name='image_paths')
        labels_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='labels')
        control_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='control')
        
        # 上面的队列是一个样本索引的出队队列，只用来出列
        # 用来每次出列一个epoch中用到的样本
        # 这里是第二个队列，这个队列用来入列，每个元素的大小为shape=[(1,),(1,),(1,)]
        nrof_preprocess_threads = 4
        # 这里的shape代表每一个元素的维度
        input_queue = data_flow_ops.FIFOQueue(capacity=600000,
                                    dtypes=[tf.string, tf.int32, tf.int32],
                                    shapes=[(1,), (1,), (1,)],
                                    shared_name=None, name=None)
        
        # 这时一个入列的操作，这个操作将在session run的时候用到
        # 每次入列的是image_paths_placeholder, labels_placeholder对
        # 注意，这里只有2个队列，一个用来出列打乱的元素序号
        # 一个根据对应的需要读取指定的文件
        enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder, control_placeholder], name='enqueue_op')
        image_batch, label_batch = facenet.create_input_pipeline(input_queue, image_size, nrof_preprocess_threads, batch_size_placeholder)

        # image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        label_batch = tf.identity(label_batch, 'label_batch')
        
        print('Total number of classes: %d' % nrof_classes)
        print('Total number of examples: %d' % len(image_list))
        print('Building training graph')

        # Build the inference graph
        prelogits, _ = network.inference(image_batch, args.keep_probability, 
            phase_train=phase_train_placeholder, bottleneck_layer_size=args.embedding_size, 
            weight_decay=args.weight_decay)
        
        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
        # arcface_logits = arcface_logits_compute(embeddings, label_batch, args, nrof_classes)

        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
            args.learning_rate_decay_epochs*args.epoch_size, args.learning_rate_decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)
        with tf.variable_scope('Logits'):
            cosface_logits = AM_logits_compute(embeddings, label_batch, args, nrof_classes)
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_batch, logits=cosface_logits,name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses',cross_entropy_mean)

        correct_prediction = tf.cast(tf.equal(tf.argmax(cosface_logits, 1), tf.cast(label_batch, tf.int64)), tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

        # for weights in slim.get_variables_by_name('embedding_weights'):
        #    weight_regularization = tf.contrib.layers.l2_regularizer(args.weight_decay)(weights)
        #    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weight_regularization)

        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        
        if args.weight_decay==0:
            total_loss = tf.add_n([cross_entropy_mean], name='total_loss')
        else:
            total_loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss')

        #define two saver in case under 'finetuning on different dataset' situation 
        saver_save = tf.train.Saver(tf.trainable_variables(), max_to_keep =1)

        train_op = facenet.train(total_loss, global_step, args.optimizer, 
            learning_rate, args.moving_average_decay, tf.global_variables(), args.log_histograms)
        summary_op = tf.summary.merge_all()

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()

        tf.train.start_queue_runners(coord=coord, sess=sess)

        with sess.as_default():
            if pretrained_model:
                print('Restoring pretrained model: %s' % pretrained_model)
                facenet.load_model(pretrained_model)

            print('Running cosface training')

            best_accuracy = 0.0
            for epoch in range(1,args.max_nrof_epochs+1):
                step = sess.run(global_step, feed_dict=None)
                cont = train(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder, labels_placeholder,
                    learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step, 
                    total_loss, accuracy, train_op, summary_op, summary_writer, regularization_losses, args.learning_rate_schedule_file, args.random_rotate, args.random_crop, args.random_flip, args.use_fixed_image_standardization,control_placeholder)

                if not cont:
                    break
                print('validation running...')
                if args.lfw_dir:
                    best_accuracy = evaluate(sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder, batch_size_placeholder, control_placeholder, embeddings, 
                        label_batch, lfw_paths, actual_issame, args.lfw_batch_size, args.lfw_nrof_folds, log_dir, step, summary_writer,best_accuracy,saver_save,model_dir, subdir,
                        args.lfw_subtract_mean, args.lfw_use_flipped_images, args.use_fixed_image_standardization, args.lfw_distance_metric)
    return model_dir
  
def find_threshold(var, percentile):
    hist, bin_edges = np.histogram(var, 100)
    cdf = np.float32(np.cumsum(hist)) / np.sum(hist)
    bin_centers = (bin_edges[:-1]+bin_edges[1:])/2
    threshold = np.interp(percentile*0.01, cdf, bin_centers)
    return threshold
  
def filter_dataset(dataset, data_filename, percentile, min_nrof_images_per_class):
    with h5py.File(data_filename,'r') as f:
        distance_to_center = np.array(f.get('distance_to_center'))
        label_list = np.array(f.get('label_list'))
        image_list = np.array(f.get('image_list'))
        distance_to_center_threshold = find_threshold(distance_to_center, percentile)
        indices = np.where(distance_to_center>=distance_to_center_threshold)[0]
        filtered_dataset = dataset
        removelist = []
        for i in indices:
            label = label_list[i]
            image = image_list[i]
            if image in filtered_dataset[label].image_paths:
                filtered_dataset[label].image_paths.remove(image)
            if len(filtered_dataset[label].image_paths)<min_nrof_images_per_class:
                removelist.append(label)

        ix = sorted(list(set(removelist)), reverse=True)
        for i in ix:
            del(filtered_dataset[i])

    return filtered_dataset
  
def train(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder, labels_placeholder, 
      learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step, 
      loss, accuracy, train_op, summary_op, summary_writer, regularization_losses, learning_rate_schedule_file, random_rotate, random_crop, random_flip, use_fixed_image_standardization,control_placeholder):
    batch_number = 0
    
    if args.learning_rate>0.0:
        lr = args.learning_rate
    else:
        lr = facenet.get_learning_rate_from_file(learning_rate_schedule_file, epoch)

    if lr<=0:
        return False

    index_epoch = sess.run(index_dequeue_op)
    label_epoch = np.array(label_list)[index_epoch]
    image_epoch = np.array(image_list)[index_epoch]
    
    # Enqueue one epoch of image paths and labels
    labels_array = np.expand_dims(np.array(label_epoch),1)
    image_paths_array = np.expand_dims(np.array(image_epoch),1)
    control_value = facenet.RANDOM_ROTATE * random_rotate + facenet.RANDOM_CROP * random_crop + facenet.RANDOM_FLIP * random_flip + facenet.FIXED_STANDARDIZATION * use_fixed_image_standardization
    control_array = np.ones_like(labels_array) * control_value
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array, control_placeholder: control_array})
    print('training a epoch...')
    # Training loop
    train_time = 0
    while batch_number < args.epoch_size:
        start_time = time.time()
        feed_dict = {learning_rate_placeholder: lr, phase_train_placeholder:True, batch_size_placeholder:args.batch_size}
        if batch_number % 100 == 0:
            err, accuracy_,  _, step, reg_loss, summary_str = sess.run([loss, accuracy, train_op, global_step, regularization_losses, summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=step)
        else:
            err, accuracy_, _, step, reg_loss = sess.run([loss, accuracy, train_op, global_step, regularization_losses], feed_dict=feed_dict)
        duration = time.time() - start_time
        print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tRegLoss %2.3f\tAccuracy %2.3f' %
              (epoch, batch_number+1, args.epoch_size, duration, err, np.sum(reg_loss), accuracy_))
        batch_number += 1
        train_time += duration
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/total', simple_value=train_time)
    summary_writer.add_summary(summary, step)
    return True

def evaluate(sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder, batch_size_placeholder, control_placeholder,
        embeddings, labels, image_paths, actual_issame, batch_size, nrof_folds, log_dir, step, summary_writer,best_accuracy,saver_save,model_dir, subdir,
        subtract_mean, use_flipped_images, use_fixed_image_standardization, distance_metric):
    start_time = time.time()
    # Run forward pass to calculate embeddings
    print('Runnning forward pass on LFW images')
    
    # Enqueue one epoch of image paths and labels
    nrof_embeddings = len(actual_issame)*2
    nrof_flips = 2 if use_flipped_images else 1
    nrof_images = nrof_embeddings*nrof_flips

    labels_array = np.expand_dims(np.arange(0,nrof_images),1)
    image_paths_array = np.expand_dims(np.repeat(np.array(image_paths),nrof_flips),1)
    control_array = np.zeros_like(labels_array, np.int32)
    if use_fixed_image_standardization:
        control_array += np.ones_like(labels_array)*facenet.FIXED_STANDARDIZATION
    if use_flipped_images:
        control_array += (labels_array % 2)*facenet.FLIP
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array, control_placeholder: control_array})
    
    embedding_size = int(embeddings.get_shape()[1])
    
    assert nrof_images % batch_size == 0, 'The number of LFW images must be an integer multiple of the LFW batch size'
    nrof_batches = nrof_images // batch_size
    emb_array = np.zeros((nrof_images, embedding_size))
    lab_array = np.zeros((nrof_images,))
    for i in range(nrof_batches):
        feed_dict = {phase_train_placeholder:False, batch_size_placeholder:batch_size}
        emb, lab = sess.run([embeddings, labels], feed_dict=feed_dict)
        lab_array[lab] = lab
        emb_array[lab, :] = emb
        if i % 10 ==9:
            print('.',end='.')
            sys.stdout.flush()
    print('')
    embeddings = np.zeros((nrof_embeddings, embedding_size*nrof_flips))
    if use_flipped_images:
        # Concatenate embeddings for flipped and non flipped version of the images
        embeddings[:,:embedding_size] = emb_array[0::2,:]
        embeddings[:,embedding_size:] = emb_array[1::2,:]
    else:
        embeddings = emb_array

    assert np.array_equal(lab_array, np.arange(nrof_images))==True, 'Wrong labels used for evaluation, possibly caused by training examples left in the input pipeline'
    _, _, accuracy, val, val_std, far = lfw.evaluate(embeddings, actual_issame, nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    
    if np.mean(accuracy) > best_accuracy:
        save_variables_and_metagraph(sess, saver_save, summary_writer, model_dir, subdir, step)
        best_accuracy = np.mean(accuracy)

    print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    lfw_time = time.time() - start_time
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='lfw/accuracy', simple_value=np.mean(accuracy))
    summary.value.add(tag='lfw/val_rate', simple_value=val)
    summary.value.add(tag='time/lfw', simple_value=lfw_time)
    summary_writer.add_summary(summary, step)
    with open(os.path.join(log_dir,'lfw_result.txt'),'at') as f:
        f.write('%d\t%.5f\t%.5f\n' % (step, np.mean(accuracy), val))
    return best_accuracy

def load_data(image_paths):
    nrof_samples = len(image_paths)
    images = np.zeros((nrof_samples, 112, 96, 3))
    for i in range(nrof_samples):
        img = misc.imread(image_paths[i])
        img = (img*1.0-127.5)/128
        images[i,:,:,:] = img
    return images

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.greater(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
  
    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    acc = float(tp+tn)/dist.size
    return tpr, fpr, acc

def evaluate_with_no_cv(emb_array, actual_issame):
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = emb_array[0::2]
    embeddings2 = emb_array[1::2]

    nrof_thresholds = len(thresholds)
    accuracys = np.zeros((nrof_thresholds))
    
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff),1)

    for threshold_idx, threshold in enumerate(thresholds):
        _, _, accuracys[threshold_idx] = facenet.calculate_accuracy(threshold, dist, actual_issame)

    best_acc = np.max(accuracys)
    best_thre = thresholds[np.argmax(accuracys)]
    return best_acc,best_thre

def evaluate_customize(sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder, batch_size_placeholder, 
        embeddings, labels, image_paths, actual_issame, batch_size, nrof_folds, log_dir, step, summary_writer,best_accuracy,saver_save,model_dir,subdir):
    start_time = time.time()
    # Run forward pass to calculate embeddings
    print('Runnning forward pass on LFW images')
    
    # Enqueue one epoch of image paths and labels
    labels_array = np.expand_dims(np.arange(0,len(image_paths)),1)
    image_paths_array = np.expand_dims(np.array(image_paths),1)
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
    
    embedding_size = embeddings.get_shape()[1]
    nrof_images = len(actual_issame)*2
    assert nrof_images % batch_size == 0, 'The number of LFW images must be an integer multiple of the LFW batch size'
    nrof_batches = nrof_images // batch_size
    emb_array = np.zeros((nrof_images, embedding_size))
    lab_array = np.zeros((nrof_images,))
    for _ in range(nrof_batches):
        feed_dict = {phase_train_placeholder:False, batch_size_placeholder:batch_size}
        emb, lab = sess.run([embeddings, labels], feed_dict=feed_dict)
        #lab_array is used for detecting whether there are some label left in the input pipeline
        lab_array[lab] = lab
        emb_array[lab] = emb

    assert np.array_equal(lab_array, np.arange(nrof_images))==True, 'Wrong labels used for evaluation, possibly caused by training examples left in the input pipeline'
    accuracy,thre = evaluate_with_no_cv(emb_array, actual_issame)
    
    print('Accuracy: %1.3f, Threshold: %1.3f' % (accuracy,thre))

    lfw_time = time.time() - start_time
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='lfw/accuracy', simple_value=accuracy)
    summary.value.add(tag='time/lfw', simple_value=lfw_time)
    summary_writer.add_summary(summary, step)

def evaluate_double(sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder, batch_size_placeholder, 
        embeddings, labels, image_paths, actual_issame, batch_size, nrof_folds, log_dir, step, summary_writer,best_accuracy,saver_save,model_dir,subdir,images_placeholder,args):
    start_time = time.time()
    
    # Run forward pass to calculate embeddings
    print('Runnning forward pass on LFW images')
    pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))
    paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs)
    batch_size = args.lfw_batch_size
    nrof_images = len(paths) #图片的数量
    nrof_batches = int(math.ceil(1.0*nrof_images / batch_size))
    #math.ceil为向上取整，意味这最后一个batch可能样本数少于batch_size
    emb_array = np.zeros((nrof_images, args.embedding_size))
    for i in range(nrof_batches):
        start_index = i*batch_size
        end_index = min((i+1)*batch_size, nrof_images)#保证最后一个batch的正确性
        paths_batch = paths[start_index:end_index]
        images = load_data(paths_batch)
        #by charles
        images_flip = np.flip(images, 2)
        feed_dict = { images_placeholder:images, phase_train_placeholder:False }
        feed_dict_flip = { images_placeholder:images_flip, phase_train_placeholder:False }
        emb = sess.run(embeddings, feed_dict=feed_dict)
        emb_flip = sess.run(embeddings, feed_dict=feed_dict_flip)
        emb_average = (emb + emb_flip)/2.0
        emb_array[start_index:end_index,:] = emb_average
        
    accuracy,thre = evaluate_with_no_cv(emb_array, actual_issame)
    
    if np.mean(accuracy) > best_accuracy:
        save_variables_and_metagraph(sess, saver_save, summary_writer, model_dir, subdir, step)
        best_accuracy = np.mean(accuracy)
    
    print('Accuracy: %1.3f Threshold: %1.3f' % (accuracy,thre))
    
    lfw_time = time.time() - start_time
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='lfw/accuracy', simple_value=accuracy)
    summary.value.add(tag='time/lfw', simple_value=lfw_time)
    summary_writer.add_summary(summary, step)
    
    return best_accuracy
    

def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0  
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)
  

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--logs_base_dir', type=str,
        help='Directory where to write event logs.', default='~/logs/facenet')
    parser.add_argument('--models_base_dir', type=str,
        help='Directory where to write trained models and checkpoints.', default='~/models/facenet')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--pretrained_model', type=str,
        help='Load a pretrained model before training starts.')
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned face patches.',
        default='~/datasets/casia/casia_maxpy_mtcnnalign_182_160')
    parser.add_argument('--model_def', type=str,
        help='Model definition. Points to a module containing the definition of the inference graph.', default='models.inception_resnet_v1')
    parser.add_argument('--max_nrof_epochs', type=int,
        help='Number of epochs to run.', default=500)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--epoch_size', type=int,
        help='Number of batches per epoch.', default=1000)
    parser.add_argument('--embedding_size', type=int,
        help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--random_crop', 
        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
         'If the size of the images in the data directory is equal to image_size no cropping is performed', action='store_false')
    parser.add_argument('--random_flip', 
        help='Performs random horizontal flipping of training images.', action='store_false')
    parser.add_argument('--random_rotate', 
        help='Performs random rotations of training images.', action='store_true')
    parser.add_argument('--use_fixed_image_standardization', 
        help='Performs fixed standardization of images.', action='store_false')
    parser.add_argument('--keep_probability', type=float,
        help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
    parser.add_argument('--weight_decay', type=float,
        help='L2 weight regularization.', default=0.0)
    parser.add_argument('--center_loss_factor', type=float,
        help='Center loss factor.', default=0.0)
    parser.add_argument('--center_loss_alfa', type=float,
        help='Center update rate for center loss.', default=0.95)
    parser.add_argument('--prelogits_norm_loss_factor', type=float,
        help='Loss based on the norm of the activations in the prelogits layer.', default=0.0)
    parser.add_argument('--prelogits_norm_p', type=float,
        help='Norm to use for prelogits norm loss.', default=1.0)
    parser.add_argument('--prelogits_hist_max', type=float,
        help='The max value for the prelogits histogram.', default=10.0)
    parser.add_argument('--learning_rate', type=float,
        help='Initial learning rate. If set to a negative value a learning rate ' +
        'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.1)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
        help='Number of epochs between learning rate decay.', default=100)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'SGD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
        help='The optimization algorithm to use', default='SGD')
    parser.add_argument('--learning_rate_decay_factor', type=float,
        help='Learning rate decay factor.', default=1.0)
    parser.add_argument('--moving_average_decay', type=float,
        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--nrof_preprocess_threads', type=int,
        help='Number of preprocessing (data loading and augmentation) threads.', default=4)
    parser.add_argument('--log_histograms', 
        help='Enables logging of weight/bias histograms in tensorboard.', action='store_true')
    parser.add_argument('--learning_rate_schedule_file', type=str,
        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.', default='data/learning_rate_schedule.txt')
    parser.add_argument('--filter_filename', type=str,
        help='File containing image data used for dataset filtering', default='')
    parser.add_argument('--filter_percentile', type=float,
        help='Keep only the percentile images closed to its class center', default=100.0)
    parser.add_argument('--filter_min_nrof_images_per_class', type=int,
        help='Keep only the classes with this number of examples or more', default=0)
    parser.add_argument('--validate_every_n_epochs', type=int,
        help='Number of epoch between validation', default=5)
    parser.add_argument('--validation_set_split_ratio', type=float,
        help='The ratio of the total dataset to use for validation', default=0.0)
    parser.add_argument('--min_nrof_val_images_per_class', type=float,
        help='Classes with fewer images will be removed from the validation set', default=0)
 
    # Parameters for validation on LFW
    parser.add_argument('--lfw_pairs', type=str,
        help='The file containing the pairs to use for validation.', default='data/pairs.txt')
    parser.add_argument('--lfw_dir', type=str,
        help='Path to the data directory containing aligned face patches.', default='')
    parser.add_argument('--lfw_file_ext', type=str,
        help='The file extension for the LFW dataset.', default='png', choices=['jpg', 'png'])
    parser.add_argument('--lfw_batch_size', type=int,
        help='Number of images to process in a batch in the LFW test set.', default=100)
    parser.add_argument('--lfw_nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    parser.add_argument('--lfw_distance_metric', type=int,
        help='Type of distance metric to use. 0: Euclidian, 1:Cosine similarity distance.', default=0)
    parser.add_argument('--lfw_use_flipped_images', 
        help='Concatenates embeddings for the image and its horizontally flipped counterpart.', action='store_true')
    parser.add_argument('--lfw_subtract_mean', 
        help='Subtract feature mean before calculating distance.', action='store_true')
    return parser.parse_args(argv)
  

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
