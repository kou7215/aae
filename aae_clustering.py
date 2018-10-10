import tensorflow as tf
import numpy as np
import os
import math
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib
matplotlib.use('Agg')
#import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.animation as animation
from PIL import Image

LOAD_MODEL = False
LOAD_MODEL_PATH = '/home/konosuke-a/python_code/cnncancer_k/MNIST/aae_clustering'
BATCH_SIZE = 32
EPOCH = 187500
GIF_FREQ = int(EPOCH/100)
SAVE_DIR = '/home/konosuke-a/python_code/cnncancer_k/MNIST/aae_clustering_y20_withBN_threshold2_drop'   # absolute path
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
Y_DIM = 20
Z_DIM = 2
CLUSTER_DIST_THRESHOLD = 2.0
EPS = 1e-12

mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

def batchnorm(input):
    with tf.variable_scope('batchnorm',reuse=tf.AUTO_REUSE):
        input = tf.identity(input)
        channels = input.get_shape()[-1]
        print(input.get_shape())
        offset = tf.get_variable("offset_{}".format(channels), [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale_{}".format(channels), [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[0], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized

def Save_all_variables(save_dir,sess):
    names = [v.name for v in tf.trainable_variables()]
    print(names)
    for n in names:
        v = tf.get_default_graph().get_tensor_by_name(n)
        save_name = n.split('/')[-1].split(':')[0]
        print('saved variables  ', save_dir +'/variables/'+ save_name + '.npy')
        np.save(save_dir +'/variables/'+ save_name + '.npy', sess.run(v))

def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1

def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2

def P(n, r):
    return math.factorial(n)//math.factorial(n-r)

def C(n, r):
    return P(n, r)//math.factorial(r)


def plot_q_z(data, step, n_label=10, xlim=None, ylim=None):
    plt.clf()
    fig, ax = plt.subplots(ncols=1, figsize=(8,8))
    color = cm.rainbow(np.linspace(0,1,n_label))
    for l, c in zip(range(10), color):
        ix = np.where(data[:,2]==l)
        ax.scatter(data[ix,0], data[ix, 1], c=c, label=l, s=8, linewidth=0)
        ax.legend()
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    plt.title('step = {0:8d}'.format(step))
    plt.suptitle('latent variable z of AAE')
    plt.savefig(SAVE_DIR+ '/latent_z_png' + '/AAE_z_step_{}'.format(step))
    plt.close()

def Save_gif():
    images = []
    for i in range(0,EPOCH,GIF_FREQ):
        images.append(Image.open(SAVE_DIR + '/latent_z_png' + '/AAE_z_step_{}.png'.format(i)))
    
    images[0].save(SAVE_DIR + '/' + SAVE_DIR.split('/')[-1] +'.gif', save_all=True, append_images=images[1:],
                    optimize=False, duration=100, loop=0)

def Encoder(x,y_dim,z_dim):
    with tf.variable_scope('Encoder'):
        with tf.variable_scope('layer1'):
            w1 = tf.get_variable(name='w1',shape=[784,3000], dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.02))
            b1 = tf.get_variable(name='b1',shape=[3000], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            out1 = tf.nn.relu(batchnorm(tf.matmul(x,w1) + b1))
        with tf.variable_scope('layer2'):
            w2 = tf.get_variable(name='w2',shape=[3000,3000], dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.02))
            b2 = tf.get_variable(name='b2',shape=[3000], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            out2 = tf.nn.relu(batchnorm(tf.matmul(out1,w2) + b2))
        with tf.variable_scope('layer3_y'):
            w3_y = tf.get_variable(name='w3_y',shape=[3000,y_dim], dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.02))
            b3_y = tf.get_variable(name='b3_y',shape=[y_dim], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            out_y = tf.nn.softmax(batchnorm(tf.matmul(out2,w3_y) + b3_y))
        with tf.variable_scope('layer3_z'):
            w3_z = tf.get_variable(name='w3_z',shape=[3000,z_dim], dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.02))
            b3_z = tf.get_variable(name='b3_z',shape=[z_dim], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            out_z = batchnorm(tf.matmul(out2,w3_z) + b3_z)
    return (out_y, out_z)

def Cluster_head(y,z_dim):
    bs_y, ch_y = [int(i) for i in y.get_shape()]
    w_clus = tf.get_variable(name='w_clus',shape=[ch_y,z_dim], dtype=tf.float32, initializer=tf.random_normal_initializer(0.,1.0))   #NOTE (0.,0.02)?
    b_clus = tf.get_variable(name='b_clus',shape=[z_dim], dtype=tf.float32, initializer=tf.constant_initializer(0.0))  
    out_clus = tf.matmul(y,w_clus) + b_clus
    return out_clus

def Decoder(yz):
    bs, ch = [int(i) for i in yz.get_shape()]
    with tf.variable_scope('layer1'):
        w1 = tf.get_variable(name='w1',shape=[ch,3000], dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.02))
        b1 = tf.get_variable(name='b1',shape=[3000], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        out1 = tf.nn.relu(tf.matmul(yz,w1) + b1)
    with tf.variable_scope('layer2'):
        w2 = tf.get_variable(name='w2',shape=[3000,3000], dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.02))
        b2 = tf.get_variable(name='b2',shape=[3000], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        out2 = tf.nn.relu(tf.matmul(out1,w2) + b2)
    with tf.variable_scope('layer3'):
        w3 = tf.get_variable(name='w3',shape=[3000,784], dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.02))
        b3 = tf.get_variable(name='b3',shape=[784], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        out3 = tf.nn.sigmoid(tf.matmul(out2,w3) + b3)
    return out3
 
def Discriminator(inputs):
    bs,ch = [int(i) for i in inputs.get_shape()]
    with tf.variable_scope('layer_1'):
        w = tf.get_variable(name='w1', shape=[ch,3000], dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.02))
        b = tf.get_variable(name='b1', shape=[3000], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        out = tf.nn.relu(tf.matmul(inputs,w) + b)
    with tf.variable_scope('layer_2'):
        w = tf.get_variable(name='w2', shape=[3000,3000], dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.02))
        b = tf.get_variable(name='b2', shape=[3000], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        out = tf.nn.relu(tf.matmul(out,w) + b)
    with tf.variable_scope('layer_3'):
        w = tf.get_variable(name='w3', shape=[3000,1], dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.02))
        b = tf.get_variable(name='b3', shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        out = tf.nn.sigmoid(tf.matmul(out,w) + b)   # In order to classify 2 class, need sigmoid
        return out

# define model
with tf.variable_scope('Autoencoder'):
    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 784], name='input_x')
    dropout_rate = tf.placeholder(tf.float32, name='dropout_rate')
    dropped_x = tf.nn.dropout(x, dropout_rate)
    encode_y, encode_z = Encoder(dropped_x, y_dim=Y_DIM, z_dim=Z_DIM)
    with tf.variable_scope('cluster_head'):
        cluster_head_y = Cluster_head(encode_y, z_dim=Z_DIM)
        cluster_head_yz = cluster_head_y + encode_z
    with tf.variable_scope('Decoder'):
        decode_yz = Decoder(cluster_head_yz)

with tf.name_scope('Cluster_head_labels'):
    num_combination = C(Y_DIM,Z_DIM)
    a_labels = np.zeros((num_combination, Y_DIM),dtype=np.float32)
    for i in range(1, Y_DIM):
        for n in range(i):
            j = int(0.5*i*(i-1) + n)
            a_labels[j,i] = 1
    b_labels = np.zeros((num_combination, Y_DIM), dtype=np.float32)
    for i in range(1, Y_DIM):
        for n in range(i):
            j = int(0.5 * i * (i - 1) + n)
            b_labels[j, n] = 1
    a_labels = tf.constant(a_labels, name='a_labels')
    b_labels = tf.constant(b_labels, name='b_labels')
    with tf.variable_scope('Autoencoder',reuse=True):
        with tf.variable_scope('cluster_head',reuse=True):
            cluster_head_a = Cluster_head(a_labels, z_dim=Z_DIM)
    with tf.variable_scope('Autoencoder',reuse=True):
        with tf.variable_scope('cluster_head',reuse=True):
            cluster_head_b = Cluster_head(b_labels, z_dim=Z_DIM)

# Discriminator1 is part of latent y
with tf.name_scope('Discriminator_y_predict_fake'):
    with tf.variable_scope('Discriminator_y'):   # TODO reuse
        predict_fake_y = Discriminator(encode_y)

with tf.name_scope('Discriminator_y_predict_real'):
    with tf.variable_scope('Discriminator_y', reuse=True):   # Categorical distribution
        indices = tf.random_uniform(shape=[BATCH_SIZE], minval=0, maxval=Y_DIM, dtype=tf.int32)    # minval <= x < maxval
        cat = tf.one_hot(indices=indices, depth=Y_DIM, name='categorical_dist')
        predict_real_y = Discriminator(cat)

# Discriminator2 is part of latent z
with tf.name_scope('Discriminator_z_predict_fake'):
    with tf.variable_scope('Discriminator_z'):
        predict_fake_z = Discriminator(encode_z)

with tf.name_scope('Discriminator_z_predict_real'):
    with tf.variable_scope('Discriminator_z', reuse=True):
        predict_real_z = Discriminator(tf.random_normal([BATCH_SIZE, Z_DIM], mean=0.0, stddev=1.0))

# loss
with tf.name_scope('ae_loss'):
    ae_loss = tf.reduce_mean(tf.square(x-decode_yz))

with tf.name_scope('dis_y_loss'):
    dis_y_loss = tf.reduce_mean(-(tf.log(predict_real_y + EPS) + tf.log(1 - predict_fake_y + EPS)))

with tf.name_scope('gen_y_loss'):
    gen_y_loss = tf.reduce_mean(-tf.log(predict_fake_y + EPS))

with tf.name_scope('dis_z_loss'):
    dis_z_loss = tf.reduce_mean(-(tf.log(predict_real_z + EPS) + tf.log(1 - predict_fake_z + EPS)))

with tf.name_scope('gen_z_loss'):
    gen_z_loss = tf.reduce_mean(-tf.log(predict_fake_z + EPS))

with tf.name_scope('clus_loss'):
    clus_mse = tf.reduce_mean(tf.square(cluster_head_a - cluster_head_b))
    clus_loss = -tf.clip_by_value(clus_mse, 0.0, CLUSTER_DIST_THRESHOLD)     # 0 <= MSE <= CLUSTER_DIST_THRESHOLD

# optimizer
with tf.name_scope('autoencoder_train'):
    ae_trainable_vars_list = [var for var in tf.trainable_variables() if var.name.startswith('Autoencoder')]
    print('ae_trainable_vars_list : ',ae_trainable_vars_list)
    ae_adam = tf.train.AdamOptimizer(0.0002,0.5)
    ae_gradients_vars = ae_adam.compute_gradients(ae_loss, var_list=ae_trainable_vars_list)
    ae_train_op = ae_adam.apply_gradients(ae_gradients_vars)

with tf.name_scope('discriminator_y_train'):
    dis_y_trainable_vars_list = [var for var in tf.trainable_variables() if var.name.startswith('Discriminator_y')]
    dis_y_adam = tf.train.AdamOptimizer(0.0002,0.5)
    dis_y_gradients_vars = dis_y_adam.compute_gradients(dis_y_loss, var_list=dis_y_trainable_vars_list)
    dis_y_train_op = dis_y_adam.apply_gradients(dis_y_gradients_vars)

with tf.name_scope('generator_y_train'):
    gen_y_trainable_vars_list = [var for var in tf.trainable_variables() if (not var.name.startswith('Autoencoder/Encoder/layer3_z')) and (var.name.startswith('Autoencoder/Encoder'))]
    gen_y_adam = tf.train.AdamOptimizer(0.0002,0.5)
    gen_y_gradients_vars = gen_y_adam.compute_gradients(gen_y_loss, var_list=gen_y_trainable_vars_list)
    gen_y_train_op = gen_y_adam.apply_gradients(gen_y_gradients_vars)

with tf.name_scope('discriminator_z_train'):
    dis_z_trainable_vars_list = [var for var in tf.trainable_variables() if var.name.startswith('Discriminator_z')]
    dis_z_adam = tf.train.AdamOptimizer(0.0002,0.5)
    dis_z_gradients_vars = dis_z_adam.compute_gradients(dis_z_loss, var_list=dis_z_trainable_vars_list)
    dis_z_train_op = dis_z_adam.apply_gradients(dis_z_gradients_vars)

with tf.name_scope('generator_z_train'):
    gen_z_trainable_vars_list = [var for var in tf.trainable_variables() if (not var.name.startswith('Autoencoder/Encoder/layer3_y')) and (var.name.startswith('Autoencoder/Encoder'))]
    gen_z_adam = tf.train.AdamOptimizer(0.0002,0.5)
    gen_z_gradients_vars = gen_z_adam.compute_gradients(gen_z_loss, var_list=gen_z_trainable_vars_list)
    gen_z_train_op = gen_z_adam.apply_gradients(gen_z_gradients_vars)

with tf.name_scope('cluster_head_train'):
    cluster_head_trainable_vars_list = [var for var in tf.trainable_variables() if var.name.startswith('Autoencoder/cluster_head')]
    print('cluster_head_trainable_vars_list : ',cluster_head_trainable_vars_list)
    cluster_head_adam = tf.train.AdamOptimizer(0.0002,0.5)
    cluster_head_gradients_vars = cluster_head_adam.compute_gradients(clus_loss, var_list=cluster_head_trainable_vars_list)
    cluster_head_train_op = cluster_head_adam.apply_gradients(cluster_head_gradients_vars)

with tf.name_scope('summary'):
    with tf.name_scope('Input_image_summary'):
        tf.summary.image('Input_image', tf.image.convert_image_dtype(tf.reshape(x,[BATCH_SIZE,28,28,1]), dtype=tf.uint8, saturate=True))

    with tf.name_scope('Reconstruct_image_summary'):
        tf.summary.image('Reconstruct_image', tf.image.convert_image_dtype(tf.reshape(decode_yz,[BATCH_SIZE,28,28,1]), dtype=tf.uint8, saturate=True))

    with tf.name_scope('Loss_summary'):
        tf.summary.scalar('ae_loss', ae_loss)
        tf.summary.scalar('dis_y_loss', dis_y_loss)
        tf.summary.scalar('gen_y_loss', gen_y_loss)
        tf.summary.scalar('dis_z_loss', dis_z_loss)
        tf.summary.scalar('gen_z_loss', gen_z_loss)
        tf.summary.scalar('clus_loss', clus_loss)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + '/Variable_histogram', var)

    for grad, var in ae_gradients_vars + dis_y_gradients_vars + dis_z_gradients_vars + cluster_head_gradients_vars:
        tf.summary.histogram(var.op.name + '/Gradients', grad)

# Session
init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session(config=config) as sess:
    if LOAD_MODEL is not True:
        sess.run(init)
    
        # mkdir if not exist directory
        if not os.path.exists(SAVE_DIR): # NOT CHANGE
            os.mkdir(SAVE_DIR)
            os.mkdir(os.path.join(SAVE_DIR,'summary'))
            os.mkdir(os.path.join(SAVE_DIR,'variables'))
            os.mkdir(os.path.join(SAVE_DIR,'model'))
            os.mkdir(os.path.join(SAVE_DIR, 'latent_z_png'))
        
        # remove old summary if already exist
        if tf.gfile.Exists(os.path.join(SAVE_DIR,'summary')):    # NOT CHANGE
            tf.gfile.DeleteRecursively(os.path.join(SAVE_DIR,'summary'))
        
        # merging summary & set summary writer
        merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(os.path.join(SAVE_DIR,'summary'), graph=sess.graph)
    
        # train
        for step in range(EPOCH):
            x_batch, t_batch = mnist.train.next_batch(BATCH_SIZE)
            sess.run(ae_train_op, feed_dict={x:x_batch, dropout_rate:0.8}) 
            sess.run(dis_y_train_op, feed_dict={x:x_batch,dropout_rate:0.8}) 
            sess.run(gen_y_train_op, feed_dict={x:x_batch,dropout_rate:0.8}) 
            sess.run(dis_z_train_op, feed_dict={x:x_batch,dropout_rate:0.8}) 
            sess.run(gen_z_train_op, feed_dict={x:x_batch,dropout_rate:0.8}) 
            sess.run(cluster_head_train_op, feed_dict={x:x_batch, dropout_rate:0.8}) 

            #
            # log loss
            if step % 500 == 0:
                print('step', step)
                print('ae_loss: ', sess.run(ae_loss, feed_dict={x:x_batch,dropout_rate:0.8}))
                print('dis_y_loss: ', sess.run(dis_y_loss, feed_dict={x:x_batch, dropout_rate:0.8}))
                print('gen_y_loss: ', sess.run(gen_y_loss, feed_dict={x:x_batch, dropout_rate:0.8}))
                print('dis_z_loss: ', sess.run(dis_z_loss, feed_dict={x:x_batch, dropout_rate:0.8}))
                print('gen_z_loss: ', sess.run(gen_z_loss, feed_dict={x:x_batch, dropout_rate:0.8}))
                print('cluster_head_loss: ', sess.run(clus_loss, feed_dict={x:x_batch, dropout_rate:0.8}))
                print()
                summary_writer.add_summary(sess.run(merged, feed_dict={x:x_batch, dropout_rate:0.8}), step)
            # plot latent z
            if step % GIF_FREQ == 0:
                batches = np.arange(0, 10000-BATCH_SIZE, BATCH_SIZE)
                z_tmp = np.zeros([np.max(batches)+BATCH_SIZE, 3])
                for i in batches:
                    x_test, t_test = mnist.test.next_batch(BATCH_SIZE)
                    z_tmp[i:i+BATCH_SIZE,0:2] = sess.run(cluster_head_yz, feed_dict={x:x_test, dropout_rate:1.0})
                    z_tmp[i:i+BATCH_SIZE, 2] = t_test
            
                plot_q_z(z_tmp, step, xlim=(-10,10), ylim=(-10,10))
        
        Save_all_variables(save_dir=SAVE_DIR, sess=sess)
        saver.save(sess, SAVE_DIR + "/model/model.ckpt")
        Save_gif()
        print('saved!!')
    else:
        saver.restore(sess, LOAD_MODEL_PATH + '/model/model.ckpt')
        print('Load model from {}'.format(LOAD_MODEL_PATH))

