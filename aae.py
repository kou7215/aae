import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib
matplotlib.use('Agg')
#import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.animation as animation
from PIL import Image

LOAD_MODEL = True
LOAD_MODEL_PATH = '/home/konosuke-a/python_code/cnncancer_k/MNIST/aae_final_with_BN_ema'
BATCH_SIZE = 32
EPOCH = 750000
GIF_FREQ = int(EPOCH/100)
SAVE_DIR = '/home/konosuke-a/python_code/cnncancer_k/MNIST/aae_final_with_BN_ema'   # absolute path
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
Z_DIM = 2
EPS = 1e-12

mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

def batchnorm(input):
    with tf.variable_scope('batchnorm',reuse=tf.AUTO_REUSE):
        input = tf.identity(input)
        channels = input.get_shape()[-1]    # output channel size
        print(input.get_shape())
        offset = tf.get_variable("offset_{}".format(channels), [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale_{}".format(channels), [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = 0., 1.
        # MLP
        if len(input.get_shape()) == 2: # rank 2 tensor
            mean, variance = tf.nn.moments(input, axes=[0], keep_dims=False)
        # CNN
        if len(input.get_shape()) == 4: # rank 4 tensor
            mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized

def Save_all_variables(save_dir,sess):
    names = [v.name for v in tf.trainable_variables()]
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

with tf.variable_scope('Autoencoder'):
    with tf.variable_scope('Encoder'):
        with tf.variable_scope('Input'):
            x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 784], name='input_x')
        
        with tf.variable_scope('layer_1'):
            w1 = tf.get_variable(name='w1',shape=[784,1000], dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.02))
            b1 = tf.get_variable(name='b1',shape=[1000], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            y1 = tf.nn.relu(batchnorm(tf.matmul(x,w1) + b1))
        
        with tf.variable_scope('layer_2'):
            w2 = tf.get_variable(name='w2',shape=[1000,1000], dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.02))
            b2 = tf.get_variable(name='b1',shape=[1000], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            y2 = tf.nn.relu(batchnorm(tf.matmul(y1,w2) + b2))
        
        with tf.variable_scope('latent_z'):
            w_z = tf.get_variable(name='w_z', shape=[1000,Z_DIM],dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.02))
            b_z = tf.get_variable(name='b_z', shape=[Z_DIM],dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            z_vector = tf.matmul(y2, w_z) + b_z
    
    with tf.variable_scope('Decoder'):
        with tf.variable_scope('latent_zt'):
            w_zt = tf.get_variable(name='w_zt', shape=[Z_DIM,1000], initializer=tf.random_normal_initializer(0,0.02))
            b_zt = tf.get_variable(name='b_zt', shape=[1000], initializer=tf.constant_initializer(0.0))
            z_vectort = tf.nn.relu(batchnorm(tf.matmul(z_vector,w_zt) + b_zt))
        
        with tf.variable_scope('layer_2t'):
            w2t = tf.get_variable(name='w2t',shape=[1000,1000], dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.02))
            b2t = tf.get_variable(name='b2t',shape=[1000], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            y2t = tf.nn.relu(batchnorm(tf.matmul(z_vectort,w2t) + b2t))

        with tf.variable_scope('layer_1t'):
            w1t = tf.get_variable(name='w1t',shape=[1000,784], dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.02))
            b1t = tf.get_variable(name='b1t',shape=[784], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            y1t = tf.nn.sigmoid(batchnorm(tf.matmul(y2t,w1t) + b1t))


def Discriminator(z_input):
    with tf.variable_scope('layer_1'):
        w_dis1 = tf.get_variable(name='w_dis1', shape=[Z_DIM,1000], dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.02))
        b_dis1 = tf.get_variable(name='b_dis1', shape=[1000], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        dis1 = tf.nn.relu(batchnorm(tf.matmul(z_input,w_dis1) + b_dis1))
    with tf.variable_scope('layer_2'):
        w_dis2 = tf.get_variable(name='w_dis2', shape=[1000,1000], dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.02))
        b_dis2 = tf.get_variable(name='b_dis2', shape=[1000], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        dis2 = tf.nn.relu(batchnorm(tf.matmul(dis1,w_dis2) + b_dis2))
    with tf.variable_scope('layer_3'):
        w_dis3 = tf.get_variable(name='w_dis3', shape=[1000,1], dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.02))
        b_dis3 = tf.get_variable(name='b_dis3', shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        dis3 = tf.nn.sigmoid(tf.matmul(dis2,w_dis3) + b_dis3)   # In order to classify 2 class, need sigmoid
        return dis3

# prediction of discriminator
with tf.name_scope('discriminator_predict_fake'):
    with tf.variable_scope('Discriminator'):    # dis,gen等を関数にした場合、名前空間は呼び出し時に定義すべき(関数内で定義しない)
        predict_fake = Discriminator(z_vector)
    
with tf.name_scope('discriminator_predict_real'):
     with tf.variable_scope('Discriminator',reuse=True):
        predict_real = Discriminator(tf.random_normal([BATCH_SIZE,Z_DIM],mean=0,stddev=1)) # NOTE stddev=5.0?

# loss
with tf.name_scope('ae_loss'):
    ae_loss = tf.reduce_mean(tf.square(x-y1t))

with tf.name_scope('dis_loss'):
    dis_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

with tf.name_scope('gen_loss'):
    gen_loss = tf.reduce_mean(-tf.log(predict_fake + EPS))

# optimizer
with tf.name_scope('autoencoder_train'):
    ae_trainable_vars_list = [var for var in tf.trainable_variables() if var.name.startswith('Autoencoder')]
    ae_adam = tf.train.AdamOptimizer(0.0002,0.5)
    ae_gradients_vars = ae_adam.compute_gradients(ae_loss, var_list=ae_trainable_vars_list)
    ae_train_op = ae_adam.apply_gradients(ae_gradients_vars)

with tf.name_scope('discriminator_train'):
    dis_trainable_vars_list = [var for var in tf.trainable_variables() if var.name.startswith('Discriminator')]
    dis_adam = tf.train.AdamOptimizer(0.0002,0.5)
    dis_gradients_vars = dis_adam.compute_gradients(dis_loss, var_list=dis_trainable_vars_list)
    dis_train_op = dis_adam.apply_gradients(dis_gradients_vars)

with tf.name_scope('generator_train'):
    with tf.control_dependencies([dis_train_op]):
        gen_trainable_vars_list = [var for var in tf.trainable_variables() if var.name.startswith('Autoencoder/Encoder')]
        gen_adam = tf.train.AdamOptimizer(0.0002,0.5)
        gen_gradients_vars = gen_adam.compute_gradients(gen_loss, var_list=gen_trainable_vars_list)
        gen_train_op = gen_adam.apply_gradients(gen_gradients_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([ae_loss, dis_loss, gen_loss])

vars = tf.trainable_variables()
for i in vars:
    print(i)

with tf.name_scope('summary'):
    with tf.name_scope('Input_image_summary'):
        tf.summary.image('Input_image', tf.image.convert_image_dtype(tf.reshape(x,[BATCH_SIZE,28,28,1]), dtype=tf.uint8, saturate=True))

    with tf.name_scope('Reconstruct_image_summary'):
        tf.summary.image('Reconstruct_image', tf.image.convert_image_dtype(tf.reshape(y1t,[BATCH_SIZE,28,28,1]), dtype=tf.uint8, saturate=True))

    with tf.name_scope('Loss_summary'):
        tf.summary.scalar('ae_loss', ae_loss)
        tf.summary.scalar('dis_loss', dis_loss)
        tf.summary.scalar('gen_loss', gen_loss)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + '/Variable_histogram', var)

    for grad, var in ae_gradients_vars + dis_gradients_vars:
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
            sess.run(ae_train_op, feed_dict={x:x_batch}) 
            sess.run(dis_train_op, feed_dict={x:x_batch}) 
            sess.run(gen_train_op, feed_dict={x:x_batch}) 
            # log loss
            if step % 500 == 0:
                print('step', step)
                print('ae_loss: ', sess.run(ae_loss, feed_dict={x:x_batch}))
                print('dis_loss: ', sess.run(dis_loss, feed_dict={x:x_batch}))
                print('gen_loss: ', sess.run(gen_loss, feed_dict={x:x_batch}))
                print()
                summary_writer.add_summary(sess.run(merged, feed_dict={x:x_batch}), step)
            # plot latent z
            if step % GIF_FREQ == 0:
                batches = np.arange(0, 10000-BATCH_SIZE, BATCH_SIZE)
                z_tmp = np.zeros([np.max(batches)+BATCH_SIZE, 3])
                for i in batches:
                    x_test, t_test = mnist.test.next_batch(BATCH_SIZE)
                    z_tmp[i:i+BATCH_SIZE,0:2] = sess.run(z_vector, feed_dict={x:x_test})
                    z_tmp[i:i+BATCH_SIZE, 2] = t_test
            
                plot_q_z(z_tmp, step, xlim=(-10,10), ylim=(-10,10))
        
        Save_all_variables(save_dir=SAVE_DIR, sess=sess)
        saver.save(sess, SAVE_DIR + "/model/model.ckpt")
        Save_gif()
        print('saved!!')
    else:
        saver.restore(sess, LOAD_MODEL_PATH + '/model/model.ckpt')
        print('Load model from {}'.format(LOAD_MODEL_PATH))

#   # test (plot latent z-vector)
#    x_test, t_test = mnist.test.images, mnist.test.labels
#    batches = np.arange(0,10000-32,32)
#    z_tmp = np.zeros([np.max(batches)+32, 2])
#    for i in batches:
#        z_tmp[i:i+32] = sess.run(z_vector, feed_dict={x:x_test[i:i+32]})
#    print(z_tmp.shape)
#    
#    plt.figure()
#    plt.title("Plot latent vector of AAE")
#    for i in range(len(z_tmp)):
#        if t_test[i] == 0:
#            p0 = plt.scatter(z_tmp[i,0], z_tmp[i, 1],c="red",s=1)
#        if t_test[i] == 1:
#            p1 = plt.scatter(z_tmp[i,0], z_tmp[i, 1], c="blue",s=1)
#        if t_test[i] == 2:
#            p2 = plt.scatter(z_tmp[i,0], z_tmp[i, 1], c="green",s=1)
#        if t_test[i] == 3:
#            p3 = plt.scatter(z_tmp[i,0], z_tmp[i, 1], c="pink",s=1)
#        if t_test[i] == 4:
#            p4 = plt.scatter(z_tmp[i,0], z_tmp[i, 1], c="yellow",s=1)
#        if t_test[i] == 5:
#            p5 = plt.scatter(z_tmp[i,0], z_tmp[i, 1], c="orange",s=1)
#        if t_test[i] == 6:
#            p6 = plt.scatter(z_tmp[i,0], z_tmp[i, 1], c="cyan",s=1)
#        if t_test[i] == 7:
#            p7 = plt.scatter(z_tmp[i,0], z_tmp[i, 1], c="deeppink",s=1)
#        if t_test[i] == 8:
#            p8 = plt.scatter(z_tmp[i,0], z_tmp[i, 1], c="c",s=1)
#        if t_test[i] == 9:
#            p9 = plt.scatter(z_tmp[i,0], z_tmp[i, 1], c="purple",s=1)
#    plt.legend([p0,p1,p2,p3,p4,p5,p6,p7,p8,p9],["0","1","2","3","4","5","6","7","8","9"], bbox_to_anchor=(1.01,1), loc=2, borderaxespad=0)
#    filename = "AAE_latent_ver3.png"
##    axes = plt.gca()
##    axes.set_xlim([-4.0, 4.0])
##    axes.set_ylim([-4.0, 4.0])
#    plt.savefig(filename)
#    
