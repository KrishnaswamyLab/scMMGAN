import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import time
##########

############################################################
####################
# LOAD TWO DOMAINS OF PREPROCESSED DATA INTO x1, x2

x1 = np.random.normal(-5, 1, [10000, 100])
x2 = np.random.normal(5, 1, [10000, 100])
##########

############################################################
####################
# ADJUST THESE TO ACCOMODATE COMPUTATIONAL CONSTRAINTS
TRAINING_STEPS = 20000
batch_size = 64
nfilt = 128


# HYPERPARAMETERS THAT DEPEND ON DATA CHARACTERISTICS
learning_rate = .0001
lambda_cycle = 1
lambda_correspondence = 1


# OPTIONS
add_noise = False
use_bn = True
##########


############################################################
####################
# DATA LOADING

class Loader(object):
    def __init__(self, data, labels=None, shuffle=False):
        """Initialize the loader with data and optionally with labels."""
        self.start = 0
        self.epoch = 0
        self.data = [x for x in [data, labels] if x is not None]
        self.labels_given = labels is not None

        if shuffle:
            self.r = list(range(data.shape[0]))
            np.random.shuffle(self.r)
            self.data = [x[self.r] for x in self.data]

    def next_batch(self, batch_size=100):
        """Yield just the next batch."""
        num_rows = self.data[0].shape[0]

        if self.start + batch_size < num_rows:
            batch = [x[self.start:self.start + batch_size] for x in self.data]
            self.start += batch_size
        else:
            self.epoch += 1
            batch_part1 = [x[self.start:] for x in self.data]
            batch_part2 = [x[:batch_size - (x.shape[0] - self.start)] for x in self.data]
            batch = [np.concatenate([x1, x2], axis=0) for x1, x2 in zip(batch_part1, batch_part2)]

            self.start = batch_size - (num_rows - self.start)

        if not self.labels_given:  # don't return length-1 list
            return batch[0]
        else:  # return list of data and labels
            return batch

    def iter_batches(self, batch_size=100):
        """Iterate over the entire dataset in batches."""
        num_rows = self.data[0].shape[0]

        end = 0

        if batch_size > num_rows:
            if not self.labels_given:
                yield [x for x in self.data][0]
            else:
                yield [x for x in self.data]
        else:
            for i in range(num_rows // batch_size):
                start = i * batch_size
                end = (i + 1) * batch_size

                if not self.labels_given:
                    yield [x[start:end] for x in self.data][0]
                else:
                    yield [x[start:end] for x in self.data]
            if end < num_rows:
                if not self.labels_given:
                    yield [x[end:] for x in self.data][0]
                else:
                    yield [x[end:] for x in self.data]

load1 = Loader(x1, shuffle=True)
load2 = Loader(x2, shuffle=True)
loadeval1 = Loader(x1, shuffle=False)
loadeval2 = Loader(x2, shuffle=False)
outdim1 = x1.shape[1]
outdim2 = x2.shape[1]
##########

############################################################
####################
# TF GRAPH

def tbn(name):

    return tf.get_default_graph().get_tensor_by_name(name)

def minibatch(input_, num_kernels=15, kernel_dim=10, name='',):
    with tf.variable_scope(name):
        W = tf.get_variable('{}/Wmb'.format(name), [input_.get_shape()[-1], num_kernels * kernel_dim])
        b = tf.get_variable('{}/bmb'.format(name), [num_kernels * kernel_dim])

    x = tf.matmul(input_, W) + b
    activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
    diffs = tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
    abs_diffs = tf.reduce_mean(tf.abs(diffs), 2)
    minibatch_features = tf.reduce_mean(tf.exp(-abs_diffs), 2)

    return tf.concat([input_, minibatch_features], axis=-1)

def nameop(op, name):

    return tf.identity(op, name=name)

def lrelu(x, leak=0.2, name="lrelu"):

    return tf.maximum(x, leak * x)

def bn(tensor, name, is_training):
    if not use_bn:
        return tensor
    return tf.layers.batch_normalization(tensor,
                      momentum=.9,
                      training=True,
                      name=name)

def get_layer(sess, intensor, data, outtensor, batch_size=100):
    out = []
    for batch in np.array_split(data, data.shape[0]/batch_size):
        feed = {intensor: batch}
        batchout = sess.run(outtensor, feed_dict=feed)
        out.append(batchout)
    out = np.concatenate(out, axis=0)

    return out

def Generator(x, nfilt, outdim, activation=lrelu, is_training=True):
    h1 = tf.layers.dense(x, nfilt * 1, activation=None, name='h1')
    h1 = bn(h1, 'h1', is_training)
    h1 = activation(h1)

    h2 = tf.layers.dense(h1, nfilt * 2, activation=None, name='h2')
    h2 = bn(h2, 'h2', is_training)
    h2 = activation(h2)

    h3 = tf.layers.dense(h2, nfilt * 4, activation=None, name='h3')
    h3 = bn(h3, 'h3', is_training)
    h3 = activation(h3)

    out = tf.layers.dense(h3, outdim, activation=None, name='out')

    return out

def Discriminator(x, nfilt, outdim, activation=lrelu, is_training=True):
    h1 = tf.layers.dense(x, nfilt * 4, activation=None, name='h1')
    h1 = activation(h1)
    h1 = minibatch(h1)

    h2 = tf.layers.dense(h1, nfilt * 2, activation=None, name='h2')
    h2 = bn(h2, 'h2', is_training)
    h2 = activation(h2)

    h3 = tf.layers.dense(h2, nfilt * 1, activation=None, name='h3')
    h3 = bn(h3, 'h3', is_training)
    h3 = activation(h3)

    out = tf.layers.dense(h3, outdim, activation=None, name='out')

    return out

def adversarial_loss(logits, labels):

    return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)

def compute_pairwise_distances(A):
    r = tf.reduce_sum(A*A, 1)

    # turn r into column vector
    r = tf.reshape(r, [-1, 1])
    D = r - 2*tf.matmul(A, tf.transpose(A)) + tf.transpose(r)

    return D

def kernel(dists, sigmas=None, use_affinities=True):
    if not use_affinities:
        return dists

    if not sigmas:
        sigmas = tf.sort(dists, axis=1)[:, 5][:, np.newaxis]

    affinities = np.e**(- (dists / (sigmas)))

    affinities = (affinities + tf.transpose(affinities)) / 2

    sqrt_rowsum = tf.sqrt(tf.reduce_sum(affinities, axis=1, keep_dims=True))
    sqrt_colsum = tf.sqrt(tf.reduce_sum(affinities, axis=0, keep_dims=True))
    affinities = affinities / sqrt_rowsum
    affinities = affinities / sqrt_colsum

    return affinities

tf.reset_default_graph()
loss_D = 0.
loss_G = 0.
tfis_training = tf.placeholder(tf.bool, [], name='tfis_training')

tfx1 = tf.placeholder(tf.float32, [None, outdim1], name='x1')
tfx2 = tf.placeholder(tf.float32, [None, outdim2], name='x2')

if add_noise:
    tfz1 = tf.random.normal(shape=[tf.shape(tfx1)[0], outdim1 // 2])
    tfz2 = tf.random.normal(shape=[tf.shape(tfx2)[0], outdim2 // 2])


with tf.variable_scope('generator12', reuse=tf.AUTO_REUSE):
    if not add_noise:
        input_ = tfx1
    else:
        input_ = tf.concat([tfx1, tfz1], axis=-1)
    fake2 = Generator(input_, nfilt, outdim=outdim2, is_training=tfis_training)
fake2 = nameop(fake2, 'fake2')

with tf.variable_scope('generator21', reuse=tf.AUTO_REUSE):
    if not add_noise:
        input_ = tfx2
    else:
        input_ = tf.concat([tfx2, tfz2], axis=-1)
    fake1 = Generator(input_, nfilt, outdim=outdim1, is_training=tfis_training)
fake1 = nameop(fake1, 'fake1')

with tf.variable_scope('generator12', reuse=tf.AUTO_REUSE):
    if not add_noise:
        input_ = fake1
    else:
        input_ = tf.concat([fake1, tfz2], axis=-1)
    cycle2 = Generator(input_, nfilt, outdim=outdim2, is_training=tfis_training)
cycle2 = nameop(cycle2, 'cycle2')

with tf.variable_scope('generator21', reuse=tf.AUTO_REUSE):
    if not add_noise:
        input_ = fake2
    else:
        input_ = tf.concat([fake2, tfz1], axis=-1)
    cycle1 = Generator(input_, nfilt, outdim=outdim1, is_training=tfis_training)
cycle1 = nameop(cycle1, 'cycle1')


with tf.variable_scope('discriminator1', reuse=tf.AUTO_REUSE):
    d_real1 = Discriminator(tfx1, 2 * nfilt, 1, is_training=tfis_training)
    d_fake1 = Discriminator(fake1, 2 * nfilt, 1, is_training=tfis_training)

with tf.variable_scope('discriminator2', reuse=tf.AUTO_REUSE):
    d_real2 = Discriminator(tfx2, 2 * nfilt, 1, is_training=tfis_training)
    d_fake2 = Discriminator(fake2, 2 * nfilt, 1, is_training=tfis_training)

real = tf.concat([d_real1, d_real2], axis=0)
fake = tf.concat([d_fake1, d_fake2], axis=0)
##########



############################################################
####################
# GAN AND CYCLE LOSSES

loss_D_fake = tf.reduce_mean(adversarial_loss(logits=real, labels=tf.ones_like(real)))
loss_D_real = tf.reduce_mean(adversarial_loss(logits=fake, labels=tf.zeros_like(fake)))
loss_G_disc = tf.reduce_mean(adversarial_loss(logits=fake, labels=tf.ones_like(fake)))

loss_D += .5 * (loss_D_fake + loss_D_real)
loss_G += loss_G_disc

tf.add_to_collection('losses', nameop(loss_D_real, 'loss_D_real'))
tf.add_to_collection('losses', nameop(loss_D_fake, 'loss_D_fake'))
tf.add_to_collection('losses', nameop(loss_G_disc, 'loss_G_disc'))

loss_cycle = tf.reduce_mean((tfx1 - cycle1)**2) + tf.reduce_mean((tfx2 - cycle2)**2)
loss_G += lambda_cycle * loss_cycle
tf.add_to_collection('losses', nameop(loss_cycle, 'loss_cycle'))
##########

############################################################
####################
# CORRESPONDENCE LOSS

def diffusion_eigenvector_loss(x1, x2, x1_mappedto_x2, x2_mappedto_x1, n_eigenvectors=1):
    def compute_pairwise_distances(A):
        r = tf.reduce_sum(A*A, 1)

        r = tf.reshape(r, [-1, 1])
        D = r - 2*tf.matmul(A, tf.transpose(A)) + tf.transpose(r)

        return D

    def dist(a, b):

        return tf.reduce_mean((a - b)**2)

    def kernel(dists, sigmas=None, power=10):
        dists = 1 - dists / tf.sort(dists, axis=1)[:, tf.shape(dists)[0] // 10][:, np.newaxis]
        dists = tf.maximum(dists, tf.zeros_like(dists))
        dists = dists / tf.reduce_sum(dists, axis=1, keep_dims=True)
        for _ in range(power):
            dists = tf.matmul(dists, dists)
            dists = dists / tf.reduce_sum(dists, axis=1, keep_dims=True)
        return dists

    def rescale(data):
        newdata = []
        for i in range(n_eigenvectors):
            col = data[:, i]
            col = col - tf.reduce_min(col)
            col = col / tf.reduce_max(col)
            col = 2 * col - 1
            newdata.append(col)
        newdata = tf.stack(newdata, axis=-1)
        return newdata

    def check_for_anticorrelation(data1, data2, thresh=-.5):
        modifier = []
        for i in range(n_eigenvectors):
            col1 = data1[:, i][:, np.newaxis]
            col2 = data2[:, i][:, np.newaxis]
            r = tfp.stats.correlation(col1, col2)[0, 0]
            modifier.append(tf.cond(r < thresh, lambda: tf.constant(-1.), lambda: tf.constant(1.)))
        modifier = tf.stack(modifier)
        return modifier

    loss = []
    x1_k = kernel(compute_pairwise_distances(x1))
    x2_k = kernel(compute_pairwise_distances(x2))
    eigv_x1 = rescale(tf.linalg.eigh(x1_k)[1][:, :n_eigenvectors])
    eigv_x2 = rescale(tf.linalg.eigh(x2_k)[1][:, :n_eigenvectors])

    x1_mappedto_x2_k = kernel(compute_pairwise_distances(x1_mappedto_x2))
    x2_mappedto_x1_k = kernel(compute_pairwise_distances(x2_mappedto_x1))
    eigv_g12 = rescale(tf.linalg.eigh(x1_mappedto_x2_k)[1][:, :n_eigenvectors])
    eigv_g21 = rescale(tf.linalg.eigh(x2_mappedto_x1_k)[1][:, :n_eigenvectors])

    eigv_x1 = nameop(eigv_x1, 'eigv_x1')
    eigv_g12 = nameop(eigv_g12, 'eigv_g12')

    eigv_g12 = check_for_anticorrelation(eigv_x1, eigv_g12) * eigv_g12
    eigv_g21 = check_for_anticorrelation(eigv_x1, eigv_g21) * eigv_g21
    loss.append(dist(eigv_x1, eigv_g12))
    loss.append(dist(eigv_x2, eigv_g21))

    for bin_size in [2, 4]:
        newvec_x1 = []
        newvec_x2 = []
        newvec_g12 = []
        newvec_g21 = []
        for bin in np.array_split(range(n_eigenvectors), n_eigenvectors // bin_size):
            newvec_x1.append(tf.reduce_sum(tf.gather(eigv_x1, bin, axis=-1), axis=-1))
            newvec_x2.append(tf.reduce_sum(tf.gather(eigv_x2, bin, axis=-1), axis=-1))
            newvec_g12.append(tf.reduce_sum(tf.gather(eigv_g12, bin, axis=-1), axis=-1))
            newvec_g21.append(tf.reduce_sum(tf.gather(eigv_g21, bin, axis=-1), axis=-1))
        newvec_x1 = tf.stack(newvec_x1, axis=-1)
        newvec_x2 = tf.stack(newvec_x2, axis=-1)
        newvec_g12 = tf.stack(newvec_g12, axis=-1)
        newvec_g21 = tf.stack(newvec_g21, axis=-1)

        loss.append((1. / bin_size) * dist(newvec_x1, newvec_g12))
        loss.append((1. / bin_size) * dist(newvec_x2, newvec_g21))

    loss = tf.reduce_mean(loss)

    return loss


loss_correspondence = diffusion_eigenvector_loss(tfx1, tfx2, fake2, fake1, n_eigenvectors=12)


loss_G += lambda_correspondence * loss_correspondence
tf.add_to_collection('losses', nameop(loss_correspondence, 'loss_correspondence'))
##########


############################################################
####################
# COLLECT UPDATE OPS

Gvars = [tv for tv in tf.global_variables() if 'generator' in tv.name]
Dvars = [tv for tv in tf.global_variables() if 'discriminator' in tv.name]

update_ops_D = [op for op in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if 'discriminator' in op.name]
update_ops_G = [op for op in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if 'generator' in op.name]
print('update ops G: {}'.format(len(update_ops_G)))
print('update ops D: {}'.format(len(update_ops_D)))

with tf.control_dependencies(update_ops_D):
    optD = tf.train.AdamOptimizer(learning_rate)
    train_op_D = optD.minimize(loss_D, var_list=Dvars)
with tf.control_dependencies(update_ops_G):
    optG = tf.train.AdamOptimizer(learning_rate)
    train_op_G = optG.minimize(loss_G, var_list=Gvars)
##########

############################################################
####################
# INITIALIZE SESSION

sess = tf.Session()

sess.run(tf.global_variables_initializer())
##########


############################################################
####################
# TRAIN

t = time.time()
training_counter = 0
losses = [tns.name[:-2].replace('loss_', '').split('/')[-1] for tns in tf.get_collection('losses')]
print("Losses: {}".format(' '.join(losses)))
while training_counter < TRAINING_STEPS + 1:
    training_counter += 1
    batch_x1 = load1.next_batch(batch_size)
    batch_x2 = load2.next_batch(batch_size)

    feed = {tbn('x1:0'): batch_x1, tbn('x2:0'): batch_x2, tbn('tfis_training:0'): True}
    sess.run(train_op_G, feed_dict=feed)
    sess.run(train_op_D, feed_dict=feed)


    if training_counter % 100 == 0:
        losses = [tns.name[:-2].replace('loss_', '').split('/')[-1] for tns in tf.get_collection('losses')]
        losses_ = sess.run(tf.get_collection('losses'), feed_dict=feed)
        lstring = ' '.join(['{:.3f}'.format(loss) for loss in losses_])
        print("{} ({:.3f} s): {}".format(training_counter, time.time() - t, lstring))
        t = time.time()
##########


############################################################
####################
# GET OUTPUT

output_fake2 = get_layer(sess, tbn('x1:0'), x1, tbn('fake2:0'))
output_fake1 = get_layer(sess, tbn('x2:0'), x2, tbn('fake1:0'))
##########


















