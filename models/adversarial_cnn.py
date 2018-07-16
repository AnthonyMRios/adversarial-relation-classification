from theano import tensor as T
from theano.ifelse import ifelse
from sklearn import cross_validation
from sklearn.utils import shuffle
import theano
from theano import config
import numpy as np
#from theano.tensor.nnet.conv import conv2d
from theano.tensor.nnet import conv2d
from theano.tensor.signal.pool import pool_2d as max_pool_2d
from theano.tensor.shared_randomstreams import RandomStreams
srng2 = RandomStreams(seed=234)

from utils import *

class CNN(object):

    def __init__(self, emb, pos, nc=2, de=100, disc_h=250, fs=[3,4,5],
                 nf=300, emb_reg=False, pos_reg=False, longhist=True):
        '''
            emb :: Embedding Matrix
            nh :: hidden layer size
            nc :: Number of classes
            de :: Dimensionality of word embeddings
            p_drop :: Dropout probability
        '''
        # Source Embeddings
        self.emb = theano.shared(name='Words',
            value=emb.astype('float32'))
        self.target_emb = theano.shared(name='Wordst',
            value=emb.astype('float32'))
        self.avg_emb = theano.shared(name='Wordsa',
            value=emb.astype('float32'))

        self.pos = theano.shared(name='Pos',
            value=pos.astype('float32'))
        self.target_pos = theano.shared(name='Post',
            value=pos.astype('float32'))
        self.avg_pos = theano.shared(name='Posta',
            value=pos.astype('float32'))
        # Targt Embeddings
        # Source Output Weights
        self.w_o = theano.shared(name='w_o',
                value=he_normal((nf*len(fs), 1))
                .astype('float32'))
        self.b_o = theano.shared(name='b_o',
            value=np.zeros((1,)).astype('float32'))

        # Discriminator Weights
        self.w_h_1 = theano.shared(name='w_h_1',
                value=he_normal((nf*len(fs), disc_h))
                .astype('float32'))
        self.b_h_1 = theano.shared(name='b_h_1',
            value=np.zeros((disc_h,)).astype('float32'))
        self.w_h_2 = theano.shared(name='w_h_2',
                value=he_normal((disc_h,disc_h))
                .astype('float32'))
        self.b_h_2 = theano.shared(name='b_h_2',
            value=np.zeros((disc_h,)).astype('float32'))
        self.w_adv = theano.shared(name='w_adv',
                value=he_normal((disc_h,1))
                .astype('float32'))
        self.b_adv = theano.shared(name='b_adv',
            value=np.zeros((1,)).astype('float32'))

        # Update these parameters
        self.params_source = [self.w_o, self.b_o, self.emb, self.pos]
        self.params_target = [self.target_emb, self.target_pos]

        #self.params_discriminator = [self.w_h_1, self.b_h_1, self.w_h_3, self.b_h_3,
        self.params_discriminator = [self.w_h_1, self.b_h_1,
            self.w_h_2, self.b_h_2, self.w_adv, self.b_adv]

        source_idxs = T.matrix()
        target_idxs = T.matrix()
        source_e1_pos_idxs = T.matrix()
        source_e2_pos_idxs = T.matrix()
        target_e1_pos_idxs = T.matrix()
        target_e2_pos_idxs = T.matrix()
        source_Y = T.ivector()

        # get word embeddings based on indicies
        source_x_word = self.emb[T.cast(source_idxs, 'int32')]
        source_x_e1_pos = self.pos[T.cast(source_e1_pos_idxs, 'int32')]
        source_x_e2_pos = self.pos[T.cast(source_e2_pos_idxs, 'int32')]
        source_x_word = T.concatenate([source_x_word, source_x_e1_pos,
            source_x_e2_pos], axis=2)
        mask = T.neq(source_idxs, 0)*1
        source_x_word = source_x_word*mask.dimshuffle(0, 1, 'x')
        source_x_word = source_x_word.reshape((source_x_word.shape[0], 1,
                                               source_x_word.shape[1],
                                               source_x_word.shape[2]))

        target_x_word = self.target_emb[T.cast(target_idxs, 'int32')]
        target_x_e1_pos = self.target_pos[T.cast(target_e1_pos_idxs, 'int32')]
        target_x_e2_pos = self.target_pos[T.cast(target_e2_pos_idxs, 'int32')]
        target_x_word = T.concatenate([target_x_word, target_x_e1_pos,
            target_x_e2_pos], axis=2)
        mask2 = T.neq(target_idxs, 0)*1
        target_x_word = target_x_word*mask2.dimshuffle(0, 1, 'x')
        target_x_word = target_x_word.reshape((target_x_word.shape[0], 1,
                                               target_x_word.shape[1],
                                               target_x_word.shape[2]))

        de = de + 2*pos.shape[1]

        source_cnn_w, source_cnn_b = cnn_weights(de, fs, nf)
        target_cnn_w, target_cnn_b = cnn_weights(de, fs, nf)
        avg_cnn_w, avg_cnn_b = cnn_weights(de, fs, nf)

        self.params_source += source_cnn_w + source_cnn_b
        self.params_target += target_cnn_w + target_cnn_b
        self.params_avg = avg_cnn_w + avg_cnn_b

        dropout_switch = T.scalar()
        real_attention = T.scalar()

        source_l1_w_all = []
        for w, b, width in zip(source_cnn_w, source_cnn_b, fs):
            l1_w = conv2d(source_x_word, w, image_shape=(None,1,None,de), filter_shape=(nf, 1, width, de))
            l1_w = rectify(l1_w+ b.dimshuffle('x', 0, 'x', 'x'))
            l1_w = T.max(l1_w, axis=2).flatten(2)
            source_l1_w_all.append(l1_w)

        target_l1_w_all = []
        for w, b, width in zip(target_cnn_w, target_cnn_b, fs):
            l1_w = conv2d(target_x_word, w, image_shape=(None,1,None,de), filter_shape=(nf, 1, width, de))
            l1_w = rectify(l1_w+ b.dimshuffle('x', 0, 'x', 'x'))
            l1_w = T.max(l1_w, axis=2).flatten(2)
            target_l1_w_all.append(l1_w)

        source_h = T.concatenate(source_l1_w_all, axis=1)
        source_h = dropout(source_h, dropout_switch, 0.5)

        target_h = T.concatenate(target_l1_w_all, axis=1)
        target_h = dropout(target_h, dropout_switch, 0.5)
    
        pyx_source = T.nnet.nnet.sigmoid(T.dot(source_h, self.w_o) + self.b_o.dimshuffle('x', 0))
        pyx_source = T.clip(pyx_source, 1e-5, 1-1e-5)

        self.step = theano.shared(np.float32(0))

        source_h2 = rectify(T.dot(source_h, self.w_h_1) + self.b_h_1)
        source_h2 = dropout(source_h2, dropout_switch, 0.5)
        source_h2 = rectify(T.dot(source_h2, self.w_h_2) + self.b_h_2)
        source_h2 = dropout(source_h2, dropout_switch, 0.5)

        target_h2 = rectify(T.dot(target_h, self.w_h_1) + self.b_h_1)
        target_h2 = dropout(target_h2, dropout_switch, 0.5)
        target_h2 = rectify(T.dot(target_h2, self.w_h_2) + self.b_h_2)
        target_h2 = dropout(target_h2, dropout_switch, 0.5)

        pyx_adv_source = T.nnet.nnet.sigmoid(T.dot(source_h2, self.w_adv) + self.b_adv.dimshuffle('x',0))
        pyx_adv_source = T.clip(pyx_adv_source, 1e-5, 1-1e-5)

        pyx_adv_target = T.nnet.nnet.sigmoid(T.dot(target_h2, self.w_adv) + self.b_adv.dimshuffle('x',0))
        pyx_adv_target = T.clip(pyx_adv_target, 1e-5, 1-1e-5)

        pyx_test= T.nnet.nnet.sigmoid(T.dot(target_h, self.w_o) + self.b_o.dimshuffle('x', 0))

        # Generator Loss
        #L_adv_generator = -.9*T.log(pyx_adv_target).mean() - .1*T.log(1.-pyx_adv_target).mean()
        L_adv_generator = -T.log(pyx_adv_target).mean()
        num_updates = theano.shared(as_floatX(1.).astype("float32"))
        if emb_reg:
            L_adv_generator += .5*((self.avg_emb - self.target_emb)**2).sum()
        if pos_reg:
            L_adv_generator += .5*((self.avg_pos - self.target_pos)**2).sum()
        if True:
            L_adv_generator += .5*sum([((s - t)**2).sum() for s,t in zip(self.params_avg, self.params_target[2:])])

        updates_generator, _ = Adam(L_adv_generator, self.params_target, lr2=0.0002)
        if not longhist:
            updates_generator.append((self.avg_emb, 0.9*self.avg_emb + 0.1*self.target_emb))
            updates_generator.append((self.avg_pos, 0.9*self.avg_pos + 0.1*self.target_pos))
            for p, t in zip(self.params_avg, self.params_target[2:]):
                updates_generator.append((p, 0.9*p + 0.1*t))
        else:
            updates_generator.append((self.avg_emb, self.avg_emb + self.target_emb))
            updates_generator.append((self.avg_pos, self.avg_pos + self.target_pos))
            updates_generator.append((num_updates, num_updates + 1.))

        self.train_batch_generator = theano.function([target_idxs, target_e1_pos_idxs, target_e2_pos_idxs,\
            dropout_switch],
            L_adv_generator, updates=updates_generator, allow_input_downcast=True, on_unused_input='ignore')

        L_adv_discriminator = -T.log(1-pyx_adv_target).mean() - ((srng2.uniform(low=0.7, high=1., size=pyx_adv_target.shape))*T.log(pyx_adv_source)).mean()
        #L_adv_discriminator = -0.9*T.log(1.-pyx_adv_target).mean() - .9*T.log(pyx_adv_source).mean() - .1*T.log(pyx_adv_target).mean() - .1*T.log(1.-pyx_adv_source).mean()
        #L_adv_discriminator = -T.log(1.-pyx_adv_target).mean() - T.log(pyx_adv_source).mean()
        #L_adv_discriminator += 1e-2*sum([(x**2).sum() for x in self.params_discriminator])

        updates_discriminator, self.disc_lr = Adam(L_adv_discriminator, self.params_discriminator, lr2=0.0002)

        #L_source = T.nnet.binary_crossentropy(pyx_source.flatten(), source_Y).mean() + 1e-4 * sum([(x**2).sum() for x in self.params_source])
        L_source = T.nnet.binary_crossentropy(pyx_source.flatten(), source_Y).mean() + 1e-4 * sum([(x**2).sum() for x in self.params_source])
        updates_source, _ = Adam(L_source, self.params_source, lr2=0.001)

        self.train_batch_source = theano.function([source_idxs, source_e1_pos_idxs, source_e2_pos_idxs, source_Y,\
            dropout_switch],
             L_source, updates=updates_source, allow_input_downcast=True, on_unused_input='ignore')

        self.train_batch_discriminator = theano.function([target_idxs, source_idxs, target_e1_pos_idxs, target_e2_pos_idxs,\
            source_e1_pos_idxs, source_e2_pos_idxs, dropout_switch],
            L_adv_discriminator, updates=updates_discriminator, allow_input_downcast=True, on_unused_input='ignore')

        self.features = theano.function([target_idxs, target_e1_pos_idxs, target_e2_pos_idxs, dropout_switch],\
                target_h, allow_input_downcast=True, on_unused_input='ignore')

        self.predict_proba = theano.function([target_idxs, target_e1_pos_idxs, target_e2_pos_idxs, dropout_switch],\
                pyx_test.flatten(), allow_input_downcast=True, on_unused_input='ignore')
        self.predict_src_proba = theano.function([source_idxs, source_e1_pos_idxs, source_e2_pos_idxs, dropout_switch],\
                pyx_source.flatten(), allow_input_downcast=True, on_unused_input='ignore')

    def __getstate__(self):
        values = [x.get_value() for x in self.params_source]
        return values

    def __getemb__(self):
        return self.target_emb.get_value()

    def __setstate__(self, weights):
        for x,w in zip(self.params_source, weights):
            x.set_value(w) 

    def __settarget__(self):
        for ps, pt in zip(self.params_source[2:], self.params_target):
            pt.set_value(ps.get_value())
        self.avg_emb.set_value(self.params_source[2].get_value())
        self.avg_pos.set_value(self.params_source[3].get_value())
        for ps, pt in zip(self.params_source[4:], self.params_avg):
            pt.set_value(ps.get_value())
        return

def cnn_weights(de, fs, nf):
    filter_w = []
    filter_b = []
    for filter_size in fs:
        filter_w.append(theano.shared(
            value=he_normal((nf, 1, filter_size, de))
            .astype('float32')))
        filter_b.append(theano.shared(
            value=np.zeros((nf,)).astype('float32')))
    return filter_w, filter_b
