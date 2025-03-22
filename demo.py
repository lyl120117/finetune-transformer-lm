import fire

from opt import adam, warmup_cosine, warmup_linear, warmup_constant
from utils import encode_dataset, flatten, iter_data, find_trainable_variables, convert_gradient_to_tensor, shape_list, ResultLogger, assign_to_gpu, average_grads, make_path
from text_utils import TextEncoder
import os

import tensorflow as tf
import numpy as np
import math
import json

class GPT1:
    def __init__(self, n_vocab, n_ctx, n_embd, n_layer, n_head, n_special, embd_pdrop, clf_pdrop, clf_token, attn_pdrop, resid_pdrop, afn):
        self.n_vocab = n_vocab
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_special = n_special
        self.embd_pdrop = embd_pdrop
        self.clf_pdrop = clf_pdrop
        self.clf_token = clf_token
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.afn = afn

        self.opt_fns = {
            'adam':adam,
        }

        self.act_fns = {
            'relu':tf.nn.relu,
            'swish':self.swish,
            'gelu':self.gelu
        }

    def gelu(self, x):
        return 0.5*x*(1+tf.tanh(math.sqrt(2/math.pi)*(x+0.044715*tf.pow(x, 3))))

    def swish(self, x):
        return x*tf.nn.sigmoid(x)

    def _norm(self, x, g=None, b=None, e=1e-5, axis=[1]):
        u = tf.reduce_mean(x, axis=axis, keep_dims=True)
        s = tf.reduce_mean(tf.square(x-u), axis=axis, keep_dims=True)
        x = (x - u) * tf.rsqrt(s + e)
        if g is not None and b is not None:
            x = x*g + b
        return x

    def norm(self, x, scope, axis=[-1]):
        with tf.variable_scope(scope):
            n_state = shape_list(x)[-1]
            g = tf.get_variable("g", [n_state], initializer=tf.constant_initializer(1))
            b = tf.get_variable("b", [n_state], initializer=tf.constant_initializer(0))
            return self._norm(x, g, b, axis=axis)

    def dropout(self, x, pdrop, train):
        if train and pdrop > 0:
            x = tf.nn.dropout(x, 1-pdrop)
        return x

    def mask_attn_weights(self, w):
        n = shape_list(w)[-1]
        b = tf.matrix_band_part(tf.ones([n, n]), -1, 0)
        b = tf.reshape(b, [1, 1, n, n])
        w = w*b + -1e9*(1-b)
        return w

    def _attn(self, q, k, v, train=False, scale=False):
        w = tf.matmul(q, k)

        if scale:
            n_state = shape_list(v)[-1]
            w = w*tf.rsqrt(tf.cast(n_state, tf.float32))

        w = self.mask_attn_weights(w)
        w = tf.nn.softmax(w)

        w = self.dropout(w, self.attn_pdrop, train)

        a = tf.matmul(w, v)
        return a

    def split_states(self, x, n):
        x_shape = shape_list(x)
        m = x_shape[-1]
        new_x_shape = x_shape[:-1]+[n, m//n]
        return tf.reshape(x, new_x_shape)

    def merge_states(self, x):
        x_shape = shape_list(x)
        new_x_shape = x_shape[:-2]+[np.prod(x_shape[-2:])]
        return tf.reshape(x, new_x_shape)

    def split_heads(self, x, n, k=False):
        if k:
            return tf.transpose(self.split_states(x, n), [0, 2, 3, 1])
        else:
            return tf.transpose(self.split_states(x, n), [0, 2, 1, 3])

    def merge_heads(self, x):
        return self.merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def conv1d(self, x, scope, nf, rf, w_init=tf.random_normal_initializer(stddev=0.02), b_init=tf.constant_initializer(0), pad='VALID', train=False):
        with tf.variable_scope(scope):
            nx = shape_list(x)[-1]
            w = tf.get_variable("w", [rf, nx, nf], initializer=w_init)
            b = tf.get_variable("b", [nf], initializer=b_init)
            if rf == 1: #faster 1x1 conv
                c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf]))+b, shape_list(x)[:-1]+[nf])
            else: #was used to train LM
                c = tf.nn.conv1d(x, w, stride=1, padding=pad)+b
            return c

    def attn(self, x, scope, n_state, n_head, train=False, scale=False):
        assert n_state%n_head==0
        with tf.variable_scope(scope):
            c = self.conv1d(x, 'c_attn', n_state*3, 1, train=train)
            q, k, v = tf.split(c, 3, 2)
            q = self.split_heads(q, n_head)
            k = self.split_heads(k, n_head, k=True)
            v = self.split_heads(v, n_head)
            a = self._attn(q, k, v, train=train, scale=scale)
            a = self.merge_heads(a)
            a = self.conv1d(a, 'c_proj', n_state, 1, train=train)
            a = self.dropout(a, self.resid_pdrop, train)
            return a

    def mlp(self, x, scope, n_state, train=False):
        with tf.variable_scope(scope):
            nx = shape_list(x)[-1]
            act = self.act_fns[self.afn]
            h = act(self.conv1d(x, 'c_fc', n_state, 1, train=train))
            h2 = self.conv1d(h, 'c_proj', nx, 1, train=train)
            h2 = self.dropout(h2, self.resid_pdrop, train)
            return h2

    def block(self, x, scope, train=False, scale=False):
        with tf.variable_scope(scope):
            nx = shape_list(x)[-1]
            a = self.attn(x, 'attn', nx, self.n_head, train=train, scale=scale)
            n = self.norm(x+a, 'ln_1')
            m = self.mlp(n, 'mlp', nx*4, train=train)
            h = self.norm(n+m, 'ln_2')
            return h
        
    def clf(self, x, ny, w_init=tf.random_normal_initializer(stddev=0.02), b_init=tf.constant_initializer(0), train=False):
        with tf.variable_scope('clf'):
            nx = shape_list(x)[-1]
            w = tf.get_variable("w", [nx, ny], initializer=w_init)
            b = tf.get_variable("b", [ny], initializer=b_init)
            return tf.matmul(x, w)+b
    
    def embed(self, X, we):
        we = convert_gradient_to_tensor(we)
        e = tf.gather(we, X)
        h = tf.reduce_sum(e, 2)
        return h

    def model(self, X, M, Y, train=False, reuse=False):
        with tf.variable_scope('model', reuse=reuse):
            we = tf.get_variable("we", [self.n_vocab+self.n_special+self.n_ctx, self.n_embd], initializer=tf.random_normal_initializer(stddev=0.02))
            we = tf.Print(we, [tf.shape(we)], message='### model we shape:', summarize=100)
            we = self.dropout(we, self.embd_pdrop, train)

            X = tf.reshape(X, [-1, self.n_ctx, 2])
            X = tf.Print(X, [tf.shape(X)], message='### model X shape:', summarize=100)
            M = tf.reshape(M, [-1, self.n_ctx])
            M = tf.Print(M, [tf.shape(M)], message='### model M shape:', summarize=100)

            h = self.embed(X, we)
            h = tf.Print(h, [tf.shape(h)], message='### model embed h shape:', summarize=100)
            for layer in range(self.n_layer):
                h = self.block(h, 'h%d'%layer, train=train, scale=True)
            
            h = tf.Print(h, [tf.shape(h)], message='### model block h shape:', summarize=100)

            lm_h = tf.reshape(h[:, :-1], [-1, self.n_embd])
            lm_h = tf.Print(lm_h, [tf.shape(lm_h)], message='### model lm_h shape:', summarize=100)
            lm_logits = tf.matmul(lm_h, we, transpose_b=True)
            lm_logits = tf.Print(lm_logits, [tf.shape(lm_logits)], message='### model lm_logits shape:', summarize=100)
            lm_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=lm_logits, labels=tf.reshape(X[:, 1:, 0], [-1]))
            lm_losses = tf.reshape(lm_losses, [shape_list(X)[0], shape_list(X)[1]-1])
            lm_losses = tf.reduce_sum(lm_losses*M[:, 1:], 1)/tf.reduce_sum(M[:, 1:], 1)

            clf_h = tf.reshape(h, [-1, self.n_embd])
            clf_h = tf.Print(clf_h, [tf.shape(clf_h)], message='### model clf_h shape:', summarize=100)
            pool_idx = tf.cast(tf.argmax(tf.cast(tf.equal(X[:, :, 0], self.clf_token), tf.float32), 1), tf.int32)
            pool_idx = tf.Print(pool_idx, [tf.shape(pool_idx)], message='### model pool_idx shape:', summarize=100)
            pool_idx = tf.Print(pool_idx, [pool_idx], message='### model pool_idx:', summarize=100)
            clf_h = tf.gather(clf_h, tf.range(shape_list(X)[0], dtype=tf.int32)*self.n_ctx+pool_idx)
            clf_h = tf.Print(clf_h, [tf.shape(clf_h)], message='### model gather clf_h shape:', summarize=100)

            clf_h = tf.reshape(clf_h, [-1, 2, self.n_embd])
            clf_h = tf.Print(clf_h, [tf.shape(clf_h)], message='### model reshape clf_h shape:', summarize=100)
            if train and self.clf_pdrop > 0:
                shape = shape_list(clf_h)
                shape[1] = 1
                clf_h = tf.nn.dropout(clf_h, 1-self.clf_pdrop, shape)
            clf_h = tf.reshape(clf_h, [-1, self.n_embd])
            clf_logits = self.clf(clf_h, 1, train=train)
            clf_logits = tf.reshape(clf_logits, [-1, 2])

            clf_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=clf_logits, labels=Y)
            return clf_logits, clf_losses, lm_losses


def main(model_path='model/', n_ctx=512, n_embd=768, n_layer=12, n_head=12, embd_pdrop=0.1, clf_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1, afn='gelu', n_transfer=12, debug=False):
    print('Loading model from:', model_path)

    encoder_path = os.path.join(model_path, 'encoder_bpe_40000.json')
    vocab_path = os.path.join(model_path, 'vocab_40000.bpe')
    # Load the text encoder
    text_encoder = TextEncoder(encoder_path, vocab_path)

    raw_text = 'Hello, how are you?'
    # raw_text = 'Hello?'
    print('Raw text:', raw_text, ', len:', len(raw_text))
    embdded_text = text_encoder.encode(raw_text)
    print('Encoded text:', embdded_text)
    print('Encoded text len:', len(embdded_text))

    n_special = 3
    if not debug:
        n_vocab = len(text_encoder.encoder)
        print('Vocabulary size:', n_vocab)

        encoder = text_encoder.encoder
        encoder['_classify_'] = len(encoder)
        clf_token = encoder['_classify_']
        print('Classifier token:', clf_token)
    else:
        n_vocab = 6
        clf_token = n_vocab
        print('Classifier token:', clf_token)
        print('Vocabulary size:', n_vocab)

    X = tf.placeholder(tf.int32, [None, 2, n_ctx, 2])
    M = tf.placeholder(tf.float32, [None, 2, n_ctx])
    Y = tf.placeholder(tf.int32, [None])

    input_x = np.random.randint(0, n_vocab, (1, 2, n_ctx, 2)).astype(np.int32)
    input_m = np.random.randint(0, 2, (1, 2, n_ctx)).astype(np.float32)
    input_y = np.random.randint(0, 2, (1)).astype(np.int32)
    print('Input x:', input_x.shape)
    print('Input m:', input_m.shape)
    print('Input y:', input_y.shape)

    # Initialize the model
    gpt = GPT1(n_vocab, n_ctx, n_embd, n_layer, n_head, n_special, embd_pdrop, clf_pdrop, clf_token, attn_pdrop, resid_pdrop, afn)
    print('Model initialized')

    if not debug:
        params_shapes_path = os.path.join(model_path, 'params_shapes.json')
        shapes = json.load(open(params_shapes_path))
        print('Parameters shapes loaded from:', params_shapes_path)
        offsets = np.cumsum([np.prod(shape) for shape in shapes])
        init_params = [np.load(os.path.join(model_path, 'params_{}.npy'.format(n))) for n in range(10)]
        init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
        init_params = [param.reshape(shape) for param, shape in zip(init_params, shapes)]
        init_params[0] = init_params[0][:n_ctx]
        init_params[0] = np.concatenate([init_params[1], (np.random.randn(n_special, n_embd)*0.02).astype(np.float32), init_params[0]], 0)
        del init_params[1]
        if n_transfer == -1:
            n_transfer = 0
        else:
            n_transfer = 1+n_transfer*12
        
        print('Transfer parameters:', n_transfer)
        print('init_params:', len(init_params))

    clf_logits, clf_losses, lm_losses = gpt.model(X, M, Y, train=False, reuse=tf.AUTO_REUSE)

    params = find_trainable_variables('model')
    print('Trainable parameters shapes:')
    for p in params:
        print(p.name, p.get_shape().as_list())
    print('Trainable parameters:', len(params))

    init = tf.global_variables_initializer()

    # Initialize the session
    with tf.Session() as sess:
        # Initialize variables
        sess.run(init)
        print('Variables initialized')

        if not debug:
            sess.run([p.assign(ip) for p, ip in zip(params[:n_transfer], init_params[:n_transfer])])
            print('Parameters loaded')
        
        # Run the model
        clf_logits_out, clf_losses_out, lm_losses_out = sess.run([clf_logits, clf_losses, lm_losses], feed_dict={X: input_x, M: input_m, Y: input_y})
        print('Classifier logits:', clf_logits_out)
        print('Classifier losses:', clf_losses_out)
        print('LM losses:', lm_losses_out)

# python demo.py --debug true --n_ctx=4 --n_embd=4 --n_layer=1 --n_head=2 
if __name__ == '__main__':
    fire.Fire(main)