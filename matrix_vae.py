from __future__ import division, print_function

import numpy as np
import tensorflow as tf


def weight_variable(shape, name):
    #xavier initialisation
    in_dim = shape[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.Variable(tf.random_normal(shape=size, stddev=xavier_stddev), name=name)


def bias_variable(shape, name):
    b_init = tf.constant_initializer(0.)
    return tf.get_variable(name, shape, initializer=b_init)


class VAEMF(object):

    def __init__(self, sess, num_user, num_item,
                 hidden_encoder_dim=216, hidden_decoder_dim=216, latent_dim=24,
                 learning_rate=0.002, batch_size=64, reg_param=0,
                 user_embed_dim=216, item_embed_dim=216, activate_fn=tf.tanh, vae=True):

        if reg_param < 0 or reg_param > 1:
            raise ValueError("regularization parameter must be in [0,1]")

        self.sess = sess
        self.num_user = num_user
        self.num_item = num_item
        self.hidden_encoder_dim = hidden_encoder_dim
        self.hidden_decoder_dim = hidden_decoder_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.reg_param = reg_param
        self.user_embed_dim = user_embed_dim
        self.item_embed_dim = item_embed_dim
        self.activate_fn = activate_fn
        self.vae = vae
        self.build_model()

    def build_model(self):
        self.l2_loss = tf.constant(0.0)

        self.user = tf.placeholder("float", shape=[None, self.num_item])
        self.valid_rating = tf.placeholder("float", shape=[None, self.num_item])
        self.test_rating = tf.placeholder("float", shape=[None, self.num_item])

        self.W_encoder_input_hidden_user = weight_variable(
            [self.num_item, self.hidden_encoder_dim], 'W_encoder_input_hidden_user')
        self.b_encoder_input_hidden_user = bias_variable(
            [self.hidden_encoder_dim], 'b_encoder_input_hidden_user')
        self.l2_loss += tf.nn.l2_loss(self.W_encoder_input_hidden_user)

        # Hidden layer encoder
        self.hidden_encoder_user = self.activate_fn(tf.matmul(
            self.user, self.W_encoder_input_hidden_user) + self.b_encoder_input_hidden_user)

        self.W_encoder_hidden_mu_user = weight_variable(
            [self.hidden_encoder_dim, self.latent_dim], 'W_encoder_hidden_mu_user')
        self.b_encoder_hidden_mu_user = bias_variable(
            [self.latent_dim], 'b_encoder_hidden_mu_user')
        self.l2_loss += tf.nn.l2_loss(self.W_encoder_hidden_mu_user)

        # Mu encoder
        self.mu_encoder_user = tf.matmul(
            self.hidden_encoder_user, self.W_encoder_hidden_mu_user) + self.b_encoder_hidden_mu_user

        self.W_encoder_hidden_logvar_user = weight_variable(
            [self.hidden_encoder_dim, self.latent_dim], 'W_encoder_hidden_logvar_user')
        self.b_encoder_hidden_logvar_user = bias_variable(
            [self.latent_dim], 'b_encoder_hidden_logvar_user')
        self.l2_loss += tf.nn.l2_loss(self.W_encoder_hidden_logvar_user)

        if self.vae:
            # Sigma encoder
            self.logvar_encoder_user = tf.matmul(
                self.hidden_encoder_user, self.W_encoder_hidden_logvar_user) + self.b_encoder_hidden_logvar_user

            # Sample epsilon
            self.epsilon_user = tf.random_normal(
                tf.shape(self.logvar_encoder_user), name='epsilon_user')

            # Sample latent variable
            self.std_encoder_user = tf.exp(0.5 * self.logvar_encoder_user)
            self.z_user = self.mu_encoder_user + \
                tf.multiply(self.std_encoder_user, self.epsilon_user)
        else:
            self.z_user = self.mu_encoder_user

        # decoding network
        self.W_decoder_z_hidden_user = weight_variable(
            [self.latent_dim, self.hidden_decoder_dim], 'W_decoder_z_hidden_user')
        self.b_decoder_z_hidden_user = bias_variable(
            [self.hidden_decoder_dim], 'b_decoder_z_hidden_user')
        self.l2_loss += tf.nn.l2_loss(self.W_decoder_z_hidden_user)

        # Hidden layer decoder
        self.hidden_decoder_user = self.activate_fn(tf.matmul(
            self.z_user, self.W_decoder_z_hidden_user) + self.b_decoder_z_hidden_user)

        self.W_decoder_hidden_reconstruction_user = weight_variable(
            [self.hidden_decoder_dim, self.num_item], 'W_decoder_hidden_reconstruction_user')
        self.b_decoder_hidden_reconstruction_user = bias_variable(
            [self.num_item], 'b_decoder_hidden_reconstruction_user')
        self.l2_loss += tf.nn.l2_loss(
            self.W_decoder_hidden_reconstruction_user)

        self.reconstructed_user = tf.matmul(
            self.hidden_decoder_user, self.W_decoder_hidden_reconstruction_user) + self.b_decoder_hidden_reconstruction_user

        weight = tf.not_equal(self.user, tf.constant(0, dtype=tf.float32))

        self.MSE = tf.losses.mean_squared_error(self.user, self.reconstructed_user, weight)
        self.MAE = tf.losses.absolute_difference(self.user, self.reconstructed_user, weight)

        if self.vae:
            # KL divergence between prior and variational distributions
            self.KLD = -0.5 * tf.reduce_sum(1 + self.logvar_encoder_user - tf.pow(
                self.mu_encoder_user, 2) - tf.exp(self.logvar_encoder_user), reduction_indices=1)
            self.loss = tf.reduce_mean(self.KLD + self.MSE)
        else:
            self.loss = tf.reduce_mean(self.MSE)

        self.regularized_loss = self.loss + self.reg_param * self.l2_loss

        valid_weight = tf.not_equal(self.valid_rating, tf.constant(0, dtype=tf.float32))
        test_weight = tf.not_equal(self.test_rating, tf.constant(0, dtype=tf.float32))
        self.valid_RMSE = tf.sqrt(tf.losses.mean_squared_error(self.valid_rating, self.reconstructed_user, valid_weight))
        self.test_RMSE = tf.sqrt(tf.losses.mean_squared_error(self.test_rating, self.reconstructed_user, test_weight))

        tf.summary.scalar("MSE", self.MSE)
        tf.summary.scalar("MAE", self.MAE)
        tf.summary.scalar("Loss", self.loss)
        tf.summary.scalar("valid-RMSE", self.valid_RMSE)
        tf.summary.scalar("test-RMSE", self.test_RMSE)
        tf.summary.scalar("Reg-Loss", self.regularized_loss)

        self.train_step = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.regularized_loss)

        # add op for merging summary
        self.summary_op = tf.summary.merge_all()

        # add Saver ops
        self.saver = tf.train.Saver()

    def train_test_validation(self, M, train_idx, test_idx, valid_idx, n_steps=100000, result_path='result/'):
        nonzero_user_idx = M.nonzero()[0]
        nonzero_item_idx = M.nonzero()[1]

        trainM = np.zeros(M.shape)
        trainM[nonzero_user_idx[train_idx], nonzero_item_idx[train_idx]] = M[nonzero_user_idx[train_idx], nonzero_item_idx[train_idx]]

        validM = np.zeros(M.shape)
        validM[nonzero_user_idx[valid_idx], nonzero_item_idx[valid_idx]] = M[nonzero_user_idx[valid_idx], nonzero_item_idx[valid_idx]]

        testM = np.zeros(M.shape)
        testM[nonzero_user_idx[test_idx], nonzero_item_idx[test_idx]] = M[nonzero_user_idx[test_idx], nonzero_item_idx[test_idx]]

        for i in range(self.num_user):
            if np.sum(trainM[i]) == 0:
                testM[i] = 0
                validM[i] = 0

        train_writer = tf.summary.FileWriter(
            result_path + '/train', graph=self.sess.graph)

        best_val_rmse = np.inf
        best_test_rmse = 0

        self.sess.run(tf.global_variables_initializer())
        for step in range(1, n_steps):
            feed_dict = {self.user: trainM, self.valid_rating:validM, self.test_rating:testM}

            _, mse, mae, valid_rmse, test_rmse,  summary_str = self.sess.run(
                [self.train_step, self.MSE, self.MAE, self.valid_RMSE, self.test_RMSE, self.summary_op], feed_dict=feed_dict)
            train_writer.add_summary(summary_str, step)
            print("Iter {0} Train RMSE:{1}, Valid RMSE:{2}, Test RMSE:{3}".format(step, np.sqrt(mse), valid_rmse, test_rmse))

            if best_val_rmse > valid_rmse:
                best_val_rmse = valid_rmse
                best_test_rmse = test_rmse

        self.saver.save(self.sess, result_path + "/model.ckpt")
        return best_test_rmse
