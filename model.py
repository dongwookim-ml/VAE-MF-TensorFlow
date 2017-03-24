from __future__ import division, print_function

import numpy as np
import tensorflow as tf


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.001)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    b_init = tf.constant_initializer(0.)
    return tf.get_variable(name, shape, initializer=b_init)


class VAEMF(object):

    def __init__(self, sess, user_input_dim, item_input_dim,
                 hidden_encoder_dim=216, hidden_decoder_dim=216, latent_dim=24,
                 output_dim=24, learning_rate=0.002, batch_size=64, reg_param=0):
        self.sess = sess
        self.user_input_dim = user_input_dim
        self.item_input_dim = item_input_dim
        self.hidden_encoder_dim = hidden_encoder_dim
        self.hidden_decoder_dim = hidden_decoder_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.reg_param = reg_param
        self.build_model()

    def build_model(self):
        self.l2_loss = tf.constant(0.0)

        self.user = tf.placeholder("float", shape=[None, self.user_input_dim])
        self.item = tf.placeholder("float", shape=[None, self.item_input_dim])
        self.rating = tf.placeholder("float", shape=[None])

        self.W_encoder_input_hidden_user = weight_variable(
            [self.user_input_dim, self.hidden_encoder_dim], 'W_encoder_input_hidden_user')
        self.b_encoder_input_hidden_user = bias_variable(
            [self.hidden_encoder_dim], 'b_encoder_input_hidden_user')
        self.l2_loss += tf.nn.l2_loss(self.W_encoder_input_hidden_user)

        self.W_encoder_input_hidden_item = weight_variable(
            [self.item_input_dim, self.hidden_encoder_dim], 'W_encoder_input_hidden_item')
        self.b_encoder_input_hidden_item = bias_variable(
            [self.hidden_encoder_dim], 'b_encoder_input_hidden_item')
        self.l2_loss += tf.nn.l2_loss(self.W_encoder_input_hidden_item)

        # Hidden layer encoder
        self.hidden_encoder_user = tf.nn.relu(tf.matmul(
            self.user, self.W_encoder_input_hidden_user) + self.b_encoder_input_hidden_user)
        self.hidden_encoder_item = tf.nn.relu(tf.matmul(
            self.item, self.W_encoder_input_hidden_item) + self.b_encoder_input_hidden_item)

        self.W_encoder_hidden_mu_user = weight_variable(
            [self.hidden_encoder_dim, self.latent_dim], 'W_encoder_hidden_mu_user')
        self.b_encoder_hidden_mu_user = bias_variable(
            [self.latent_dim], 'b_encoder_hidden_mu_user')
        self.l2_loss += tf.nn.l2_loss(self.W_encoder_hidden_mu_user)

        self.W_encoder_hidden_mu_item = weight_variable(
            [self.hidden_encoder_dim, self.latent_dim], 'W_encoder_hidden_mu_item')
        self.b_encoder_hidden_mu_item = bias_variable(
            [self.latent_dim], 'b_encoder_hidden_mu_item')
        self.l2_loss += tf.nn.l2_loss(self.W_encoder_hidden_mu_item)

        # Mu encoder
        self.mu_encoder_user = tf.matmul(
            self.hidden_encoder_user, self.W_encoder_hidden_mu_user) + self.b_encoder_hidden_mu_user
        self.mu_encoder_item = tf.matmul(
            self.hidden_encoder_item, self.W_encoder_hidden_mu_item) + self.b_encoder_hidden_mu_item

        self.W_encoder_hidden_logvar_user = weight_variable(
            [self.hidden_encoder_dim, self.latent_dim], 'W_encoder_hidden_logvar_user')
        self.b_encoder_hidden_logvar_user = bias_variable(
            [self.latent_dim], 'b_encoder_hidden_logvar_user')
        self.l2_loss += tf.nn.l2_loss(self.W_encoder_hidden_logvar_user)

        self.W_encoder_hidden_logvar_item = weight_variable(
            [self.hidden_encoder_dim, self.latent_dim], 'W_encoder_hidden_logvar_item')
        self.b_encoder_hidden_logvar_item = bias_variable(
            [self.latent_dim], 'b_encoder_hidden_logvar_item')
        self.l2_loss += tf.nn.l2_loss(self.W_encoder_hidden_logvar_item)

        # Sigma encoder
        self.logvar_encoder_user = tf.matmul(
            self.hidden_encoder_user, self.W_encoder_hidden_logvar_user) + self.b_encoder_hidden_logvar_user
        self.logvar_encoder_item = tf.matmul(
            self.hidden_encoder_item, self.W_encoder_hidden_logvar_item) + self.b_encoder_hidden_logvar_item

        # Sample epsilon
        self.epsilon_user = tf.random_normal(
            tf.shape(self.logvar_encoder_user), name='epsilon_user')
        self.epsilon_item = tf.random_normal(
            tf.shape(self.logvar_encoder_item), name='epsilon_item')

        # Sample latent variable
        self.std_encoder_user = tf.exp(0.5 * self.logvar_encoder_user)
        self.z_user = self.mu_encoder_user + \
            tf.multiply(self.std_encoder_user, self.epsilon_user)

        self.std_encoder_item = tf.exp(0.5 * self.logvar_encoder_item)
        self.z_item = self.mu_encoder_item + \
            tf.multiply(self.std_encoder_item, self.epsilon_item)

        # decoding network
        self.W_decoder_z_hidden_user = weight_variable(
            [self.latent_dim, self.hidden_decoder_dim], 'W_decoder_z_hidden_user')
        self.b_decoder_z_hidden_user = bias_variable(
            [self.hidden_decoder_dim], 'b_decoder_z_hidden_user')
        self.l2_loss += tf.nn.l2_loss(self.W_decoder_z_hidden_user)

        self.W_decoder_z_hidden_item = weight_variable(
            [self.latent_dim, self.hidden_decoder_dim], 'W_decoder_z_hidden_item')
        self.b_decoder_z_hidden_item = bias_variable(
            [self.hidden_decoder_dim], 'b_decoder_z_hidden_item')
        self.l2_loss += tf.nn.l2_loss(self.W_decoder_z_hidden_item)

        # Hidden layer decoder
        self.hidden_decoder_user = tf.nn.relu(tf.matmul(
            self.z_user, self.W_decoder_z_hidden_user) + self.b_decoder_z_hidden_user)
        self.hidden_decoder_item = tf.nn.relu(tf.matmul(
            self.z_item, self.W_decoder_z_hidden_item) + self.b_decoder_z_hidden_item)

        self.W_decoder_hidden_reconstruction_user = weight_variable(
            [self.hidden_decoder_dim, self.output_dim], 'W_decoder_hidden_reconstruction_user')
        self.b_decoder_hidden_reconstruction_user = bias_variable(
            [self.output_dim], 'b_decoder_hidden_reconstruction_user')
        self.l2_loss += tf.nn.l2_loss(
            self.W_decoder_hidden_reconstruction_user)

        self.W_decoder_hidden_reconstruction_item = weight_variable(
            [self.hidden_decoder_dim, self.output_dim], 'W_decoder_hidden_reconstruction_item')
        self.b_decoder_hidden_reconstruction_item = bias_variable(
            [self.output_dim], 'b_decoder_hidden_reconstruction_item')
        self.l2_loss += tf.nn.l2_loss(
            self.W_decoder_hidden_reconstruction_item)

        self.reconstructed_user = tf.matmul(
            self.hidden_decoder_user, self.W_decoder_hidden_reconstruction_user) + self.b_decoder_hidden_reconstruction_user
        self.reconstructed_item = tf.matmul(
            self.hidden_decoder_item, self.W_decoder_hidden_reconstruction_item) + self.b_decoder_hidden_reconstruction_item

        # KL divergence between prior and variational distributions
        self.KLD = -0.5 * tf.reduce_sum(1 + self.logvar_encoder_user - tf.pow(
            self.mu_encoder_user, 2) - tf.exp(self.logvar_encoder_user), reduction_indices=1)
        self.KLD -= 0.5 * tf.reduce_sum(1 + self.logvar_encoder_item - tf.pow(
            self.mu_encoder_item, 2) - tf.exp(self.logvar_encoder_item), reduction_indices=1)

        # rating_hat = tf.diag_part(tf.matmul(z_user, tf.transpose(z_item)))
        self.rating_hat = tf.diag_part(
            tf.matmul(self.reconstructed_user, tf.transpose(self.reconstructed_item)))
        self.MSE = tf.reduce_mean(
            tf.square(tf.subtract(self.rating, self.rating_hat)))
        self.MAE = tf.reduce_mean(
            tf.abs(tf.subtract(self.rating, self.rating_hat)))

        self.loss = tf.reduce_mean(self.KLD + self.MSE)
        self.regularized_loss = self.loss + self.reg_param * self.l2_loss

        self.loss_sum = tf.summary.scalar("mean_squred_error", self.MSE)
        self.train_step = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.regularized_loss)

        # add op for merging summary
        self.summary_op = tf.summary.merge_all()

        # add Saver ops
        self.saver = tf.train.Saver()

    def train(self, M, train_idx=None, test_idx=None, n_steps=100000, result_path='result/'):
        nonzero_user_idx = M.nonzero()[0]
        nonzero_item_idx = M.nonzero()[1]

        if train_idx is None:
            train_idx = np.arange(nonzero_user_idx.size)
        train_size = train_idx.size
        # construct training set
        trainM = np.zeros(M.shape)
        trainM[nonzero_user_idx[train_idx], nonzero_item_idx[train_idx]] = M[
            nonzero_user_idx[train_idx], nonzero_item_idx[train_idx]]

        summary_writer = tf.summary.FileWriter(
            result_path, graph=self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

        for step in range(1, n_steps):
            batch_idx = np.random.randint(train_size, size=self.batch_size)
            user_idx = nonzero_user_idx[train_idx[batch_idx]]
            item_idx = nonzero_item_idx[train_idx[batch_idx]]

            feed_dict = {self.user: trainM[user_idx, :], self.item: trainM[
                :, item_idx].transpose(), self.rating: trainM[user_idx, item_idx]}
            _, mse, mae, summary_str = self.sess.run(
                [self.train_step, self.MSE, self.MAE, self.summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)

            if step % int(n_steps / 10) == 0:

                if test_idx is not None:
                    user_idx = nonzero_user_idx[test_idx]
                    item_idx = nonzero_item_idx[test_idx]

                    feed_dict = {self.user: M[user_idx, :], self.item: M[
                        :, item_idx].transpose(), self.rating: M[user_idx, item_idx]}
                    mse_test, mae_test = self.sess.run(
                        [self.MSE, self.MAE], feed_dict=feed_dict)
                    print("Step {0} | Train MSE: {1:3.4f}, MAE: {2:3.4f}".format(
                        step, mse, mae))
                    print("         | Test  MSE: {0:3.4f}, MAE: {1:3.4f}".format(
                        mse_test, mae_test))
                else:
                    print("Step {0} | Train MSE: {1:3.4f}, MAE: {2:3.4f}".format(
                        step, mse, mae))

        self.saver.save(self.sess, result_path + "/model.ckpt")
