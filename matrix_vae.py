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

    def __init__(self, sess, num_user, num_item,
                 hidden_encoder_dim=216, hidden_decoder_dim=216, latent_dim=24,
                 output_dim=24, learning_rate=0.002, batch_size=64, reg_param=0,
                 user_embed_dim=216, item_embed_dim=216):

        if reg_param < 0 or reg_param > 1:
            raise ValueError("regularization parameter must be in [0,1]")

        self.sess = sess
        self.num_user = num_user
        self.num_item = num_item
        self.hidden_encoder_dim = hidden_encoder_dim
        self.hidden_decoder_dim = hidden_decoder_dim
        self.latent_dim = latent_dim
        self.output_dim = num_item
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.reg_param = reg_param
        self.user_embed_dim = user_embed_dim
        self.item_embed_dim = item_embed_dim
        self.user_input_dim = num_item
        self.build_model()

    def build_model(self):
        self.l2_loss = tf.constant(0.0)

        self.user = tf.placeholder("float", shape=[None, self.user_input_dim])
        self.rating = tf.placeholder("float", shape=[None, self.output_dim])

        self.W_encoder_input_hidden_user = weight_variable(
            [self.user_input_dim, self.hidden_encoder_dim], 'W_encoder_input_hidden_user')
        self.b_encoder_input_hidden_user = bias_variable(
            [self.hidden_encoder_dim], 'b_encoder_input_hidden_user')
        self.l2_loss += tf.nn.l2_loss(self.W_encoder_input_hidden_user)

        # Hidden layer encoder
        self.hidden_encoder_user = tf.nn.relu(tf.matmul(
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

        # decoding network
        self.W_decoder_z_hidden_user = weight_variable(
            [self.latent_dim, self.hidden_decoder_dim], 'W_decoder_z_hidden_user')
        self.b_decoder_z_hidden_user = bias_variable(
            [self.hidden_decoder_dim], 'b_decoder_z_hidden_user')
        self.l2_loss += tf.nn.l2_loss(self.W_decoder_z_hidden_user)

        # Hidden layer decoder
        self.hidden_decoder_user = tf.nn.relu(tf.matmul(
            self.z_user, self.W_decoder_z_hidden_user) + self.b_decoder_z_hidden_user)

        self.W_decoder_hidden_reconstruction_user = weight_variable(
            [self.hidden_decoder_dim, self.output_dim], 'W_decoder_hidden_reconstruction_user')
        self.b_decoder_hidden_reconstruction_user = bias_variable(
            [self.output_dim], 'b_decoder_hidden_reconstruction_user')
        self.l2_loss += tf.nn.l2_loss(
            self.W_decoder_hidden_reconstruction_user)

        self.reconstructed_rating = tf.matmul(
            self.hidden_decoder_user, self.W_decoder_hidden_reconstruction_user) + self.b_decoder_hidden_reconstruction_user

        # KL divergence between prior and variational distributions
        self.KLD = -0.5 * tf.reduce_sum(1 + self.logvar_encoder_user - tf.pow(
            self.mu_encoder_user, 2) - tf.exp(self.logvar_encoder_user), reduction_indices=1)

        zero = tf.constant(0, dtype=tf.float32)
        weight = tf.not_equal(self.rating, zero)

        self.MSE = tf.losses.mean_squared_error(self.rating, self.reconstructed_rating, weight)
        self.MAE = tf.losses.absolute_difference(self.rating, self.reconstructed_rating, weight)

        self.loss = tf.reduce_mean(self.KLD + self.MSE)
        self.regularized_loss = self.loss + self.reg_param * self.l2_loss

        tf.summary.scalar("MSE", self.MSE)
        tf.summary.scalar("MAE", self.MAE)
        tf.summary.scalar("Loss", self.loss)
        tf.summary.scalar("Reg-Loss", self.regularized_loss)

        self.train_step = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.regularized_loss)

        # add op for merging summary
        self.summary_op = tf.summary.merge_all()

        # add Saver ops
        self.saver = tf.train.Saver()

    def construct_feeddict(self, user_idx, M):
        feed_dict = {self.user: M[user_idx, :], self.rating: M[user_idx, :]}
        return feed_dict

    def train_test_validation(self, M, train_idx, test_idx, valid_idx, n_steps=100000, result_path='result/'):
        nonzero_user_idx = M.nonzero()[0]
        nonzero_item_idx = M.nonzero()[1]

        train_size = train_idx.size
        trainM = np.zeros(M.shape)
        trainM[nonzero_user_idx[train_idx], nonzero_item_idx[train_idx]] = M[
            nonzero_user_idx[train_idx], nonzero_item_idx[train_idx]]

        validM = np.zeros(M.shape)
        validM[nonzero_user_idx[valid_idx], nonzero_item_idx[valid_idx]] = M[
            nonzero_user_idx[valid_idx], nonzero_item_idx[valid_idx]]

        testM = np.zeros(M.shape)
        testM[nonzero_user_idx[test_idx], nonzero_item_idx[test_idx]] = M[
            nonzero_user_idx[test_idx], nonzero_item_idx[test_idx]]

        train_writer = tf.summary.FileWriter(
            result_path + '/train', graph=self.sess.graph)
        valid_writer = tf.summary.FileWriter(
            result_path + '/validation', graph=self.sess.graph)
        test_writer = tf.summary.FileWriter(
            result_path + '/test', graph=self.sess.graph)

        best_val_mse = np.inf
        best_val_mae = np.inf
        best_test_mse = 0
        best_test_mae = 0

        self.sess.run(tf.global_variables_initializer())

        for step in range(1, n_steps):
            batch_idx = np.random.randint(train_size, size=self.batch_size)
            user_idx = nonzero_user_idx[train_idx[batch_idx]]
            feed_dict = self.construct_feeddict(user_idx, trainM)

            _, mse, mae, summary_str = self.sess.run(
                [self.train_step, self.MSE, self.MAE, self.summary_op], feed_dict=feed_dict)
            train_writer.add_summary(summary_str, step)

            if step % int(n_steps / 10) == 0:
                valid_user_idx = nonzero_user_idx[valid_idx]
                feed_dict = self.construct_feeddict(
                    valid_user_idx, validM)
                mse_valid, mae_valid, summary_str = self.sess.run(
                    [self.MSE, self.MAE, self.summary_op], feed_dict=feed_dict)

                valid_writer.add_summary(summary_str, step)

                test_user_idx = nonzero_user_idx[test_idx]
                feed_dict = self.construct_feeddict(
                    test_user_idx, testM)
                mse_test, mae_test, summary_str = self.sess.run(
                    [self.MSE, self.MAE, self.summary_op], feed_dict=feed_dict)

                test_writer.add_summary(summary_str, step)

                print("Step {0} | Train MSE: {1:3.4f}, MAE: {2:3.4f}".format(
                    step, mse, mae))
                print("         | Valid  MSE: {0:3.4f}, MAE: {1:3.4f}".format(
                    mse_valid, mae_valid))
                print("         | Test  MSE: {0:3.4f}, MAE: {1:3.4f}".format(
                    mse_test, mae_test))

                if best_val_mse > mse_valid:
                    best_val_mse = mse_valid
                    best_test_mse = mse_test

                if best_val_mae > mae_valid:
                    best_val_mae = mae_valid
                    best_test_mae = mae_test

        self.saver.save(self.sess, result_path + "/model.ckpt")
        return best_test_mse, best_test_mae

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

        testM = M - trainM

        train_writer = tf.summary.FileWriter(
            result_path + '/train', graph=self.sess.graph)
        test_writer = tf.summary.FileWriter(
            result_path + '/test', graph=self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

        for step in range(1, n_steps):
            user_idx = np.random.randint(self.num_user, size=self.batch_size)
            feed_dict = self.construct_feeddict(user_idx, trainM)

            _, mse, mae, summary_str = self.sess.run(
                [self.train_step, self.MSE, self.MAE, self.summary_op], feed_dict=feed_dict)
            train_writer.add_summary(summary_str, step)

            if step % int(n_steps / 100) == 0:

                if test_idx is not None:
                    user_idx = nonzero_user_idx[test_idx]
                    feed_dict = self.construct_feeddict(user_idx, testM)

                    mse_test, mae_test, summary_str = self.sess.run(
                        [self.MSE, self.MAE, self.summary_op], feed_dict=feed_dict)
                    print("Step {0} | Train MSE: {1:3.4f}, MAE: {2:3.4f}".format(
                        step, mse, mae))
                    print("         | Test  MSE: {0:3.4f}, MAE: {1:3.4f}".format(
                        mse_test, mae_test))

                    test_writer.add_summary(summary_str, step)
                else:
                    print("Step {0} | Train MSE: {1:3.4f}, MAE: {2:3.4f}".format(
                        step, mse, mae))

        self.saver.save(self.sess, result_path + "/model.ckpt")
