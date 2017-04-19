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
                 one_hot=False, user_embed_dim=216, item_embed_dim=216):

        if reg_param < 0 or reg_param > 1:
            raise ValueError("regularization parameter must be in [0,1]")

        self.sess = sess
        self.num_user = num_user
        self.num_item = num_item
        self.hidden_encoder_dim = hidden_encoder_dim
        self.hidden_decoder_dim = hidden_decoder_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.reg_param = reg_param
        self.one_hot = one_hot
        self.user_embed_dim = user_embed_dim
        self.item_embed_dim = item_embed_dim
        if self.one_hot:
            self.user_input_dim = self.num_user
            self.item_input_dim = self.num_item
        else:
            self.user_input_dim = self.num_item
            self.item_input_dim = self.num_user
        self.build_model()

    def build_model(self):
        self.l2_loss = tf.constant(0.0)

        self.user = tf.placeholder("float", shape=[None, self.user_input_dim])
        self.item = tf.placeholder("float", shape=[None, self.item_input_dim])
        self.user_idx = tf.placeholder(tf.int64, shape=[None])
        self.item_idx = tf.placeholder(tf.int64, shape=[None])
        self.rating = tf.placeholder("float", shape=[None])

        if self.one_hot:
            self.W_user_embed = weight_variable(
                [self.user_input_dim, self.user_embed_dim], 'user_embed')
            self.W_item_embed = weight_variable(
                [self.item_input_dim, self.item_embed_dim], 'item_embed')

            self.W_encoder_input_hidden_user = weight_variable(
                [self.user_embed_dim, self.hidden_encoder_dim], 'W_encoder_input_hidden_user')
            self.b_encoder_input_hidden_user = bias_variable(
                [self.hidden_encoder_dim], 'b_encoder_input_hidden_user')
            self.W_encoder_input_hidden_item = weight_variable(
                [self.item_embed_dim, self.hidden_encoder_dim], 'W_encoder_input_hidden_item')
            self.b_encoder_input_hidden_item = bias_variable(
                [self.hidden_encoder_dim], 'b_encoder_input_hidden_item')

            self.l2_loss += tf.nn.l2_loss(self.W_user_embed)
            self.l2_loss += tf.nn.l2_loss(self.W_item_embed)
        else:
            self.W_encoder_input_hidden_user = weight_variable(
                [self.user_input_dim, self.hidden_encoder_dim], 'W_encoder_input_hidden_user')
            self.b_encoder_input_hidden_user = bias_variable(
                [self.hidden_encoder_dim], 'b_encoder_input_hidden_user')
            self.W_encoder_input_hidden_item = weight_variable(
                [self.item_input_dim, self.hidden_encoder_dim], 'W_encoder_input_hidden_item')
            self.b_encoder_input_hidden_item = bias_variable(
                [self.hidden_encoder_dim], 'b_encoder_input_hidden_item')

        self.l2_loss += tf.nn.l2_loss(self.W_encoder_input_hidden_user)
        self.l2_loss += tf.nn.l2_loss(self.W_encoder_input_hidden_item)

        # Hidden layer encoder
        if self.one_hot:
            self.user_embed = tf.nn.embedding_lookup(
                self.W_user_embed, self.user_idx)
            self.item_embed = tf.nn.embedding_lookup(
                self.W_item_embed, self.item_idx)
            self.hidden_encoder_user = tf.nn.relu(tf.matmul(
                self.user_embed, self.W_encoder_input_hidden_user) + self.b_encoder_input_hidden_user)
            self.hidden_encoder_item = tf.nn.relu(tf.matmul(
                self.item_embed, self.W_encoder_input_hidden_item) + self.b_encoder_input_hidden_item)
        else:
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

        # KL divergence between prior and variational distributions
        self.KLD = -0.5 * tf.reduce_sum(1 + self.logvar_encoder_user - tf.pow(
            self.mu_encoder_user, 2) - tf.exp(self.logvar_encoder_user), reduction_indices=1)
        self.KLD -= 0.5 * tf.reduce_sum(1 + self.logvar_encoder_item - tf.pow(
            self.mu_encoder_item, 2) - tf.exp(self.logvar_encoder_item), reduction_indices=1)

        # where the tricky part starts
        # self.W_encoder_latent_latent = weight_variable(
        #     [5, self.latent_dim, self.latent_dim], 'W_encoder_latent_latent')
        # self.l2_loss += tf.nn.l2_loss(self.W_encoder_latent_latent)

        self.user_bias = bias_variable(self.num_user, 'user_bias')
        self.item_bias = bias_variable(self.num_item, 'item_bias')

        # self.multi = list()
        # for i in range(5):
        #     self.multi.append(tf.diag_part(
        #     tf.matmul(tf.matmul(self.z_user, self.W_encoder_latent_latent[i]), self.z_item, transpose_b=True)))
        #     self.multi[i] = tf.add(self.multi[i], tf.nn.embedding_lookup(self.user_bias, self.user_idx))
        #     self.multi[i] = tf.add(self.multi[i], tf.nn.embedding_lookup(self.item_bias, self.item_idx))
        #     self.multi[i] = tf.exp(self.multi[i])
        #
        # self.multi_sum = tf.add_n(self.multi)
        # self.rating_hat = tf.divide(self.multi[0], self.multi_sum)
        # for i in range(1, 5):
        #     self.rating_hat += (i+1) * tf.divide(self.multi[i], self.multi_sum)

        self.W_encoder_latent_latent = weight_variable([self.latent_dim, self.latent_dim], 'weighted_inner_product')

        # self.rating_hat = tf.diag_part(tf.matmul(tf.matmul(self.z_user, self.W_encoder_latent_latent), self.z_item, transpose_b=True))
        self.rating_hat = tf.reduce_sum(tf.multiply(tf.matmul(self.z_user, self.W_encoder_latent_latent), self.z_item), reduction_indices=1)
        self.rating_hat = tf.add(self.rating_hat, tf.nn.embedding_lookup(self.user_bias, self.user_idx))
        self.rating_hat = tf.add(self.rating_hat, tf.nn.embedding_lookup(self.item_bias, self.item_idx))

        self.MSE = tf.reduce_mean(
            tf.square(tf.subtract(self.rating, self.rating_hat)))
        self.MAE = tf.reduce_mean(
            tf.abs(tf.subtract(self.rating, self.rating_hat)))

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

    def construct_feeddict(self, user_idx, item_idx, M):
        if self.one_hot:
            feed_dict = {self.user_idx: user_idx, self.item_idx: item_idx,
                         self.rating: M[user_idx, item_idx]}
        else:
            feed_dict = {self.user: M[user_idx, :], self.item: M[
                :, item_idx].transpose(), self.user_idx:user_idx, self.item_idx:item_idx, self.rating: M[user_idx, item_idx]}
        return feed_dict

    def train_test_validation(self, M, train_idx, test_idx, valid_idx, n_steps=100000, result_path='result/'):
        nonzero_user_idx = M.nonzero()[0]
        nonzero_item_idx = M.nonzero()[1]

        train_size = train_idx.size
        trainM = np.zeros(M.shape)
        trainM[nonzero_user_idx[train_idx], nonzero_item_idx[train_idx]] = M[
            nonzero_user_idx[train_idx], nonzero_item_idx[train_idx]]

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
            item_idx = nonzero_item_idx[train_idx[batch_idx]]
            feed_dict = self.construct_feeddict(user_idx, item_idx, trainM)

            _, mse, mae, summary_str = self.sess.run(
                [self.train_step, self.MSE, self.MAE, self.summary_op], feed_dict=feed_dict)
            train_writer.add_summary(summary_str, step)

            if step % int(n_steps / 10) == 0:
                valid_user_idx = nonzero_user_idx[valid_idx]
                valid_item_idx = nonzero_item_idx[valid_idx]
                feed_dict = self.construct_feeddict(
                    valid_user_idx, valid_item_idx, M)
                mse_valid, mae_valid, summary_str = self.sess.run(
                    [self.MSE, self.MAE, self.summary_op], feed_dict=feed_dict)

                valid_writer.add_summary(summary_str, step)

                test_user_idx = nonzero_user_idx[test_idx]
                test_item_idx = nonzero_item_idx[test_idx]
                feed_dict = self.construct_feeddict(
                    test_user_idx, test_item_idx, M)
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

        train_writer = tf.summary.FileWriter(
            result_path + '/train', graph=self.sess.graph)
        test_writer = tf.summary.FileWriter(
            result_path + '/test', graph=self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

        for step in range(1, n_steps):
            batch_idx = np.random.randint(train_size, size=self.batch_size)
            user_idx = nonzero_user_idx[train_idx[batch_idx]]
            item_idx = nonzero_item_idx[train_idx[batch_idx]]
            feed_dict = self.construct_feeddict(user_idx, item_idx, trainM)

            _, mse, mae, summary_str = self.sess.run(
                [self.train_step, self.MSE, self.MAE, self.summary_op], feed_dict=feed_dict)
            train_writer.add_summary(summary_str, step)

            if step % 1000 == 0:

                if test_idx is not None:
                    user_idx = nonzero_user_idx[test_idx]
                    item_idx = nonzero_item_idx[test_idx]
                    feed_dict = self.construct_feeddict(user_idx, item_idx, M)

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
