import itertools
import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold

from model import VAEMF

num_user = 943
num_item = 1682

hidden_encoder_dim = 500
hidden_decoder_dim = 500
latent_dim = 250
output_dim = 250
learning_rate = 0.001
batch_size = 128
reg_param = 0
one_hot = False

n_steps = 1000000

hedims = [500]
hddims = [500]
ldims = [25, 64, 128]
odims = [25, 64, 128]
lrates = [0.001, 0.002, 0.01, 0.02]
bsizes = [512, 1024, 2048]
regs = [1e-7, 0.001, 0.002, 0.01, 0.1, 0.5, 1]
one_hots = [False, True]


def read_dataset():
    M = np.zeros([num_user, num_item])
    with open('./data/ml-100k/u.data', 'r') as f:
        for line in f.readlines():
            tokens = line.split()
            user_id = int(tokens[0]) - 1  # 0 base index
            item_id = int(tokens[1]) - 1
            rating = int(tokens[2])
            M[user_id, item_id] = rating
    return M


def cross_validation():
    M = read_dataset()
    n_fold = 10

    rating_idx = np.array(M.nonzero()).T
    kf = KFold(n_splits=n_fold, random_state=0)

    with tf.Session() as sess:
        model = VAEMF(sess, num_user, num_item,
                      hidden_encoder_dim=hidden_encoder_dim, hidden_decoder_dim=hidden_decoder_dim,
                      latent_dim=latent_dim, output_dim=output_dim, learning_rate=learning_rate, batch_size=batch_size, reg_param=reg_param, one_hot=one_hot)

        for i, (train_idx, test_idx) in enumerate(kf.split(rating_idx)):
            print("{0}/{1} Fold start| Train size={2}, Test size={3}".format(i,
                                                                             n_fold, train_idx.size, test_idx.size))
            model.train(M, train_idx=train_idx,
                        test_idx=test_idx, n_steps=n_steps)


def train():
    M = read_dataset()

    num_rating = np.count_nonzero(M)
    idx = np.arange(num_rating)
    np.random.seed(0)
    np.random.shuffle(idx)

    train_idx = idx[:int(0.8 * num_rating)]
    valid_idx = idx[int(0.8 * num_rating):int(0.9 * num_rating)]
    test_idx = idx[int(0.9 * num_rating):]

    result_path = "{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}".format(
        hidden_encoder_dim, hidden_decoder_dim, latent_dim, output_dim, learning_rate, batch_size, reg_param, one_hot)
    if not os.path.exists(result_path + "/model.ckpt.index"):
        with tf.Session() as sess:
            model = VAEMF(sess, num_user, num_item,
                          hidden_encoder_dim=hidden_encoder_dim, hidden_decoder_dim=hidden_decoder_dim,
                          latent_dim=latent_dim, output_dim=output_dim, learning_rate=learning_rate, batch_size=batch_size, reg_param=reg_param, one_hot=one_hot)
            print("Train size={0}, Validation size={1}, Test size={2}".format(
                train_idx.size, valid_idx.size, test_idx.size))
            best_mse, best_mae = model.train_test_validation(
                M, train_idx=train_idx, test_idx=test_idx, valid_idx=valid_idx, n_steps=n_steps, result_path=result_path)

            print("Best MSE = {0}, best MAE = {1}".format(
                best_mse, best_mae))

def train_test_validation():
    M = read_dataset()

    num_rating = np.count_nonzero(M)
    idx = np.arange(num_rating)
    np.random.seed(0)
    np.random.shuffle(idx)

    train_idx = idx[:int(0.8 * num_rating)]
    valid_idx = idx[int(0.8 * num_rating):int(0.9 * num_rating)]
    test_idx = idx[int(0.9 * num_rating):]

    for hidden_encoder_dim, hidden_decoder_dim, latent_dim, output_dim, learning_rate, batch_size, reg_param, one_hot in itertools.product(hedims, hddims, ldims, odims, lrates, bsizes, regs, one_hots):
        result_path = "{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}".format(
            hidden_encoder_dim, hidden_decoder_dim, latent_dim, output_dim, learning_rate, batch_size, reg_param, one_hot)
        if not os.path.exists(result_path + "/model.ckpt.index"):
            with tf.Session() as sess:
                model = VAEMF(sess, num_user, num_item,
                              hidden_encoder_dim=hidden_encoder_dim, hidden_decoder_dim=hidden_decoder_dim,
                              latent_dim=latent_dim, output_dim=output_dim, learning_rate=learning_rate, batch_size=batch_size, reg_param=reg_param, one_hot=one_hot)
                print("Train size={0}, Validation size={1}, Test size={2}".format(
                    train_idx.size, valid_idx.size, test_idx.size))
                best_mse, best_mae = model.train_test_validation(
                    M, train_idx=train_idx, test_idx=test_idx, valid_idx=valid_idx, n_steps=n_steps, result_path=result_path)

                print("Best MSE = {0}, best MAE = {1}".format(
                    best_mse, best_mae))

                with open('result.csv', 'a') as f:
                    f.write("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}\n".format(hidden_encoder_dim, hidden_decoder_dim,
                                                                               latent_dim, output_dim, learning_rate, batch_size, reg_param, one_hot, best_mse, best_mae))

        tf.reset_default_graph()


if __name__ == '__main__':
    train()
    # train_test_validation()
