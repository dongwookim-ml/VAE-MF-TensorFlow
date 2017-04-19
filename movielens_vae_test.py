import itertools
import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold

from matrix_vae import VAEMF

# dataset = "ml-100k"
dataset = "ml-1m"

if dataset == "ml-100k":
    # 100k dataset
    num_user = 943
    num_item = 1682
else:
    # 1M dataset
    num_user = 6040
    num_item = 3952

hidden_encoder_dim = 216
hidden_decoder_dim = 216
latent_dim = 24
learning_rate = 0.002
batch_size = 64
reg_param = 1e-10

n_steps = 2

hedims = [500]
hddims = [500]
ldims = [100]
lrates = [0.001]
bsizes = [512]
regs = [0, 1e-10, 1e-7, 1e-5]
vaes = [True]

def read_dataset():
    M = np.zeros([num_user, num_item])
    if dataset == "ml-100k":
        path ="./data/ml-100k/u.data"
        delim = "\t"
    else:
        path = "./data/ml-1m/ratings.dat"
        delim = "::"
    with open(path, 'r') as f:
        for line in f.readlines():
            tokens = line.split(delim)
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
                      latent_dim=latent_dim, output_dim=output_dim, learning_rate=learning_rate, batch_size=batch_size, reg_param=reg_param)

        for i, (train_idx, test_idx) in enumerate(kf.split(rating_idx)):
            print("{0}/{1} Fold start| Train size={2}, Test size={3}".format(i,
                                                                             n_fold, train_idx.size, test_idx.size))
            model.train(M, train_idx=train_idx,
                        test_idx=test_idx, n_steps=n_steps)


def train_test_validation():
    M = read_dataset()

    num_rating = np.count_nonzero(M)
    idx = np.arange(num_rating)
    np.random.seed(1)
    np.random.shuffle(idx)

    train_idx = idx[:int(0.85 * num_rating)]
    valid_idx = idx[int(0.85 * num_rating):int(0.90 * num_rating)]
    test_idx = idx[int(0.90 * num_rating):]

    for hidden_encoder_dim, hidden_decoder_dim, latent_dim, learning_rate, batch_size, reg_param, vae in itertools.product(hedims, hddims, ldims, lrates, bsizes, regs, vaes):
        result_path = "{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}".format(
            hidden_encoder_dim, hidden_decoder_dim, latent_dim, learning_rate, batch_size, reg_param, vae)
        if not os.path.exists(result_path + "/model.ckpt.index"):
            config = tf.ConfigProto()
            config.gpu_options.allow_growth=True
            with tf.Session(config=config) as sess:
                model = VAEMF(sess, num_user, num_item,
                              hidden_encoder_dim=hidden_encoder_dim, hidden_decoder_dim=hidden_decoder_dim,
                              latent_dim=latent_dim, learning_rate=learning_rate, batch_size=batch_size, reg_param=reg_param, vae=vae)
                print("Train size={0}, Validation size={1}, Test size={2}".format(
                    train_idx.size, valid_idx.size, test_idx.size))
                print(result_path)
                best_rmse = model.train_test_validation(M, train_idx=train_idx, test_idx=test_idx, valid_idx=valid_idx, n_steps=n_steps, result_path=result_path)

                print("Best MSE = {0}".format(best_rmse))

                with open('result.csv', 'a') as f:
                    f.write("{0},{1},{2},{3},{4},{5},{6},{7}\n".format(hidden_encoder_dim, hidden_decoder_dim,
                                                                               latent_dim, learning_rate, batch_size, reg_param, vae, best_rmse))

        tf.reset_default_graph()

if __name__ == '__main__':
    train_test_validation()
    # cross_validation()
