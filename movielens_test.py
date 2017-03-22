from model import VAEMF
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    n_user = 943
    n_item = 1682

    M = np.zeros([n_user, n_item])

    with open('./data/ml-100k/u.data', 'r') as f:
        for line in f.readlines():
            tokens = line.split()
            user_id = int(tokens[0]) - 1  # 0 base index
            item_id = int(tokens[1]) - 1
            rating = int(tokens[2])
            M[user_id, item_id] = rating

    num_user, num_item = M.shape

    user_input_dim = num_item
    item_input_dim = num_user
    with tf.Session() as sess:
        model = VAEMF(sess, user_input_dim, item_input_dim)
        model.train(M)
