import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold

from model import VAEMF

if __name__ == '__main__':
    n_user = 943
    n_item = 1682

    n_fold = 10

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

    num_rating = np.count_nonzero(M)
    rating_idx = np.array(M.nonzero()).T
    kf = KFold(n_splits=n_fold, random_state=0)

    with tf.Session() as sess:
        model = VAEMF(sess, user_input_dim, item_input_dim,
                      hidden_encoder_dim=500, hidden_decoder_dim=500)
        for i, (train_idx, test_idx) in enumerate(kf.split(rating_idx)):
            print("{0}/{1} Fold start| Train size={2}, Test size={3}".format(i,
                                         n_fold, train_idx.size, test_idx.size))
            model.train(M, train_idx=train_idx,
                        test_idx=test_idx, n_steps=100000)
