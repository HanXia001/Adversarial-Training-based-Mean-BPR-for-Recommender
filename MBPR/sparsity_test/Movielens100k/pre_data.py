import random
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
import scores
from tqdm import tqdm

class BPR:
    user_count = 943
    item_count = 1682
    latent_factors = 20
    lr = 0.01
    reg = 0.01
    train_count = 1000
    train_data_path = 'u5.base'
    test_data_path = 'u5.test'
    size_u_i = user_count * item_count
    # latent_factors of U & V
    U = np.random.rand(user_count, latent_factors) * 0.01
    np.savetxt("U.txt",U)
    # U = np.loadtxt("U.txt")
    V = np.random.rand(item_count, latent_factors) * 0.01
    np.savetxt("V.txt",V)
    # V = np.loadtxt("V.txt")

    biasV = np.random.rand(item_count) * 0.01
    np.savetxt("biasV.txt",biasV)
    # biasV = np.loadtxt("biasV.txt")

    test_data = np.zeros((user_count, item_count))
    test = np.zeros(size_u_i)
    predict_ = np.zeros(size_u_i)

    def load_data(self, path):
        line_num = 0
        train_file = open("u5_50.base","w")
        with open(path, 'r') as f:
            for line in f.readlines():
                if line_num % 2 == 0:
                    train_file.writelines(line)
                line_num += 1
        train_file.close()

    def load_test_data(self, path):
        line_num = 0
        test_file = open("u5_50.test", "w")
        with open(path, 'r') as f:
            for line in f.readlines():
                if line_num % 2 == 0:
                    test_file.writelines(line)
                line_num += 1
        test_file.close()

if __name__ == '__main__':
    bpr = BPR()
    # bpr.load_data(bpr.train_data_path)
    bpr.load_test_data(bpr.test_data_path)