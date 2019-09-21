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
    # np.savetxt("U.txt",U)
    U = np.loadtxt("U.txt")
    # V = np.random.rand(item_count, latent_factors) * 0.01
    # np.savetxt("V.txt",V)
    V = np.loadtxt("V.txt")

    biasV = np.random.rand(item_count) * 0.01
    # np.savetxt("biasV.txt",biasV)
    biasV = np.loadtxt("biasV.txt")

    test_data = np.zeros((user_count, item_count))
    test = np.zeros(size_u_i)
    predict_ = np.zeros(size_u_i)

    def load_data(self, path):
        user_ratings = defaultdict(set)
        max_u_id = -1
        max_i_id = -1
        with open(path, 'r') as f:
            for line in f.readlines():
                u, i, r, t = line.split("	")
                u = int(u)
                i = int(i)
                user_ratings[u].add(i)
                max_u_id = max(u, max_u_id)
                max_i_id = max(i, max_i_id)
        return user_ratings

    def load_test_data(self, path):
        file = open(path, 'r')
        for line in file:
            line = line.split('	')
            user = int(line[0])
            item = int(line[1])
            self.test_data[user - 1][item - 1] = 1

    def predict(self, user, item):
        predict = np.mat(user) * np.mat(item.T)
        return predict

    def main(self):
        user_ratings_train = self.load_data(self.train_data_path)
        self.load_test_data(self.test_data_path)
        tbar_1 = tqdm(total=self.user_count*self.item_count)
        for u in range(self.user_count):
            for item in range(self.item_count):
                tbar_1.update(1)
                if int(self.test_data[u][item]) == 1:
                    self.test[u * self.item_count + item] = 1
                else:
                    self.test[u * self.item_count + item] = 0
        re_count = 30
        # BPR加噪音
        # self.U = np.loadtxt("self.U.txt")
        # self.V = np.loadtxt("self.V.txt")

        # MBPR加噪音
        self.U = np.loadtxt("self.U_MBPR.txt")
        self.V = np.loadtxt("self.V_MBPR.txt")

        # 此处可加噪音
        noise_U = np.random.rand(self.U.shape[0],self.U.shape[1]) * 1.0
        noise_V = np.random.rand(self.V.shape[0],self.V.shape[1]) * 1.0
        self.U = self.U + noise_U
        self.V = self.V + noise_V

        predict_matrix = self.predict(self.U, self.V)

        # prediction
        self.predict_ = predict_matrix.getA().reshape(-1)
        self.predict_ = pre_handel(user_ratings_train, self.predict_, self.item_count)
        auc_score = roc_auc_score(self.test, self.predict_)
        print('AUC:', auc_score)
        # Top-K evaluation
        MAP,MRR,Prec,Rec,F1,NDCG,l_call = scores.topK_scores(self.test, self.predict_, re_count, self.user_count, self.item_count)

def pre_handel(set, predict, item_count):
    # Ensure the recommendation cannot be positive items in the training set.
    for u in set.keys():
        for j in set[u]:
            predict[(u - 1) * item_count + j - 1] = 0
    return predict

if __name__ == '__main__':
    bpr = BPR()
    bpr.main()