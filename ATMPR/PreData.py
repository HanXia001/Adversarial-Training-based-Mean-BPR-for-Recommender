# -*- coding:utf-8 -*-
import random
from collections import defaultdict
import numpy as np
import time

class PreData(object):
    
    def __init__(self):
        user_count = 25676
        item_count = 25814
        latent_factors = 20
        lr = 0.01
        reg = 0.01
        train_count = 1000
        train_data_path = "yelp.train_yuan.rating"
        test_data_path = "yelp.test_yuan.rating"
        size_u_i = user_count * item_count

    def get_mean_rating(self,path):
        u_i_r = np.zeros((self.user_count+1,self.item_count+1))
        with open(path,"r") as f:
            for line in f.readlines():
                u, i, r, t = line.split("	")
                u = int(u)
                i = int(i)
                r = int(float(r))
                u_i_r[u][i] = r
        u_mean_rating = {}
        tbar = tqdm(total = self.user_count+1)
        for u in range(self.user_count+1):
            tbar.update(1)
            # if u == 0:
            #    continue
            num = 0
            for i in range(self.item_count+1):
                #if i ==0:
                    #continue
                if u_i_r[u][i] != 0:
                    num += 1
                    continue
            u_mean_rating[u] = sum(u_i_r[u])/num
        tbar.close()
        return u_i_r,u_mean_rating

    # 用于划分训练集与测试集
    def load_data(self, path_1):
        user_ratings = defaultdict(set)
        u_i_r_t ={} # {(u_i_r):t}
        max_u_id = -1
        max_i_id = -1
        u_i_r = {} # {(u,i):r}
        with open(path_1, 'r') as f:
            for line in f.readlines():
                u, i, r, t = line.split("	")
                u = int(u)
                i = int(i)
                r = int(float(r))
                t = int(t)
                u_i_r_t[(u,i,r)] = t
                u_i_r[(u,i)] = r
                user_ratings[u].add(i)
                max_u_id = max(u, max_u_id)
                max_i_id = max(i, max_i_id)
        # 将评分高于平均分的划分为训练集，将评分低于平均分的划分为测试集
        _,u_mean_rating = self.get_mean_rating(path=path_1)
        train_over_mean = []
        test_lower_mean = []
        # print(u_i_r_t)
        for line in u_i_r_t.keys():
            tbar_1.update(1)
            if line[2] >= u_mean_rating[line[0]]+1:
                train_over_mean.append((line,u_i_r_t[line]))
            else:
                test_lower_mean.append((line,u_i_r_t[line]))
        with open("train_over_mean.txt","w+") as f:
            for i in train_over_mean:
                f.write(str(i[0][0]) + "\t" + str(i[0][1]) + "\t" + str(i[0][2]) + "\t" + str(i[1]) + "\n")
        with open("test_lower_mean.txt","w+") as f:
            for i in test_lower_mean:
                f.write(str(i[0][0]) + "\t" + str(i[0][1]) + "\t" + str(i[0][2]) + "\t" + str(i[1]) + "\n")
