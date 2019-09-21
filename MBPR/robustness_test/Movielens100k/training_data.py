import numpy as np
import math


def get_y(x):
    return x

def sample_data(u_data=None,temp_u=None,v_data=None,temp_i=None):
    temp_u_x_y = []
    if u_data != None:
        all_u_data = np.loadtxt(u_data)
        pre_temp_u = all_u_data[temp_u]
        for i in range(len(pre_temp_u)):
            temp_u_x_y.append([i,pre_temp_u[i]])

    temp_v_x_y = []
    if v_data != None:
        all_v_data = np.loadtxt(v_data)
        pre_temp_v = all_v_data[temp_i]
        for i in range(len(pre_temp_v)):
            temp_v_x_y.append([i, pre_temp_v[i]])
    return np.array(temp_u_x_y),np.array(temp_v_x_y)
if __name__ == '__main__':
    max_c = -1
    min_c = 1
    t = []
    for i_temp in range(1682):
        print("当前项目：",i_temp)
        u, v = sample_data(v_data="self.V.txt",temp_i=i_temp)
        for i in v:
            if max_c < i[1]:
                max_c = i[1]
            if min_c > i[1]:
                min_c = i[1]
            t.append(i[1])
    print(max_c,min_c)
    print(max(t),min(t))