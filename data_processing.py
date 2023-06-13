import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


def read_attr(path):
    data = np.loadtxt(path, delimiter=",")
    sim = SimpleImputer(missing_values=0, strategy='mean')
    sim.fit(data[:4000, 1:])
    data[:, 1:] = sim.transform(data[:, 1:])
    # one-hot encoding
    # enc = OneHotEncoder(sparse_output=False)
    # one_hot = enc.fit_transform(data[:, 2:5])
    # data = np.hstack([data[:, 0:2], one_hot, data[:, 5:7]])
    # one-hot encoding end
    return data


def read_edgelist(path):
    f = open(path, "r")

    # This assumes your spacing is arbitrary
    data = [[x for x in line.strip().split(',') if x] for line in f]

    f.close()

    vertex = [int(i[0]) for i in data]
    adj = [i[1:] for i in data]
    adj_list = [[[i, int(j)] for j in adj[i]] for i in vertex]

    adj_ = []
    for i in adj_list:
        for j in i:
            if j:
                adj_.append(j)
                adj.append([j[1], j[0]])

    return np.unique(np.asarray(adj_), axis=0)


def read_adj(path):
    f = open(path, "r")
    # This assumes your spacing is arbitrary
    data = [[x for x in line.strip().split(',') if x] for line in f]
    f.close()

    v_size = len(data)
    adj = np.zeros([v_size, v_size]).astype(int)

    for i in range(len(data)):
        for j in data[i]:
            adj[i, int(j)] = 1
            adj[int(j), i] = 1
    return adj


def read_label(path):
    data = np.loadtxt(path, delimiter=",")
    return data
