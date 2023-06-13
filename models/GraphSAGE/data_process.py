import pandas as pd
import numpy as np
import torch
from torch_geometric.utils import to_undirected


def adjlist2edgeindex(path="data/adjlist.csv"):
    def read_adjlist(path):
        with open(path, 'r') as f:
            line = f.readline()
            while True:
                line_list = line.rstrip('\n').rstrip(',').split(",")
                if line_list[0] == '':
                    break
                node_adj_list = [int(i) for i in line_list]
                node_adj = torch.tensor(node_adj_list, dtype=torch.long)
                if node_adj.size(0) == 1:
                    line = f.readline()
                    continue
                node_idx = node_adj[0]
                adj_idxes = node_adj[1:]

                s = node_idx.reshape(1, -1).repeat(1, adj_idxes.size(0))
                e = adj_idxes.reshape(1, - 1)
                edge_index_ = torch.cat([s, e], dim=0)
                print(node_idx.item())
                yield edge_index_
                line = f.readline()

    edge_index = torch.empty((2, 0))
    for edge_index_part in read_adjlist(path):
        edge_index = torch.cat([edge_index, edge_index_part], dim=1)

    with open('data/edge_index.pth', 'wb') as f:
        torch.save(edge_index, f)


def attr2feat(path="data/attr.csv"):
    feat = np.loadtxt(path, delimiter=',', dtype=np.float32)[:, 1:]
    with open("data/xfeat.pth", 'wb') as f:
        torch.save(torch.tensor(feat, dtype=torch.float32), f)


def label2pth(path="data/label_train.csv"):
    label = np.loadtxt(path, delimiter=',', dtype=np.int32)[:, 1:]
    out_name = path.rstrip(".csv") + ".pth"
    with open(out_name, 'wb') as f:
        torch.save(torch.tensor(label, dtype=torch.long), f)


def load_edge_index(path="data/edge_index.pth"):
    with open(path, 'rb') as f:
        edge_index = torch.load(f)
    return edge_index


def load_feat_label(path="data/xfeat.pth"):
    with open(path, 'rb') as f:
        out = torch.load(f)
    print(out.size())
    print(out.min(), out.max())
    return out


def load_prediction(path="data/best_predict.pth"):
    pred = torch.load(path)[:, 1]
    train_pred = pred[:4000]
    test_pred = pred[4000:]
    test_true = load_feat_label("data/label_test.pth").reshape(-1, )

    test_cmp = test_pred == test_true
    test_acc = test_cmp.sum() / test_cmp.size(0)
    print(pred)


def load_dataset():
    x = load_feat_label("data/xfeat.pth")
    edge_index = load_edge_index().long()

    y_train = load_feat_label("data/label_train.pth")
    y_test = load_feat_label("data/label_test.pth")
    y = torch.cat([y_train, y_test], dim=0)

    min_y, max_y = 1900, 2010
    num_classes = max_y - min_y + 1
    y -= min_y

    train_split = torch.tensor([i for i in range(4000)], dtype=torch.long)
    valid_split = torch.tensor([], dtype=torch.long)
    test_split = torch.tensor([i for i in range(4000, 5298)], dtype=torch.long)

    split_idx = {"train": train_split, "valid": valid_split, "test": test_split}

    return x, edge_index, y, split_idx, num_classes


if __name__ == "__main__":
    # adjlist2edgeindex()
    # load_edge_index()
    # attr2feat()
    # label2pth()
    # load_feat_label("data/label_train.pth")
    load_prediction()
