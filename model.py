from node2vec import Node2Vec
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from torch import nn

from models.deepwalk.deepwalk.__main__ import DeepWalk
import numpy as np
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import networkx as nx
from sklearn.multiclass import OneVsRestClassifier

class node_embedding(object):
    def __init__(self, model_name):
        self.model_name = model_name
        self.embedding_model = None

        if self.model_name.upper() == 'DEEPWALK':
            self.embedding_model = DeepWalk()
        if self.model_name.upper() == 'NODE2VEC':
            self.embedding_model = None

    def process_embedding(self, edge):
        if self.model_name.upper() == 'DEEPWALK':
            self.embedding_model.edge = edge
            self.embedding_model.max_memory_data_size = 0
            self.embedding_model.number_walks = 8
            self.embedding_model.representation_size = 128
            self.embedding_model.walk_length = 40
            self.embedding_model.window_size = 10
            self.embedding_model.workers = 8
            self.embedding_model.output = 'output/log.embeddings'

            return self.embedding_model.process()

        if self.model_name.upper() == 'NODE2VEC':
            G = nx.Graph()
            G.add_edges_from(edge)
            node2vec = Node2Vec(G, dimensions=128, walk_length=30, num_walks=200, workers=4)
            model = node2vec.fit(window=10, min_count=1, batch_words=4)

            feature = model.wv.vectors
            index_to_key = np.asarray(model.wv.index_to_key).astype(int)
            feature = np.hstack([feature[np.argsort(index_to_key), :]])
            index_to_key = np.sort(index_to_key)

            return index_to_key, feature

class Predict(object):
    def __init__(self, model_name, feature_dim, classes):
        self.model = None
        self.model_name = model_name
        if model_name.upper() == 'MLP':
            self.model = MLP(feature_dim, 128, classes, True)
            self.model.cuda()
        if model_name.upper() == 'TOPKRANKER':
            self.model = OneVsRestClassifier(LogisticRegression())
        if model_name.upper() == 'RANDOMFOREST':
            self.model = RandomForestClassifier(n_estimators=100)

    def train(self, X_train, X_test, Y_train, Y_test):
        if self.model_name.upper() == 'MLP':
            X_train = torch.from_numpy(X_train).to(torch.float32)
            X_test = torch.from_numpy(X_test).to(torch.float32)
            labels_train = torch.from_numpy(Y_train)
            labels_test = torch.from_numpy(Y_test)

            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)
            self.model.train()
            for epoch in range(10000):
                optimizer.zero_grad()
                out = self.model(X_train.to('cuda'))
                loss = torch.nn.CrossEntropyLoss()(out, labels_train.to('cuda'))

                out_test = self.model(X_test.to('cuda'))
                acc_train = accuracy(out, labels_train.to('cuda'))
                acc_test = accuracy(out_test, labels_test.to('cuda'))
                loss.backward()
                optimizer.step()
                print(f"epoch:{epoch + 1}, loss:{loss.item()}, acc_train:{acc_train}, acc_test:{acc_test}")

        if self.model_name.upper() == 'TOPKRANKER':
            clf = SVC(C=1)
            clf = CalibratedClassifierCV(clf, method='sigmoid')
            clf = OneVsRestClassifier(clf)
            clf.fit(X_train, Y_train)

            # Predict probabilities:
            y_pred = clf.predict(X_train)
            y_pred_test = clf.predict(X_test)
            print('training acc = ' + str(np.sum(y_pred == Y_test)/Y_train.shape[0]) + ' testing acc = ' + str(np.sum(y_pred_test == Y_test)/Y_test.shape[0]))

        if self.model_name.upper() == 'RANDOMFOREST':
            acc = cross_val_score(estimator=self.model, X=X_train, y=Y_train, cv=10)
            print("average accuracy :", np.mean(acc))
            print("average std :", np.std(acc))

            self.model.fit(X_train, Y_train)
            print("test accuracy :", self.model.score(X_test, Y_test))

        return self.model

class GAT(nn.Module):
    def __init__(self, feature, hidden, classes, heads=1):
        super(GAT,self).__init__()
        self.gat1 = GATConv(feature, hidden, heads=heads)
        self.gat2 = GATConv(hidden*heads, classes)
    def forward(self, features, edges):
        features = self.gat1(features, edges)       # edges 这里输入是(1,2),表示1和2有边相连。
        features = F.relu(features)
        features = F.dropout(features, training=self.training)
        features = self.gat2(features, edges)
        return F.log_softmax(features, dim=1)

class MLP(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, self_attention=False):
        super(MLP, self).__init__()
        torch.manual_seed(12345)
        self.self_attention = self_attention
        if self_attention:
            self.attention_layers = SelfAttention(num_features)
        self.lin1 = Linear(num_features, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels//2)
        self.lin3 = Linear(hidden_channels//2, num_classes)
    def forward(self, x):
        if self.self_attention:
            w = self.attention_layers(x)
            x = w * x
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)
        x = x.relu()
        x = self.lin3(x)
        return x

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        x = x.unsqueeze(0)
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values).squeeze(0)
        return weighted