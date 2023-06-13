import argparse

import torch
import torch.nn.functional as F

from torch_geometric.transforms import ToSparseTensor
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.data import Data

from GNNCluster import GNNCluster, GNNClusterDataset, GNNClusterLoader, ClusterOptimizer

from logger import Logger
from data_process import load_dataset


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x  # Modify


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x  # Modify


# Modify
def train(model, loader, train_idx, optimizer, device):
    model.train()

    optimizer.zero_grad()

    data = loader[0].to(device)
    out, infos, _ = model(data.x, data.adj_t)
    loader.update_cluster(infos[1])
    out = F.log_softmax(out, dim=-1)
    out, y = loader.convert(out), loader.convert(data.y)
    loss = F.nll_loss(out[train_idx], y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    # cluster_ids, n_per_c = torch.unique(infos[1], return_counts=True)
    # print(f"cluster infos: {len(cluster_ids)} clusters, "
    #       f"cluster_id:num_nodes->{dict(zip(cluster_ids.tolist(), n_per_c.tolist()))}")
    return loss.item()


# Modify
@torch.no_grad()
def test(model, loader, split_idx, device):
    def get_accuracy(y1: torch.Tensor, y2: torch.Tensor):
        y_cmp = y1 == y2
        y_cmp = y_cmp.reshape(-1, )
        acc = y_cmp.sum() / y_cmp.size(0)
        return acc

    model.eval()

    data = loader[0].to(device)

    out, infos, _ = model(data.x, data.adj_t)
    out = F.log_softmax(out, dim=-1)
    y_pred = out.argmax(dim=-1, keepdim=True)
    y_pred, y_true = loader.convert(y_pred), loader.convert(data.y)

    train_acc = get_accuracy(
        y_true[split_idx['train']],
        y_pred[split_idx['train']],
    )
    test_acc = get_accuracy(
        y_true[split_idx['test']],
        y_pred[split_idx['test']],
    )

    return train_acc, 0, test_acc, loader.convert(infos[3]), loader.convert(infos[1])


def main():
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--runs', type=int, default=10)

    parser.add_argument('--num_parts', type=int, default=5)
    parser.add_argument('--epoch_gap', type=int, default=99)
    parser.add_argument('--dropout_cluster', type=float, default=0.3)
    parser.add_argument('--warm_up', type=int, default=0)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    x, edge_index, y, split_idx, num_classes = load_dataset()
    train_idx = split_idx["train"]
    dataset = None

    data = Data(x=x, edge_index=edge_index, y=y)
    sparser = ToSparseTensor()
    data = sparser(data)
    data = data.to(device)

    # Modify
    if args.use_sage:
        model = SAGE(args.hidden_channels, args.hidden_channels,
                     args.hidden_channels, args.num_layers,
                     args.dropout).to(device)
    else:
        model = GCN(args.hidden_channels, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout).to(device)

    logger = Logger(args.runs, args)

    # Modify
    model = GNNCluster(model, data.num_features, args.hidden_channels, num_classes, None,
                       num_parts=args.num_parts, dropout=args.dropout_cluster).to(device)

    for run in range(args.runs):
        # Modify
        dataset = GNNClusterDataset(dataset, data, split_idx, num_parts=args.num_parts)
        training_loader = GNNClusterLoader(dataset, "all", is_eval=False, batch_size=-1, shuffle=False)
        testing_loader = GNNClusterLoader(dataset, "all", is_eval=True, batch_size=-1, shuffle=False)

        model.reset_parameters(dataset.get_init_vnode(device))

        cluster_optimizer = ClusterOptimizer(model, args.epoch_gap, args.lr, args.warm_up)
        optimizer = torch.optim.Adam(cluster_optimizer.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            # Modify
            cluster_optimizer.zero_grad_step(epoch)
            loss = train(model, training_loader, train_idx, optimizer, device)
            result = test(model, testing_loader, split_idx, device)
            logger.add_result(run, result[:3])

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result[:3]
                print(f'\033[1;31m'
                      f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%'
                      f'\033[0m')

        logger.print_statistics(run)
    logger.print_statistics()


if __name__ == "__main__":
    main()
