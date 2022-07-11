import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch_geometric.data import Data
from torch_sparse import SparseTensor
from tqdm import tqdm
from networks import HGNN


# adjacency matrix for bipartite graph
def get_adj(N):
    b0 = torch.zeros(N, N)
    b1 = torch.ones(N, N)
    bn = torch.cat((torch.cat((b0, b1), dim=0), torch.cat((b1, b0), dim=0)), dim=1)
    adj = SparseTensor.from_dense(bn)
    row, col, edge_attr = adj.t().coo()
    edge_index = torch.stack([row, col], dim=0)
    return edge_index


def avoid_coll(prednp, param_dict):
    pp = np.zeros((param_dict['N'], param_dict['N']))
    minn = prednp.min()
    for elms in range(param_dict['N']):
        r1, c1 = np.where(prednp == prednp.max())
        prednp[r1, :] = np.repeat(minn, param_dict['N'])
        prednp[:, c1] = np.expand_dims(np.repeat(minn, param_dict['N']), axis=0).T
        pp[r1, c1] = 1
    return np.argmax(pp, axis=1)


def validate_fixed_fn(model, loss_fn, val_data, param_dict):
    eval_correct = 0
    eval_loss = 0
    edge_index = get_adj(param_dict['N'])
    model.eval()
    for v_idx in range(len(val_data)):
        correct = 0
        cost_matrix = val_data[v_idx]
        x = torch.from_numpy(np.concatenate((cost_matrix, cost_matrix.T), axis=0)).float()
        G = Data(x, edge_index)
        r, c = linear_sum_assignment(cost_matrix)  # truth
        pred = model(G.x, G.edge_index)  # predicted
        loss_1 = loss_fn(pred, torch.from_numpy(c))
        d = np.argsort(c)
        loss_2 = loss_fn(pred.T, torch.from_numpy(d))
        eval_loss = loss_1.item() + loss_2.item()
        t_idx = avoid_coll(pred.detach().numpy(), param_dict)
        # soft threshold criterion
        for a_idx in range(param_dict['N']):
            if t_idx[a_idx] == c[a_idx]:
                correct += 1
        eval_correct += (correct / param_dict['N'])
    return eval_correct / len(val_data), eval_loss / len(val_data)


def validate_random_fn(model, loss_fn, val_iters, param_dict):
    eval_correct = 0
    eval_loss = 0
    edge_index = get_adj(param_dict['N'])
    model.eval()
    for v_idx in range(val_iters):
        correct = 0
        cost_matrix = np.random.random_sample((param_dict['N'], param_dict['N']))       # generate synthetic train sample
        x = torch.from_numpy(np.concatenate((cost_matrix, cost_matrix.T), axis=0)).float()
        G = Data(x, edge_index)
        r, c = linear_sum_assignment(cost_matrix)  # truth
        pred = model(G.x, G.edge_index)  # predicted
        loss_1 = loss_fn(pred, torch.from_numpy(c))
        d = np.argsort(c)
        loss_2 = loss_fn(pred.T, torch.from_numpy(d))
        eval_loss = loss_1.item() + loss_2.item()
        t_idx = avoid_coll(pred.detach().numpy(), param_dict)
        # soft threshold criterion
        for a_idx in range(param_dict['N']):
            if t_idx[a_idx] == c[a_idx]:
                correct += 1
        eval_correct += (correct / param_dict['N'])
    return eval_correct / val_iters, eval_loss / val_iters


def generate_data(n_samples, param_dict, fname):
    cm = param_dict['K'] * np.random.random_sample((1, param_dict['N'], param_dict['N']))
    for t in tqdm(range(1, n_samples)):
        cm2 = param_dict['K'] * np.random.random_sample((1, param_dict['N'], param_dict['N']))
        cm = np.concatenate((cm, cm2))
    np.save(fname, cm)
    print('done')


