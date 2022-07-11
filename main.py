import sys
import argparse
import numpy as np
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
import pandas as pd
from helper_fn import validate_fixed_fn, validate_random_fn, get_adj, generate_data
from networks import HGNN
import torch
from torch_geometric.data import Data

# torch.manual_seed(184)
# np.random.seed(184)

my_parser = argparse.ArgumentParser()
my_parser.add_argument('--mode',                                    help='train or generate')
my_parser.add_argument('--id',          type=str,                   help='experiment ID')
my_parser.add_argument('--n',           type=int, default = 4,      help='LSAP problem dimension')
my_parser.add_argument('--h',           type=int, default = 32,     help='GNN hidden dimension')
my_parser.add_argument('--k',           type=int, default = 1,      help='scale factor C = k*rnd[0,1) (default 1)')
my_parser.add_argument('--s',           type=int, default = 1000,   help='number of samples to generate')
my_parser.add_argument('--fp',          type=str,                   help='path to save the generated samples')
my_parser.add_argument('--train_it',    type=int, default = 100000, help='number of training iterations')
my_parser.add_argument('--train_file',  type=str,                   help='path to the train samples')
my_parser.add_argument('--val_it',      type=int, default = 20000,  help='number of evaluation iterations')
my_parser.add_argument('--val_file',    type=str,                   help='path to the validation samples')
my_parser.add_argument('--test_file',   type=str,                   help='path to the test samples')
my_parser.add_argument('--test_chkpt',  type=str,                   help='path to the pre-trained checkpoint for test')
my_parser.add_argument('--e',           type=int, default = 50,     help='number of training epochs')
args = my_parser.parse_args()

param_dict = {
    'N': args.n,        # problem dimension
    'H': args.h,        # hidden dim
    'K': args.k      
}

loss_fn = torch.nn.CrossEntropyLoss()

model = HGNN(param_dict['N'],
             param_dict['H'],
             param_dict['N'])

optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

def start_train():
    train_iters = args.train_it
    val_iters = args.val_it
    epochs = args.e

    loss_list = []
    eval_acc_list = []
    eval_loss_list = []
    print('Start Train')
    print('LSAP dim. N = ', param_dict['N'])

    edge_index = get_adj(param_dict['N'])
    
    # Load fixed train/validation/test data
    if args.train_file is not None:
        train_data = np.load(args.train_file)
        train_iters = train_data.shape[0]
        print(f'train data {train_data.shape} loaded')
    
    if args.val_file is not None:
        val_data = np.load(args.val_file)
        val_iters = val_data.shape[0]
        print(f'validation data {val_data.shape} loaded')

    model.reset_parameters()
    for epoch in range(epochs):
       
        print(f'train epoch {epoch+1}/{epochs}...')
        
        model.train()
        
        for t_idx in tqdm(range(train_iters)):
            
            epoch_total_loss = 0
            optimizer.zero_grad()
            
            if args.train_file is not None:
                cost_matrix = train_data[t_idx]       # load train sample
            else:
                cost_matrix = np.random.random_sample((param_dict['N'], param_dict['N']))       # generate synthetic train sample
            
            x = torch.from_numpy(np.concatenate((cost_matrix, cost_matrix.T), axis=0)).float()

            # build graph
            G = Data(x, edge_index)

            # compute ground truth with Hungarian alg.
            r, c = linear_sum_assignment(cost_matrix)

            # get predictions
            pred = model(G.x, G.edge_index)

            # compute loss, store and backpropagate
            loss_1 = loss_fn(pred, torch.from_numpy(c))

            d = np.argsort(c)  # column-wise truth indices (d) from row-wise ones (c)
            loss_2 = loss_fn(pred.T, torch.from_numpy(d))

            total_loss = loss_1 + loss_2

            epoch_total_loss += total_loss.item()

            total_loss.backward()

            # optionally, clip gradients
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # update weights
            optimizer.step()

        loss_list.append(epoch_total_loss / train_iters)

        # perform evaluation every epoch
        print('validate...')
        if args.val_file is not None:
            eval_correct, eval_loss = validate_fixed_fn(model, loss_fn, val_data, param_dict)  # validate on fixed data
        else:
            eval_correct, eval_loss = validate_random_fn(model, loss_fn, val_iters, param_dict)  # validate on random data
        eval_acc_list.append(eval_correct)
        eval_loss_list.append(eval_loss)
        print(f' Accuracy: {(100 * eval_correct):.4f}% - loss: {eval_loss:.12f}')
        print('')
        
    # save results and model
    perf_dict = {'eval_accuracy': eval_acc_list, 'eval_loss': eval_loss_list} 
    df = pd.DataFrame(perf_dict)
    df.to_csv('log_' + args.id + '.csv')
    print('log saved')
    torch.save(model.state_dict(), 'trained_net_' + args.id + '.pth')
    print('model saved')


def start_test():
    
    edge_index = get_adj(param_dict['N'])
    # Load test data
    if args.test_file is not None:
        test_data = np.load(args.test_file)
        test_iters = test_data.shape[0]
        print(f'test data {test_data.shape} loaded')
    else:
        print('test_file not found')
        sys.exit()
        
    model.load_state_dict(torch.load(args.test_chkpt))
    model.eval()
    print(f'Test...  ')
    test_correct, test_loss = validate_fixed_fn(model, loss_fn, test_data, param_dict)  # validate/test
    
    print(f'Accuracy: {(100 * test_correct):.4f}% - loss: {test_loss:.12f}')
    print('done')
    

if __name__ == '__main__':

    if args.mode == 'train':
        start_train()
    elif args.mode == 'generate':
        generate_data(args.s, param_dict, args.fp)
    elif args.mode == 'test':
        start_test()
    else:
        print(f'unsupported mode: {args.mode}')
