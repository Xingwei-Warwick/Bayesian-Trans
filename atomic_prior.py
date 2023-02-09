import torch
import numpy as np
import time
from dataset_util import ATOMIC_dataset
from argparse import ArgumentParser
from rgcn.link_predict import LinkPredict, node_norm_to_edge_norm
from rgcn import utils
import pickle


def get_adj_and_degrees(num_nodes, triplets):
    """ Get adjacency list and degrees of the graph
    """
    adj_list = [[] for _ in range(num_nodes)]
    for i,triplet in enumerate(triplets):
        adj_list[triplet[0]].append([i, triplet[2]])
        #adj_list[triplet[2]].append([i, triplet[0]])

    degrees = np.array([len(a) for a in adj_list])
    adj_list = [np.array(a) for a in adj_list]
    return adj_list, degrees


def compute_f1(num_r, embedding, w, s, r, o):
    # need to comput multi-label, to fussy

    # all predict postitive, something wrong with negative sampling??

    emb_s = embedding[s]
    emb_r = w[r]
    emb_o = embedding[o]
    emb_triplet = emb_s * emb_r * emb_o
    scores = torch.sigmoid(torch.sum(emb_triplet, dim=1)) - 0.5
    scores = scores.detach().cpu().numpy()

    scores = np.where(scores>0, scores, 0.)
    correct = scores.nonzero()
    corr_num = correct[0].shape[0]
    acc = corr_num * 1.0 / r.size(0)

    return acc

def main(args):
    if torch.cuda.is_available() and args.gpu > 0:
        device = f'cuda:%d' % (args.gpu-1)
    else:
        device = 'cpu'
    print('Using Device: ' + device)

    atomic = ATOMIC_dataset()
    num_nodes = len(atomic)
    num_rels = atomic.get_num_rels()

    model = LinkPredict(num_nodes, # in_dim
                        args.n_hidden,  # out_dim
                        num_rels,
                        num_bases=args.n_bases,
                        num_hidden_layers=args.n_layers,
                        dropout=args.dropout,
                        reg_param=args.regularization,
                        load_weight=torch.tensor(np.load('./event_rep_cache_cls.npy')))
    
    # validation and testing triplets
    valid_data = torch.LongTensor(atomic.dev)
    test_data = torch.LongTensor(atomic.test)

    # build test graph
    test_graph, test_rel, test_norm = utils.build_test_graph(
        num_nodes, num_rels, atomic.train) # using the whole trainning graph as validation (no negative samples here)
    test_deg = test_graph.in_degrees(
                range(test_graph.number_of_nodes())).float().view(-1,1)
    test_node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    test_rel = torch.from_numpy(test_rel)
    test_norm = node_norm_to_edge_norm(test_graph, torch.from_numpy(test_norm).view(-1, 1))

    model.to(device)

    # build adj list and calculate degrees for sampling
    adj_list, degrees = utils.get_adj_and_degrees(num_nodes, atomic.train)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model_state_file = args.expname + 'model_state.pth'
    forward_time = []
    backward_time = []

    # training loop
    print("start training...")

    epoch = 0
    best_mrr = 0
    while True:
        model.train()
        epoch += 1

        # perform edge neighborhood sampling to generate training graph and data
        g, node_id, edge_type, node_norm, data, labels = \
            utils.generate_sampled_graph_and_labels(
                atomic.train, args.graph_batch_size, args.graph_split_size,
                num_rels, adj_list, degrees, args.negative_sample,
                args.edge_sampler)
        print("Done edge sampling")

        #print('Average degree:', g.in_degrees().sum()*1.0/g.num_nodes(), g.out_degrees().sum()*1.0/g.num_nodes())

        # set node/edge feature
        node_id = torch.from_numpy(node_id).view(-1, 1).long()
        edge_type = torch.from_numpy(edge_type)
        edge_norm = node_norm_to_edge_norm(g, torch.from_numpy(node_norm).view(-1, 1))
        data, labels = torch.from_numpy(data), torch.from_numpy(labels)
        deg = g.in_degrees(range(g.number_of_nodes())).float().view(-1, 1)

        node_id, deg = node_id.to(device), deg.to(device)
        edge_type, edge_norm = edge_type.to(device), edge_norm.to(device)
        data, labels = data.to(device), labels.to(device)
        g = g.to(device)

        t0 = time.time()
        embed = model(g, node_id, edge_type, edge_norm)
        loss = model.get_loss(g, embed, data, labels)
        t1 = time.time()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm) # clip gradients
        optimizer.step()
        t2 = time.time()

        forward_time.append(t1 - t0)
        backward_time.append(t2 - t1)
        print("Epoch {:04d} | Loss {:.4f} | Best MRR {:.4f} | Forward {:.4f}s | Backward {:.4f}s".
              format(epoch, loss.item(), best_mrr, forward_time[-1], backward_time[-1]))

        log_dict = {'Training loss': loss.item()}

        optimizer.zero_grad()

        # validation
        if epoch % args.evaluate_every == 0:
            # perform validation on CPU because full graph is too large
            model.cpu()
            model.eval()
            print("start eval")
            with torch.no_grad():
                '''
                if use_cuda:
                    to(device)test_node_id = test_node_id.cuda()
                    test_norm = test_norm.cuda()
                    test_rel = test_rel.cuda()
                    test_graph = test_graph.to(args.gpu)
                    # might need to move these to cpu to save memory after eval'''
                embed = model(test_graph, test_node_id, test_rel, test_norm)
                mrr = utils.calc_mrr(embed, model.w_relation, torch.LongTensor(atomic.train),
                                    valid_data, test_data, hits=[1, 3, 10], eval_bz=args.eval_batch_size,
                                    eval_p=args.eval_protocol)
            # save best model
            if best_mrr < mrr:
                best_mrr = mrr
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
            
            if epoch >= args.n_epochs:
                break
            
            model.to(device)
            
    print("training done")
    print("Mean forward time: {:4f}s".format(np.mean(forward_time)))
    print("Mean Backward time: {:4f}s".format(np.mean(backward_time)))
    final_log = {'Best val mrr': best_mrr}

    print("\nstart testing:")
    # use best model checkpoint
    checkpoint = torch.load(model_state_file)
    model.to(device)
    model.eval()
    model.load_state_dict(checkpoint['state_dict'])
    print("Using best epoch: {}".format(checkpoint['epoch']))
    embed = model(test_graph, test_node_id, test_rel, test_norm)
    test_mrr = utils.calc_mrr(embed, model.w_relation, torch.LongTensor(atomic.train), valid_data,
                   test_data, hits=[1, 3, 10], eval_bz=args.eval_batch_size, eval_p=args.eval_protocol)
                   
    final_log['Test mrr': test_mrr]

def process_rgcn_emb(args):
    save_file = args.expname + 'model_state.pth'
    state_dict = torch.load(save_file, map_location='cpu')['state_dict']['w_relation'].numpy()
    relations_paras = []
    relations_paras.append(state_dict[1])
    relations_paras.append(state_dict[0])
    relations_paras.append(np.zeros_like(relations_paras[0]))
    relations_paras.append(np.zeros_like(relations_paras[0]))

    #relations_matrix = np.stack(relations_paras)
    relations_matrix = np.concatenate(relations_paras, axis=-1)

    print('Saving prior in %s')
    # print(relations_matrix.shape)

    pickle.dump(relations_matrix, open(save_file.split('.')[0]+'_prior.pkl', 'wb'))


if __name__ == '__main__':
    parser = ArgumentParser(description='Train a RGCN on ATOMIC')

    parser.add_argument("--dropout", type=float, default=0.2,
            help="dropout probability")
    parser.add_argument("--sim-thresh", type=float, default=0.9,
            help="thresh to build semantic similarity edges")
    parser.add_argument("--n-hidden", type=int, default=500,
            help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-bases", type=int, default=100,
            help="number of weight blocks for each relation")
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of propagation rounds")
    parser.add_argument("--n-epochs", type=int, default=6000,
            help="number of minimum training epochs")
    parser.add_argument("--eval-batch-size", type=int, default=500,
            help="batch size when evaluating")
    parser.add_argument("--eval-protocol", type=str, default="filtered",
            help="type of evaluation protocol: 'raw' or 'filtered' mrr")
    parser.add_argument("--regularization", type=float, default=0.01,
            help="regularization weight")
    parser.add_argument("--grad-norm", type=float, default=1.0,
            help="norm to clip gradient to")
    parser.add_argument("--graph-batch-size", type=int, default=30000,
            help="number of edges to sample in each iteration")
    parser.add_argument("--graph-split-size", type=float, default=0.5,
            help="portion of edges used as positive sample")
    parser.add_argument("--negative-sample", type=int, default=10,
            help="number of negative samples per positive sample")
    parser.add_argument("--evaluate-every", type=int, default=500,
            help="perform evaluation every n epochs")
    parser.add_argument("--edge-sampler", type=str, default="uniform",
            help="type of edge sampler: 'uniform' or 'neighbor'")
    parser.add_argument("--expname", type=str, default="atomic_prior_exp",
            help="experiment name, determine the name of save files")

    args = parser.parse_args()
    print(args)

    main(args)

    process_rgcn_emb(args)

