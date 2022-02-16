import numpy as np
import scipy.sparse as sp
import torch
import time
from collections import defaultdict
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn.init import xavier_normal_
import math
from torch.autograd import Variable
from torch.nn import functional as F
import argparse

### Acknowledgment: The code is based on https://github.com/tkipf/pygcn, https://github.com/ibalazevic/TuckER, and https://github.com/TimDettmers/ConvE ###

class Dataset:

    def __init__(self, path):
        self.trainset, self.validset, self.testset = self.load_trip(path, "train"), self.load_trip(path, "valid"), self.load_trip(path, "test")
        self.trip = self.trainset + self.validset + self.testset
        self.ent = self.load_ent(self.trip)
        self.train_rel, self.valid_rel, self.test_rel = self.load_rel(self.trainset), self.load_rel(self.validset), self.load_rel(self.testset)
        self.rel = self.train_rel + [i for i in self.valid_rel if i not in self.train_rel] + [i for i in self.test_rel if i not in self.train_rel]

    def load_trip(self, path, subset):
        with open("%s%s.txt" % (path, subset), "r") as f:
            trip = f.read().strip().split("\n")
            trip = [i.split() for i in trip]
            trip += [[i[2], i[1]+"_reverse", i[0]] for i in trip]
        return trip

    def load_ent(self, trip):
        ent = sorted(list(set([d[0] for d in trip]+[d[2] for d in trip])))
        return ent

    def load_rel(self, trip):
        rel = sorted(list(set([d[1] for d in trip])))
        return rel



class GraphConvolutionLayer(torch.nn.Module):

    def __init__(self, in_features, out_features, num_entities):
        super(GraphConvolutionLayer, self).__init__()

        self.weight_H = torch.nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.weight_I = torch.nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias1 = torch.nn.Parameter(torch.FloatTensor(out_features))
        self.bias2 = torch.nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_H.size(1))
        self.weight_H.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.weight_I.size(1))
        self.weight_I.data.uniform_(-stdv, stdv)
        self.bias1.data.uniform_(-stdv, stdv)
        self.bias2.data.uniform_(-stdv, stdv)

    def complex_conv(self, input_H, input_I):
        support_H = torch.mm(input_H, self.weight_H) - torch.mm(input_I, self.weight_I)
        support_I = torch.mm(input_H, self.weight_I) + torch.mm(input_I, self.weight_H)
        return support_H, support_I


    def forward(self, input_H, input_I, adj):
        support_H, support_I  = self.complex_conv(input_H, input_I)
        output_H = torch.spmm(adj, support_H)
        output_I = torch.spmm(adj, support_I)
        output_H = output_H + self.bias1
        output_I = output_I + self.bias2
        return output_H, output_I

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class ComplexGCN(torch.nn.Module):
    def __init__(self, d, dim, adj, gcn_layers, drop1, drop2, drop3):
        super(ComplexGCN, self).__init__()

        self.adj = adj
        self.gcn_layers = gcn_layers
        self.EH, self.EI, self.RH, self.RI  = self.get_embeddings(d, dim)
        self.WH, self.WI, self.WJ, self.WK  = self.get_parameters(dim)
        self.gc1 = GraphConvolutionLayer(dim, dim, len(d.ent))
        self.gc2 = GraphConvolutionLayer(dim, dim, len(d.ent))
        self.drop1 = torch.nn.Dropout(drop1)
        self.drop2 = torch.nn.Dropout(drop2)
        self.drop3 = torch.nn.Dropout(drop3)
        self.bn0 = torch.nn.BatchNorm1d(dim)
        self.bn1 = torch.nn.BatchNorm1d(dim)
        self.loss = torch.nn.BCELoss()
        
    def init(self):
        xavier_normal_(self.EH.weight.data)
        xavier_normal_(self.EI.weight.data)
        xavier_normal_(self.RH.weight.data)
        xavier_normal_(self.RI.weight.data)

    def get_embeddings(self, d, dim):

        EH = torch.nn.Embedding(len(d.ent), dim)
        EI = torch.nn.Embedding(len(d.ent), dim)
        RH = torch.nn.Embedding(len(d.rel), dim)
        RI = torch.nn.Embedding(len(d.rel), dim)
        return EH, EI, RH, RI  


    def get_parameters(self, dim):

        WH = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (dim, dim)), 
                                    dtype=torch.float, device="cuda", requires_grad=True))
        WI = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (dim, dim)), 
                                    dtype=torch.float, device="cuda", requires_grad=True))
        WJ = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (dim, dim)), 
                                    dtype=torch.float, device="cuda", requires_grad=True))
        WK = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (dim, dim)), 
                                    dtype=torch.float, device="cuda", requires_grad=True))
        return WH, WI, WJ, WK 

    def scut_connection(self, init_emb_H, init_emb_I, out_emb_H, out_emb_I):

        final_emb_H = out_emb_H + init_emb_H
        final_emb_I = out_emb_I + init_emb_I
        return final_emb_H, final_emb_I

    def get_batch_emb(self, batch_head, batch_rel, final_emb_H, final_emb_I):

        xh = self.drop1(self.bn0(final_emb_H[batch_head]))
        xi = self.drop1(self.bn0(final_emb_I[batch_head]))
        rh = self.RH(batch_rel)
        ri = self.RI(batch_rel)
        return xh, xi, rh, ri 

    def mat_vec_mul(self, v, m):
        vm = torch.mm(v, m.view(v.size(1), -1))
        vm = vm.view(-1, v.size(1))
        vm = self.drop2(vm)
        return vm


    def dot_product(self, v, m, n):
        vm = v * m 
        vm = self.drop3(self.bn1(vm))      
        vm = torch.mm(vm, n.transpose(1,0))
        return vm

    def get_score(self, dot1, dot2, dot3, dot4):
        score = dot1 + dot2 + dot3 - dot4
        score = torch.sigmoid(score)
        return score

    def forward(self, batch_head, batch_rel, init_ind): 
        h = self.EH(batch_head)
        init_emb_H = self.EH(init_ind)
        init_emb_I = self.EI(init_ind)
        out_emb_H, out_emb_I = self.gc1(init_emb_H, init_emb_I, self.adj)
        out_emb_H, out_emb_I = F.relu(out_emb_H), F.relu(out_emb_I)
        out_emb_H, out_emb_I = self.gc2(out_emb_H, out_emb_I, self.adj)
        out_emb_H, out_emb_I = F.relu(out_emb_H), F.relu(out_emb_I)

        final_emb_H, final_emb_I = self.scut_connection(init_emb_H, init_emb_I, out_emb_H, out_emb_I)
        xh, xi, rh, ri = self.get_batch_emb(batch_head, batch_rel, final_emb_H, final_emb_I)

        W_mat_h = self.mat_vec_mul(rh, self.WH)
        W_mat_i = self.mat_vec_mul(ri, self.WI)
        W_mat_j = self.mat_vec_mul(rh, self.WJ)
        W_mat_k = self.mat_vec_mul(ri, self.WK)

        dot1 = self.dot_product(xh, W_mat_h, final_emb_H)
        dot2 = self.dot_product(xh, W_mat_i, final_emb_I)
        dot3 = self.dot_product(xi, W_mat_j, final_emb_I)
        dot4 = self.dot_product(xi, W_mat_k, final_emb_H)
        score = self.get_score(dot1, dot2, dot3, dot4)
        #print('comlexgcn')
        return score


class AblatedGraphConvolutionLayer(torch.nn.Module):

    def __init__(self, in_features, out_features, num_entities):
        super(AblatedGraphConvolutionLayer, self).__init__()

        self.weight = torch.nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = torch.nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def real_conv(self, data):
        support = torch.mm(data, self.weight)
        return support


    def forward(self, data, adj):
        support  = self.real_conv(data)
        output = torch.spmm(adj, support)
        output = output + self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class AblatedModel(torch.nn.Module):
    def __init__(self, d, dim, adj, gcn_layers, drop1, drop2, drop3):
        super(AblatedModel, self).__init__()

        self.adj = adj
        self.gcn_layers = gcn_layers
        self.E, self.R  = self.get_embeddings(d, dim)
        self.W  = self.get_parameters(dim)
        self.gc1 = AblatedGraphConvolutionLayer(dim, dim, len(d.ent))
        self.gc2 = AblatedGraphConvolutionLayer(dim, dim, len(d.ent))
        self.drop1 = torch.nn.Dropout(drop1)
        self.drop2 = torch.nn.Dropout(drop2)
        self.drop3 = torch.nn.Dropout(drop3)
        self.bn0 = torch.nn.BatchNorm1d(dim)
        self.bn1 = torch.nn.BatchNorm1d(dim)
        self.loss = torch.nn.BCELoss()
        
    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def get_embeddings(self, d, dim):

        E = torch.nn.Embedding(len(d.ent), dim)
        R = torch.nn.Embedding(len(d.rel), dim)
        return E, R  


    def get_parameters(self, dim):

        W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (dim, dim)), 
                                    dtype=torch.float, device="cuda", requires_grad=True))
        return W 

    def scut_connection(self, init_emb, out_emb):

        final_emb =  init_emb + out_emb
        return final_emb

    def get_batch_emb(self, batch_head, batch_rel, final_emb):

        x = self.drop1(self.bn0(final_emb[batch_head]))
        r = self.R(batch_rel)
        return x, r 

    def mat_vec_mul(self, v, m):
        vm = torch.mm(v, m.view(v.size(1), -1))
        vm = vm.view(-1, v.size(1))
        vm = self.drop2(vm)
        return vm


    def dot_product(self, v, m, n):
        vm = v * m 
        vm = self.drop3(self.bn1(vm))      
        vm = torch.mm(vm, n.transpose(1,0))
        return vm

    def get_score(self, dot):
        score = torch.sigmoid(dot)
        return score

    def forward(self, batch_head, batch_rel, init_ind): 
        h = self.E(batch_head)
        init_emb = self.E(init_ind)
        out_emb = self.gc1(init_emb, self.adj)
        out_emb = F.relu(out_emb)
        out_emb = self.gc2(out_emb, self.adj)
        out_emb = F.relu(out_emb)
        final_emb = self.scut_connection(init_emb, out_emb)
        x, r = self.get_batch_emb(batch_head, batch_rel, final_emb)
        W_mat = self.mat_vec_mul(r, self.W)
        dot = self.dot_product(x, W_mat, final_emb)
        score = self.get_score(dot)
        #print('ablated')
        return score


class Base:

    def __init__(self, dim, lr, drop1, drop2, drop3, epochs, batch, dr, ls, const, cuda, gcn_layers, model_name):
        self.dim = dim
        self.lr = lr
        self.drop1 = drop1
        self.drop2 = drop2
        self.drop3 = drop3
        self.epochs = epochs
        self.batch = batch
        self.dr = dr
        self.ls = ls
        self.const = const
        self.cuda = cuda
        self.gcn_layers = gcn_layers
        self.model_name = model_name
        ####################################################################################
    def get_ent_idxs(self, ent):
        ent_idxs = {ent[i]:i for i in range(len(ent))}
        return ent_idxs

    def get_rel_idxs(self, rel):
        rel_idxs = {rel[i]:i for i in range(len(rel))}
        return rel_idxs

    def get_adj(self, train_trip_idxs):
        train_trip_idxs_np = np.array(train_trip_idxs, dtype=np.int32)
        adj = sp.coo_matrix((np.ones(train_trip_idxs_np.shape[0]), (train_trip_idxs_np[:, 0], train_trip_idxs_np[:, 2])), shape=(len(d.ent), len(d.ent)), dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = self.normalize(adj + sp.eye(adj.shape[0]))
        adj = self.sparse_mx_to_torch_sparse_tensor(adj).cuda()
        return adj

    def normalize(self, mx):
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def get_model(self, adj):

        if self.model_name.lower() == "complexgcn":
            model = ComplexGCN(d, self.dim, adj, self.gcn_layers, self.drop1, self.drop2, self.drop3).cuda()
        elif self.model_name.lower() == "ablated":
            model = AblatedModel(d, self.dim, adj, self.gcn_layers, self.drop1, self.drop2, self.drop3).cuda()
        #model = ComplexGCN(d, self.dim, adj, self.gcn_layers, self.drop1, self.drop2, self.drop3).cuda()
        #model = AblatedModel(d, self.dim, adj, self.gcn_layers, self.drop1, self.drop2, self.drop3).cuda()
        model.init()
        return model

    def get_opt(self, model):
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        sched = ExponentialLR(opt, self.dr)
        return opt, sched

    def get_init_ind(self):
        init_ind = torch.LongTensor([i for i in range(len(d.ent))]).cuda()
        return init_ind

    def train_head_rel(self, train_batch):
        batch_head = torch.tensor(train_batch[:,0]).cuda()
        batch_rel = torch.tensor(train_batch[:,1]).cuda()
        return batch_head, batch_rel 

    def valid_test_head_rel_tail(self, test_batch):
        batch_head = torch.tensor(test_batch[:,0]).cuda()
        batch_rel = torch.tensor(test_batch[:,1]).cuda()
        batch_tail = torch.tensor(test_batch[:,2]).cuda()
        return batch_head, batch_rel, batch_tail

    def smoothing(self, labels):
        labels = ((self.const-self.ls)*labels) + (self.const/labels.size(1)) 
        return labels

    def filt_scores(self, test_batch, ent_rel_pairs, scores, batch_tail):
        for j in range(test_batch.shape[0]):
            filt = ent_rel_pairs[(test_batch[j][0], test_batch[j][1])]
            label_value = scores[j,batch_tail[j]].item()
            scores[j, filt] = 0.0
            scores[j, batch_tail[j]] = label_value
        return scores

    def sort_scores(self, scores):
        _, sorted_scores = torch.sort(scores, dim=1, descending=True)
        sorted_scores = sorted_scores.cpu().numpy() 
        return sorted_scores

        ####################################################################################
        
    def get_trip_idxs(self, trip):
        trip_idxs = [(self.ent_idxs[trip[i][0]], self.rel_idxs[trip[i][1]], self.ent_idxs[trip[i][2]]) for i in range(len(trip))]
        return trip_idxs
    
    def get_ent_rel_pairs(self, trip):
        ent_rel_pairs = defaultdict(list)
        for t in trip:
            ent_rel_pairs[(t[0], t[1])].append(t[2])
        ent_rel_pairs_keys = list(ent_rel_pairs.keys())
        return ent_rel_pairs, ent_rel_pairs_keys

    def get_train_batch(self, ent_rel_pairs, ent_rel_pairs_keys, ids):
        batch = ent_rel_pairs_keys[ids:ids+self.batch]
        labels = np.zeros((len(batch), len(d.ent)))
        for ids, pair in enumerate(batch):
            labels[ids, ent_rel_pairs[pair]] = 1.
        labels = torch.FloatTensor(labels)
        labels = labels.cuda()
        batch = np.array(batch)
        return batch, labels

    def get_valid_test_batch(self, test_trip_idxs, ids):
        batch = test_trip_idxs[ids:ids+self.batch]
        batch = np.array(batch)
        return batch

    
    def model_valid(self, model, validset, init_ind):
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        valid_trip_idxs = self.get_trip_idxs(validset)
        comb_trip_idxs = self.get_trip_idxs(d.trip)
        ent_rel_pairs, _ = self.get_ent_rel_pairs(comb_trip_idxs)
        
        for i in range(0, len(valid_trip_idxs), self.batch):
            valid_batch = self.get_valid_test_batch(valid_trip_idxs, i)
            batch_head, batch_rel, batch_tail = self.valid_test_head_rel_tail(valid_batch)
            scores = model.forward(batch_head, batch_rel, init_ind)
            scores = self.filt_scores(valid_batch, ent_rel_pairs, scores, batch_tail)
            sorted_scores = self.sort_scores(scores)

            for j in range(valid_batch.shape[0]):
                rank = np.where(sorted_scores[j]==batch_tail[j].item())[0][0]
                ranks.append(rank+1)

                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)

        print('h@10: {0}'.format(np.mean(hits[9])), 'h@3: {0}'.format(np.mean(hits[2])), 'h@1: {0}'.format(np.mean(hits[0])), 'MRR: {0}'.format(np.mean(1./np.array(ranks))))


    def model_train(self):
        print("Training...", self.model_name)
        self.ent_idxs = self.get_ent_idxs(d.ent)
        self.rel_idxs = self.get_rel_idxs(d.rel)
        train_trip_idxs = self.get_trip_idxs(d.trainset)
        adj = self.get_adj(train_trip_idxs)
        model = self.get_model(adj)
        opt, sched = self.get_opt(model)
        ent_rel_pairs, ent_rel_pairs_keys = self.get_ent_rel_pairs(train_trip_idxs)
        init_ind = self.get_init_ind() 
        print("Starting training...")

        for epoch in range(1, self.epochs+1):
            train_time = time.time()
            model.train() 
            losses = []   
            np.random.shuffle(ent_rel_pairs_keys)
            for j in range(0, len(ent_rel_pairs_keys), self.batch):
                train_batch, labels = self.get_train_batch(ent_rel_pairs, ent_rel_pairs_keys, j)
                opt.zero_grad()
                batch_head, batch_rel = self.train_head_rel(train_batch)  
                scores = model.forward(batch_head, batch_rel, init_ind)
                labels = self.smoothing(labels)           
                loss = model.loss(scores, labels)
                loss.backward()
                opt.step()
                losses.append(loss.item())
                sched.step()
            print('epoch:', epoch)
            print('seconds:', time.time()-train_time)
            print('loss:', np.mean(losses))    
            model.eval()
            with torch.no_grad():
                print("Validation:")
                self.model_valid(model, d.validset, init_ind)
                if not epoch%2:
                    print("Test:")
                    self.model_valid(model, d.testset, init_ind)
           

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments')
    parser.add_argument('-dataset',  type=str, default='FB15k-237')
    parser.add_argument('-model_name',  type=str, default='complexgcn')
    parser.add_argument('-dim', type=int, default=200)
    parser.add_argument('-lr', type=float, default=0.0005)
    parser.add_argument('-drop1', type=float, default=0.3)
    parser.add_argument('-drop2',  type=float, default=0.4)
    parser.add_argument('-drop3',  type=float, default=0.5)
    parser.add_argument('-epochs', type=int, default=500)
    parser.add_argument('-batch', type=int, default=128)
    parser.add_argument('-dr', type=float, default=1.0)
    parser.add_argument('-ls',  type=float, default=0.1)
    parser.add_argument('-seed',  type=int, default=20)
    parser.add_argument('-const',  type=int, default=1.0)
    parser.add_argument('-cuda', type=bool, default=True)
    parser.add_argument('-gcn_layers', type=int, default=2)

    args = parser.parse_args()
    torch.backends.cudnn.deterministic = True 
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(args.seed)
    path = "data/%s/" % args.dataset
    d = Dataset(path=path)
    base = Base(args.dim, args.lr, args.drop1, args.drop2, args.drop3, args.epochs, args.batch, args.dr, args.ls, args.const, args.cuda, args.gcn_layers, args.model_name)
    base.model_train()
                

