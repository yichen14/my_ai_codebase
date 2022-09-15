import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import math
from collections import OrderedDict

# Adapt from:
# https://github.com/iHeartGraph/Euler/blob/main/benchmarks/models/evo_gcn.py
# https://github.com/IBM/EvolveGCN

class EGCN_O(torch.nn.Module):
    def __init__(self, args, activation, device=0, skipfeats=False):
        super(EGCN_O, self).__init__()
        GRCU_args = Namespace({})

        feats = [args.feats_per_node,
                 args.layer_1_feats,
                 args.layer_2_feats]
        self.device = device
        self.skipfeats = skipfeats
        self.GRCU_layers = []
        self.params = nn.ParameterList()
        for i in range(1,len(feats)):
            GRCU_args = Namespace({'in_feats' : feats[i-1],
                                     'out_feats': feats[i],
                                     'activation': activation})

            grcu_i = GRCU(GRCU_args)
            #print (i,'grcu_i', grcu_i)
            self.GRCU_layers.append(grcu_i.to(self.device))
            self.params.extend(list(self.GRCU_layers[-1].parameters()))

    def parameters(self):
        return self.params

    def forward(self,A_list, Nodes_list,nodes_mask_list):
        node_feats= Nodes_list[-1]

        for unit in self.GRCU_layers:
            Nodes_list = unit(A_list,Nodes_list)#,nodes_mask_list)

        out = Nodes_list[-1]
        if self.skipfeats:
            out = torch.cat((out,node_feats), dim=1)   # use node_feats.to_dense() if 2hot encoded input 
        return out


class GRCU(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        cell_args = Namespace({})
        cell_args.rows = args.in_feats
        cell_args.cols = args.out_feats

        self.evolve_weights = mat_GRU_cell(cell_args)

        self.activation = self.args.activation
        self.GCN_init_weights = Parameter(torch.Tensor(self.args.in_feats,self.args.out_feats))
        self.reset_param(self.GCN_init_weights)

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)

    def forward(self,A_list,node_embs_list):#,mask_list):
        GCN_weights = self.GCN_init_weights
        out_seq = []
        for t,Ahat in enumerate(A_list):
            node_embs = node_embs_list[t]
            #first evolve the weights from the initial and use the new weights with the node_embs
            GCN_weights = self.evolve_weights(GCN_weights)#,node_embs,mask_list[t])
            node_embs = self.activation(Ahat.matmul(node_embs.matmul(GCN_weights)))

            out_seq.append(node_embs)

        return out_seq

class mat_GRU_cell(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.update = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Sigmoid())

        self.reset = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Sigmoid())

        self.htilda = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Tanh())
        
        self.choose_topk = TopK(feats = args.rows,
                                k = args.cols)

    def forward(self,prev_Q):#,prev_Z,mask):
        # z_topk = self.choose_topk(prev_Z,mask)
        z_topk = prev_Q

        update = self.update(z_topk,prev_Q)
        reset = self.reset(z_topk,prev_Q)

        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)

        new_Q = (1 - update) * prev_Q + update * h_cap

        return new_Q

        

class mat_GRU_gate(torch.nn.Module):
    def __init__(self,rows,cols,activation):
        super().__init__()
        self.activation = activation
        #the k here should be in_feats which is actually the rows
        self.W = Parameter(torch.Tensor(rows,rows))
        self.reset_param(self.W)

        self.U = Parameter(torch.Tensor(rows,rows))
        self.reset_param(self.U)

        self.bias = Parameter(torch.zeros(rows,cols))

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)

    def forward(self,x,hidden):
        out = self.activation(self.W.matmul(x) + \
                              self.U.matmul(hidden) + \
                              self.bias)

        return out

class TopK(torch.nn.Module):
    def __init__(self,feats,k):
        super().__init__()
        self.scorer = Parameter(torch.Tensor(feats,1))
        self.reset_param(self.scorer)
        
        self.k = k

    def reset_param(self,t):
        #Initialize based on the number of rows
        stdv = 1. / math.sqrt(t.size(0))
        t.data.uniform_(-stdv,stdv)

    def forward(self,node_embs,mask):
        scores = node_embs.matmul(self.scorer) / self.scorer.norm()
        scores = scores + mask

        vals, topk_indices = scores.view(-1).topk(self.k)
        topk_indices = topk_indices[vals > -float("Inf")]

        if topk_indices.size(0) < self.k:
            topk_indices = pad_with_last_val(topk_indices,self.k)
            
        tanh = torch.nn.Tanh()

        if isinstance(node_embs, torch.sparse.FloatTensor) or \
           isinstance(node_embs, torch.cuda.sparse.FloatTensor):
            node_embs = node_embs.to_dense()

        out = node_embs[topk_indices] * tanh(scores[topk_indices].view(-1,1))

        #we need to transpose the output
        return out.t()

class Namespace(object):
    '''
    helps referencing object in a dictionary as dict.key instead of dict['key']
    '''
    def __init__(self, adict):
        self.__dict__.update(adict)

def pad_with_last_val(vect,k):
    device = 'cuda' if vect.is_cuda else 'cpu'
    pad = torch.ones(k - vect.size(0),
                         dtype=torch.long,
                         device = device) * vect[-1]
    vect = torch.cat([vect,pad])
    return vect

from types import SimpleNamespace as SN


class LP_EGCN_o(EGCN_O):
    def __init__(self, x_dim, device, h_dim = 32, z_dim = 16, inner_prod=True):
        # Why do they insist on doing it this way. Fixing it
        args = SN(
            feats_per_node=x_dim,
            layer_1_feats=h_dim,
            layer_2_feats=z_dim
        )

        # RReLU is default in their experiments, keeping it here
        super(LP_EGCN_o, self).__init__(args, torch.nn.RReLU(), device=device)

        # This is how the paper does it, but the experiments show
        # it's not as effective for LP as using inner prod decoding
        if not inner_prod:
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(z_dim*2, 1),
                torch.nn.Sigmoid()
            )
        else:
            self.classifier = None
        

    def forward(self,A_list, Nodes_list):
        '''
        Overriding their forward method to return all timesteps
        instead of just the last one
        '''
        torch.cuda.empty_cache()
        for unit in self.GRCU_layers:
            Nodes_list = unit(A_list,Nodes_list)

        return Nodes_list

    # Copied from Euler
    def calc_loss(self, t_scores, f_scores):
        EPS = 1e-6
        pos_loss = -torch.log(t_scores+EPS).mean()
        neg_loss = -torch.log(1-f_scores+EPS).mean()

        return pos_loss + neg_loss

    def decode(self, src, dst, z):
        if not self.classifier is None:
            catted = torch.cat([z[src], z[dst]], dim=1)
            return self.classifier(catted)
        else:
            dot = (z[src] * z[dst]).sum(dim=1)
            return torch.sigmoid(dot)

    def loss_fn(self, ts, fs, zs):
        tot_loss = torch.zeros((1)).to(self.device)
        T = len(ts)

        for i in range(T):
            t_src, t_dst = ts[i]
            fs[i] = fs[i].T
            f_src, f_dst = fs[i]
            z = zs[i]
            
            tot_loss += self.calc_loss(
                self.decode(t_src, t_dst, z),
                self.decode(f_src, f_dst, z)
            )   
        return tot_loss.true_divide(T)

    def score_fn(self, ts, fs, zs):
        tscores = []
        fscores = []

        T = len(ts)

        for i in range(T):
            t_src, t_dst = ts[i]
            fs[i] = fs[i].T
            f_src, f_dst = fs[i]
            z = zs[i]

            tscores.append(self.decode(t_src, t_dst, z))
            fscores.append(self.decode(f_src, f_dst, z))

        tscores = torch.cat(tscores, dim=0)
        fscores = torch.cat(fscores, dim=0)

        return tscores, fscores
