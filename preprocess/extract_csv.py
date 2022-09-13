import numpy as np
import pandas as pd
import pickle
import scipy
import scipy.sparse as sps
import os

def preprocess(data_name):
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []
    
    with open(data_name) as f:
        s = next(f)
        print(s)
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            u = int(e[0])
            i = int(e[1])
            ts = float(e[2])
            label = int(e[3])
            
            feat = np.array([float(x) for x in e[4:]])
            
            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)
            
            feat_l.append(feat)
    return pd.DataFrame({'u': u_list, 
                         'i':i_list, 
                         'ts':ts_list, 
                         'label':label_list, 
                         'idx':idx_list}), np.array(feat_l)

def reindex(df):
    assert(df.u.max() - df.u.min() + 1 == len(df.u.unique()))
    assert(df.i.max() - df.i.min() + 1 == len(df.i.unique()))
    
    upper_u = df.u.max() + 1
    new_i = df.i + upper_u
    
    new_df = df.copy()

    print(new_df.u.max())
    print(new_df.i.max())
    print(len(new_df))

    new_df.i = new_i
    # new_df.u += 1
    # new_df.i += 1
    # new_df.idx += 1
    
    return new_df

def edge_list_to_coo_matrix(edge_lists):
    adj_time_lists = []
    for t in range(len(edge_lists)):
        row, col, data = [], [], []
        for u, i in edge_lists[t]:
            row.append(int(u))
            row.append(int(i))
            col.append(int(i))
            col.append(int(u))
            data.append(1)
            data.append(1)
        adj_time_lists.append(sps.coo_matrix((data, (row, col))))
    return adj_time_lists

def prepare_edge_list(df, time_step):
    edge_lists = [[] for i in range(time_step)]
    t_max, t_min = df.ts.max(), df.ts.min()
    t_delta = (t_max - t_min) / time_step + 1e-6
    num_node = max(df.u.max(), df.i.max()) + 1
    adj_orig_dense_list = np.zeros((time_step, num_node, num_node))
    print(adj_orig_dense_list.shape)
    for _, row in df.iterrows():
        t = int((row.ts - t_min) / t_delta)
        edge_lists[t].append((row.u, row.i))
        adj_orig_dense_list[t, int(row.u), int(row.i)] = 1
    adj_time_lists = edge_list_to_coo_matrix(edge_lists)
    return adj_time_lists, adj_orig_dense_list

def run(data_name):
    PATH = '../raw_data/{0}/{1}.csv'.format(data_name, data_name)
    OUT_PATH = '../data/{}'.format(data_name)
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)
    OUT_EDGE_LIST = os.path.join(OUT_PATH, 'adj_time_list.pickle')
    OUT_DENSE_DEGE_LIST = os.path.join(OUT_PATH, 'adj_orig_dense_list.pickle')
    
    df, feat = preprocess(PATH)
    new_df = reindex(df)
    adj_time_lists, adj_orig_dense_list = prepare_edge_list(new_df, time_step = 16)
    with open(OUT_EDGE_LIST, 'wb') as f:
        pickle.dump(adj_time_lists, f)
    with open(OUT_DENSE_DEGE_LIST, 'wb') as f:
        pickle.dump(adj_orig_dense_list, f)

if __name__ == '__main__':
    run('reddit')
    run('wikipedia')


