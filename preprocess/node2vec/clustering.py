import os
import pickle
import numpy as np
from sklearn.cluster import KMeans

def load_data(data_name, data_path, test_len = 3): 
    adj_time_list_path = os.path.join(data_path, data_name, "adj_orig_dense_list.pickle")
    with open(adj_time_list_path, 'rb') as handle:
        adj_time_list = pickle.load(handle,encoding="bytes")
    edge_list = set()
    for i in range(len(adj_time_list) - test_len):
        for j in range(len(adj_time_list[i])):
            for k in range(len(adj_time_list[i][j])):
                edge_list.add((str(i) + ' ' + str(k)))
    with open(data_name + ".edgelist", 'w') as handle:
        for item in edge_list:
            handle.write(item + '\n')
        handle.close()

def cluster(data_name, n_class):
    with open("emb/{}.emb".format(data_name), 'b') as f:
        n_nodes, emb_dim = f.readline().split()
        emb = np.zeros((int(n_nodes), int(emb_dim)))
        for line in f:
            line = line.strip().split()
            emb[int(line[0])] = np.array(line[1:])
    
    kmeans = KMeans(n_clusters=n_class, random_state=0).fit(emb)
    label = kmeans.labels_
    assert len(label) == len(emb)
    return label, emb
    
    return label, emb
def main(data_name, data_path, emb_dim = 32, n_class = 3):
    load_data(data_name, data_path)
    os.system("python2 src/main.py --input graph/{}.edgelist --output emb/{}.emb ----dimensions {}".format(data_name, data_name, emb_dim))
    label, emb = cluster(data_name, n_class)
    np.save(os.path.join(data_path, data_name, "label.npy"), label)
    np.save(os.path.join(data_path, data_name, "feat.npy"), emb)

data_name = 'fb'
data_path = '../../data'
main(data_name, data_path)