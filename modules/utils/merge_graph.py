import torch
from tqdm import tqdm

def merge_graph(snapshots):
    """
    Merge time snapshots to one single graph
    
    Args:
        snapshots(list[matrix]): the list of adjacency matrices need to be merge
    
    Return:
        adj_matrix(tensor): merged adjacency matrix.
    """
    adj_matrix = torch.zeros_like(snapshots[0])
    for matrix in tqdm(snapshots):
        adj_matrix = torch.logical_or(adj_matrix, matrix)
    adj_matrix = adj_matrix.type(torch.float32)
    return adj_matrix