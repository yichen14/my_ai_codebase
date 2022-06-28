import __init_lib_path
import dataset
import argparse
import numpy as np
import torch

def main():
    data= dataset.static_graph()
    A = data.get_A_matrix("election")
    print(A)
    pass

if __name__ == '__main__':
    main()