import pickle

class static_graph():
    def __init__(self) -> None:
        # load data
        DATA_PATH = '/home/ruijiew2/Documents/RandomGraph/data/static/'
        DATA_NAME = ['bill', 'election', 'timme']
        self.data_dict = {}
        for name in DATA_NAME:
            with open(DATA_PATH + name+".pickle", 'rb') as f:
                data = pickle.load(f)
                self.data_dict[name] = data.toarray()
        
        #TODO: split dataset

        pass

    def get_A_matrix(self, name):
        """
        Return the adjacency matrix 
        Args:
            name: file name
        """
        return self.data_dict[name]
    


    