from utils import *

graph = nx.read_edgelist(file_path, create_using=nx.Graph(), nodetype=int)
# nodes = list(np.where(state == 0)[0])