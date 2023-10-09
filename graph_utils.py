import scipy
import numpy as np
import networkx as nx


class Graph(object):
    """
    Base Graph class.
    The purpose of this class is the "next_communication" function,
    to "know" in which order the communications will happen in certain graph classes.
    This is to compare to previous work such as AD-PSGD who considered bipartite graphs
    with pre-designed communication schedule.
    """
    
    def __init__(self, world_size):
        self.world_size = world_size
        # create a dictionnary to store all the nodes attributes
        self.nodes = dict()
        for i in range(world_size):
            # initialize an attribute dictionary at each node
            self.nodes[i] = dict()
            # a list of its neighbors
            self.nodes[i]["N_i"] = self.create_cycle_neighbors(i)
            # a boolean indicating if it is ready to communicate
            self.nodes[i]["is_ready_2_com"] = False
            # a count of the node's communications
            self.nodes[i]["count_iter"] = 0
        # assumes a regular graph
        self.len_cycle = len(self.nodes[0]["N_i"])
    
    def next_communication(self, rank):
        """
        returns the rank of the neighbor we are supposed to communicate with,
        and prepare the next communication.
        """
        # cycles through the list of the neighbors. Thus, suppose that the list is in the "right" order
        # to avoid deadlocks, highlighting the importance of the "create_cycle_neighbors" method.
        other_rank = self.nodes[rank]["N_i"][self.nodes[rank]["count_iter"] % self.len_cycle]
        self.nodes[rank]["count_iter"] += 1
        
        return other_rank
    
    def create_cycle_neighbors(self, rank):
        """
        Returns a list of neighbors rightly ordered according to rank and the graph's topology.
        """
        raise NotImplementedError
        

class CycleGraph(Graph):
    """
    In the cycle graph, if their rank is even,
    nodes will communicate first to the right and then to the left,
    the inverse if it is odd.
    """  
    def create_cycle_neighbors(self, rank):

        if rank%2 :
            return [(rank + 1)%self.world_size, (rank - 1)%self.world_size]
        else:
            return [(rank - 1)%self.world_size, (rank + 1)%self.world_size]
        

class ExponentialGraph(Graph):
    """
    In the exponential graph, if our rank is even, 
    first we communicate with ranks that are in the 2^k below us, and then 2^k above us,
    if it is odd, the opposite.
    Assumes that world_size is a power of 2.
    """
    
    def create_cycle_neighbors(self, rank):
        # compute the size of the neighborhood in an exponential graph.
        log_n = int(np.log2(self.world_size))
        # init the list to return.
        N_rank = []
        # cycles through the ranks both times,
        # once clockwise, and the other anti clockwise.
        for k in range(log_n):
            if rank%2:
                N_rank.append((rank + 2**k)%self.world_size)
            else:
                N_rank.append((rank - 2**k)%self.world_size)
        for k in range(log_n):
            if rank%2:
                N_rank.append((rank - 2**k)%self.world_size)
            else:
                N_rank.append((rank + 2**k)%self.world_size)
        return N_rank


def compute_graph_resistance(L, G):
    """
    Compute the graph's resistance using the Laplacian L.
    The max effective resistance of the graph is defined as 
    $\max_{(i,j) \in E} \frac{1}{2} (e_i - e_j)^\top L^+ (e_i - e_j)
    
    Parameters:
        - L (np.array): A Laplacian matrix of G.
        - G (nx.graph): A NetworkX graph.
    Returns:
        - R_max (float): the worst case resistance between two edges.
    """

    n = len(L)
    # compute the pseudo inverse of L
    L_inv = scipy.linalg.pinv(L)
    R_max = 0
    e_blank = np.zeros(n)
    # Compute the resistance of each edgge
    for (i, j) in G.edges:
        e_ij = e_blank.copy()
        e_ij[i] = 1
        e_ij[j] = -1
        R_ij = 0.5 * e_ij.T @ L_inv @ e_ij
        # save the worst case resistance
        if R_ij > R_max:
            R_max = R_ij

    return R_max