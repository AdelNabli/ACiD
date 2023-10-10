import os
import torch
import random
import numpy as np
import torch.distributed as dist
import multiprocessing as mp
from multiprocessing import Process, Manager
from graph_utils import ExponentialGraph, CycleGraph

def sync_process(rank, world_size, rank_other, new_grads, barrier_sync_averaging, log):
    # initialize the process group for rank communications
    process_group = dist.init_process_group(
        backend="gloo", init_method='tcp://'+os.environ["MASTER_ADDR"]+':'+str(int(os.environ["MASTER_PORT"])+1),
        rank=rank, world_size=2*world_size + 1
    )
    
    while True:
        # initialize a tensor to send master process
        # use the number of grad steps done by worker rank since last communication as message
        # we use the same tensor as placeholder to receive the other rank
        tensor_other_rank = torch.ones(1)*new_grads.value
        # send a tensor to master to signal worker nb rank is available to communicate
        dist.send(tensor_other_rank, rank+world_size, process_group)
        # re-initialize the new_grads value
        new_grads.value = new_grads.value - int(tensor_other_rank.data)
        # receive the rank from the last process in the group
        dist.recv(tensor_other_rank, 2*world_size, process_group)
        # changes the rank value localy saved in the mp.Value variable
        rank_other.value = int(tensor_other_rank.data)
        if rank_other.value == -2:
            # signal to the listening process to kil the process
            dist.send(tensor_other_rank, rank+world_size, process_group)
            barrier_sync_averaging.abort()
            break
        # wait for the p2p averaging
        barrier_sync_averaging.wait()
        barrier_sync_averaging.reset()


def listen_given_rank(rank, world_size, queue, nb_tot_grad_so_far, lock, log):
    # initialize the process group for rank communications
    process_group = dist.init_process_group(
        backend="gloo", init_method='tcp://'+os.environ["MASTER_ADDR"]+':'+str(int(os.environ["MASTER_PORT"])+1),
        rank=rank+world_size, world_size=2*world_size + 1
    )
    tensor_other_rank = torch.zeros(1)
    while tensor_other_rank.data != -2:
        # receive information that worker rank is available for communications
        dist.recv(tensor_other_rank, rank, process_group)
        lock.acquire()
        nb_tot_grad_so_far.value += int(tensor_other_rank.data)
        queue.put(rank)
        lock.release()

    
def master_process(world_size, nb_grad_tot_goal, log, graph_topology, deterministic_neighbor):
    # initialize a mp Manager
    #with Manager() as manager:
    queue = mp.Queue()
    lock = mp.Lock()
    nb_tot_grad_so_far = mp.Value('i', 0)
    list_processes = []
    for rank in range(world_size):
        listen_process = Process(target=listen_given_rank, args=(rank, world_size, queue, nb_tot_grad_so_far, lock, log))
        listen_process.start()
        list_processes.append(listen_process)
    # initialize the process group for rank communications
    process_group = dist.init_process_group(
        backend="gloo", init_method='tcp://'+os.environ["MASTER_ADDR"]+':'+str(int(os.environ["MASTER_PORT"])+1),
        rank=2*world_size, world_size=2*world_size + 1
    )
    tuple_of_ranks = []
    tensor_rank_0 = torch.zeros(1)
    tensor_rank_1 = torch.zeros(1)
    if graph_topology != "complete":
        # init a graph object
        if graph_topology == "cycle":
            G = CycleGraph(world_size, deterministic_neighbor)
        elif graph_topology == "exponential":
            G = ExponentialGraph(world_size, deterministic_neighbor)
    # while the total number of grad is not reached
    while nb_tot_grad_so_far.value < nb_grad_tot_goal:
        if graph_topology == "complete":
            # get the rank of an available worker
            tuple_of_ranks.append(queue.get())
            # if 2 workers are available for communication
            if len(tuple_of_ranks) == 2:
                # gather their ranks
                tensor_rank_0[0] = tuple_of_ranks[0]
                tensor_rank_1[0] = tuple_of_ranks[1]
                # send their ranks to each other
                dist.send(tensor_rank_0, tuple_of_ranks[1], process_group)
                dist.send(tensor_rank_1, tuple_of_ranks[0], process_group)
                # re-initialize the tuples as an empty one
                tuple_of_ranks = []
        else:
            # get the rank of an available worker
            i = queue.get()
            # init a neighbor var
            j_ready = None
            # gather the next ranks i is supposed to communicate with
            # if deterministic_neighbor, then the list contains only 1 item.
            # else, will go through all the neighbors in a random order.
            N_i = G.next_communication(i)
            for j in N_i:
                # if j is ready to com
                if G.nodes[j]["is_ready_2_com"]:
                    # put the bool back to False
                    G.nodes[j]["is_ready_2_com"] = False
                    j_ready = j
                    break
            # if the neighbor of i for next step is not ready
            if j_ready is None:
                # we signal that i is ready in the graph
                G.nodes[i]["is_ready_2_com"] = True
            else:
                # we perform the communication between i and j
                tensor_rank_0[0] = i
                tensor_rank_1[0] = j_ready
                # send their ranks to each other
                dist.send(tensor_rank_0, j_ready, process_group)
                dist.send(tensor_rank_1, i, process_group)

    # when we go out of the while loop, send to everybody the message to stop processes
    kill_process_signal = torch.ones(1)*(-2)
    for rank in range(world_size):
        dist.send(kill_process_signal, rank, process_group)

    # terminates all processes
    for p in list_processes:
        p.join()
