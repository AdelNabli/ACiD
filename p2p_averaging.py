import time
import torch
import numpy as np
import torch.distributed as dist
from acid_utils import acid_ode


def do_send(
    params_com,
    params_other_worker,
    process_group,
    other_rank,
    apply_acid,
    params_com_tilde,
    ode_matrix,
    t_last_spike,
    delta_t_grad,
    beta_tilde,
):
    """
    The send THEN receive function.
    Expects that the peer with whom we communicate runs the symetric function
    receive THEN send.
    """
   
    # sends and receives the params to and from an other worker
    dist.send(params_com, other_rank, process_group)
    dist.recv(params_other_worker, other_rank, process_group)
    if apply_acid:
        # retrieve the times
        t_old  = t_last_spike.value
        t_new = time.time()
        # apply continuous momentum
        acid_ode(params_com, params_com_tilde, ode_matrix, t_old, t_new, delta_t_grad.value)
        # update the t spike var
        t_last_spike.value = t_new
        # update params_com_tilde
        params_com_tilde.add_(beta_tilde * (params_other_worker - params_com))
    # inplace average of parameters
    params_com.lerp_(params_other_worker, 0.5)


def do_recv(
    params_com,
    params_other_worker,
    process_group,
    other_rank,
    apply_acid,
    params_com_tilde,
    ode_matrix,
    t_last_spike,
    delta_t_grad,
    beta_tilde,
):
    """
    The receive THEN send function.
    Expects that the peer with whom we communicate runs the symetric function
    send THEN receive.
    """
   
    # receives and sends the params to and from an other worker
    dist.recv(params_other_worker, other_rank, process_group)
    dist.send(params_com, other_rank, process_group)
    if apply_acid:
        # retrieve the times
        t_old  = t_last_spike.value
        t_new = time.time()
        # apply continuous momentum
        acid_ode(params_com, params_com_tilde, ode_matrix, t_old, t_new, delta_t_grad.value)
        # update the t spike var
        t_last_spike.value = t_new
        # update params_com_tilde
        params_com_tilde.add_(beta_tilde * (params_other_worker - params_com))
    # inplace average of parameters
    params_com.lerp_(params_other_worker, 0.5)

    
def gossip_process(rank,
                   local_rank,
                   world_size,
                   rank_other,
                   params_com,
                   params_com_other,
                   barrier_sync_averaging,
                   continue_grad_routine,
                   barrier_end_init,
                   barrier_com_grad,
                   log,
                   com_history,
                   count_grads_local,
                   count_coms_local,
                   rate_com,
                   apply_acid,
                   params_com_tilde,
                   ode_matrix,
                   t_last_spike,
                   delta_t_grad,
                   beta_tilde,
                  ):
    """
    Gossip routine for the p2p averaging of the model's parameters.
    """
    # initialize the process group for communications
    process_group = dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size
    )
    # initialize model weights by performing a first all-reduce
    torch.cuda.synchronize()
    dist.all_reduce(params_com, group=process_group, op=dist.ReduceOp.SUM)
    params_com.mul_(1 / world_size)
    # initialize the right momentum variable
    if apply_acid:
        # initialize the momentum variable
        params_com_tilde.copy_(params_com)
    # signal the end of the initialization to the main process
    barrier_end_init.wait()
    # create the gossip stream
    gossip_stream = torch.cuda.Stream(device=local_rank)
    count_coms_next_wait = 1

    # we do everything in the gossip stream
    with torch.cuda.stream(gossip_stream):
        while True:
            rank_other_here = rank_other.value
            # wait the rank of an other available worker
            while rank_other_here == -1:
                rank_other_here = rank_other.value
            # rank_other is equal to -2 when we made enough grad steps in total
            # so there is no need to communicate anymore
            if rank_other_here == -2:
                barrier_sync_averaging.abort()
                break
            # averaging with rank_other
            if rank_other_here < rank:
                do_send(params_com,
                        params_com_other,
                        process_group,
                        rank_other_here,
                        apply_acid,
                        params_com_tilde,
                        ode_matrix,
                        t_last_spike,
                        delta_t_grad,
                        beta_tilde,
                       )
            else:
                do_recv(params_com,
                        params_com_other,
                        process_group,
                        rank_other_here,
                        apply_acid,
                        params_com_tilde,
                        ode_matrix,
                        t_last_spike,
                        delta_t_grad,
                        beta_tilde,
                       )
            # logs the communication
            count_coms_local.value += 1
            count_com_rank = com_history[rank_other_here]
            count_com_rank.value += 1
            # wait or synchoronize with the grad process
            if rate_com >= 1:
                if count_coms_local.value >= count_coms_next_wait:
                    # Wait for 1 averaging step before grad
                    barrier_com_grad.wait()
                    barrier_com_grad.reset()
                    # use poisson law to implement the Poisson Point Processes for communications
                    count_coms_next_wait += np.random.poisson(lam=rate_com, size=None)
            else:
                barrier_com_grad.wait()
            # re-initialize the mp.Value var for next round
            rank_other.value = -1
            # signal to the synchronization process we are available for communication
            try:
                barrier_sync_averaging.wait()
                t_beg_com = time.time()
            except:
                # only way this fails is barrier already broken by sync process
                break
        # signal to grad process to stop
        continue_grad_routine.value = 0
        try:
            barrier_com_grad.abort()
        except:
            pass
        # alll reduce the params at the end of the training
        dist.barrier(group=process_group)
        torch.cuda.synchronize()
        dist.all_reduce(params_com, group=process_group, op=dist.ReduceOp.SUM)
        params_com.mul_(1 / world_size)
    