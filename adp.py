import time
import torch
import numpy as np
import torch.nn as nn
import multiprocessing as mp
from p2p_averaging import gossip_process
from p2p_sync import sync_process, master_process
from utils.graph_utils import compute_acid_constants
from utils.acid_utils import init_momentum_var, load_momentum, acid_ode


class ADP(nn.Module):
    """
    A wrapper around the model, with added functions to handle asynchronous communications.
    """

    def __init__(
        self,
        model,
        rank,
        local_rank,
        world_size,
        nb_grad_tot_goal,
        log,
        rate_com,
        apply_acid,
        criterion,
        optimizer,
        data_iterator,
        momentum,
        dataset_name,
        graph_topology,
        deterministic_com,
        deterministic_neighbor,
    ):
        super().__init__()

        # Check for argument consistency.
        # first, verify that when we are applying the deterministic algo,
        # we make a rate_com nb of com for each grad step that is integer.
        if int(rate_com) != rate_com and deterministic_com:
            raise ValueError(
                "A non integer number of communications has been set in a non stochastic setting."
            )

        self.module = model
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.nb_grad_tot_goal = nb_grad_tot_goal
        self.log = log
        self.rate_com = rate_com
        self.graph_topology = graph_topology
        self.deterministic_com = deterministic_com
        self.deterministic_neighbor = deterministic_neighbor
        # loads the model parameters in share memory so that both grad and com processes edit the same tensor
        params = self.get_weights()
        params = params.to(self.local_rank)
        # the communication params are in shared memory
        # so that both communication and gradient processes have access to it
        # for cuda tensor, this is not required
        self.params_com = params.share_memory_()
        # placeholder to receive the params in the p2p averaging
        self.params_com_other = params.detach().clone()
        # load the shared memory param so that grad steps are directly done on the shared params
        self.set_weights(self.params_com)

        # mp Variables
        self.rank_other = mp.Value("i", -1)
        self.new_grads = mp.Value(
            "i", 0
        )  # nb of new grads between two comms to the master process
        self.count_grads_local = mp.Value("i", 0)  # count of local grad steps
        self.count_coms_local = mp.Value("i", 0)
        self.continue_grad_routine = mp.Value("i", 1)
        self.count_grads_next_wait = 0
        self.barrier_sync_averaging = mp.Barrier(2)
        self.barrier_end_init = mp.Barrier(2)
        self.barrier_com_grad = mp.Barrier(2)
        # initialize the comm logs
        self.com_history = [mp.Value("i", 0) for k in range(self.world_size)]

        # init acid variables
        self.apply_acid = apply_acid
        self.init_acid(criterion, optimizer, data_iterator, momentum, dataset_name)

        # launch the communication processes
        self.launch_sync_process()
        self.launch_gossip_process()
        # if rank is 0, launch the master process to sync the p2p communications
        if rank == 0:
            self.launch_master_sync_process()
        # wait that all parameters have been initialized with an all-reduce in the gossip process
        self.barrier_end_init.wait()

    def init_acid(self, criterion, optimizer, data_iterator, momentum, dataset_name):
        # if we apply acid, initialize the specific variables
        if self.apply_acid:
            # initialize a momentum variable
            self.params_com_tilde = self.params_com.detach().clone().share_memory_()
            # compute ACiD momentum constants
            (eta, self.beta_tilde) = compute_acid_constants(
                self.graph_topology, self.world_size, self.rate_com
            )
            # init the ode mixing matrix
            self.ode_matrix = torch.zeros((2, 2)).double()
            self.ode_matrix[0, 0] = -eta
            self.ode_matrix[0, 1] = eta
            self.ode_matrix[1, 0] = eta
            self.ode_matrix[1, 1] = -eta
            # init the variable storing the time that takes to perform a forward-backward pass
            self.delta_t_grad = mp.Value("d", 1)
            # load the momentum variable
            self.mom_vec = init_momentum_var(
                self.module,
                criterion,
                optimizer,
                data_iterator,
                self.local_rank,
                self.delta_t_grad,
                momentum,
                dataset_name,
            )
            # load the momentum into shared memory
            load_momentum(self.mom_vec, self.module, optimizer, momentum)
            # init the variable keeping in memory the time of the last last "event" (be it a grad or averaging step)
            self.t_last_spike = mp.Value("d", time.time())
        else:
            self.params_com_tilde = None
            self.ode_matrix = None
            self.t_last_spike = None
            self.delta_t_grad = None
            self.beta_tilde = None

    def forward(self, *args, **kwargs):
        """
        Perform a forward pass.
        """
        self.new_grads.value += 1
        return self.module(*args, **kwargs)

    def start(self):
        self.t_beg_grad = time.time()

    def step(self, optimizer, scheduler, normalize_grads):
        # if apply acid, update the momentum variable
        if self.apply_acid:
            # update the time variables used to update the exponential moving average for delta_t_grad and for the spike
            # update the time variables for the spike
            t_new = time.time()
            t_old = self.t_last_spike.value
            # perform the mixing beween the 2 variables
            acid_ode(
                self.params_com,
                self.params_com_tilde,
                self.ode_matrix,
                t_old,
                t_new,
                self.delta_t_grad.value,
            )
            # update the spike time var
            self.t_last_spike.value = t_new

        if normalize_grads:
            nn.utils.clip_grad_norm_(self.module.parameters(), 1.0)
        optimizer.step()

        # take a grad step on the momentum var
        if self.apply_acid:
            lr = optimizer.param_groups[0]["lr"]
            # perform the grad step for params tilde
            self.params_com_tilde.add_(self.mom_vec, alpha=-lr)
            # update delta_t_grad
            delta_t_grad = time.time() - self.t_beg_grad
            self.delta_t_grad.value = (
                0.5 * delta_t_grad + (1 - 0.5) * self.delta_t_grad.value
            )

        scheduler.step()
        # wait or synchronize with the com process
        if self.rate_com < 1:
            if self.count_grads_local.value >= self.count_grads_next_wait:
                # Wait for 1 averaging step before grad
                try:
                    self.barrier_com_grad.wait()
                    self.barrier_com_grad.reset()
                except:
                    pass
                self.count_grads_next_wait += np.random.poisson(
                    lam=1 / self.rate_com, size=None
                )
        else:
            try:
                self.barrier_com_grad.wait()
            except:
                # only way the barrier fails is that it is already aborted by the communication process
                pass
        self.count_grads_local.value += 1

    def launch_gossip_process(self):
        """
        Creates an independent gossip process using python's multiprocessing library, start it.
        """
        averaging_process = mp.Process(
            target=gossip_process,
            args=(
                self.rank,
                self.local_rank,
                self.world_size,
                self.rank_other,
                self.params_com,
                self.params_com_other,
                self.barrier_sync_averaging,
                self.continue_grad_routine,
                self.barrier_end_init,
                self.barrier_com_grad,
                self.log,
                self.com_history,
                self.count_grads_local,
                self.count_coms_local,
                self.rate_com,
                self.apply_acid,
                self.params_com_tilde,
                self.ode_matrix,
                self.t_last_spike,
                self.delta_t_grad,
                self.beta_tilde,
                self.deterministic_com,
            ),
        )
        averaging_process.start()

    def launch_sync_process(self):
        """
        Creates an independent p2p sync process using python's multiprocessing library, start it.
        """
        p2p_sync_process = mp.Process(
            target=sync_process,
            args=(
                self.rank,
                self.world_size,
                self.rank_other,
                self.new_grads,
                self.barrier_sync_averaging,
                self.log,
            ),
        )
        p2p_sync_process.start()

    def launch_master_sync_process(self):
        """
        Creates an independent master process using python's multiprocessing library for, start it.
        """
        master_sync_process = mp.Process(
            target=master_process,
            args=(
                self.world_size,
                self.nb_grad_tot_goal,
                self.log,
                self.graph_topology,
                self.deterministic_neighbor,
            ),
        )
        master_sync_process.daemon = False  # to enable nested processes
        master_sync_process.start()

    @torch.no_grad()
    def get_weights(self):
        """
        Given a nn.Module, returns a 1D tensor containing all of its parameters.
        """
        return nn.utils.parameters_to_vector(self.module.parameters())

    @torch.no_grad()
    def set_weights(self, weights):
        """
        Given a 1D tensor containing a nn.Module parameters,
        loads the parameters into the nn.Module.
        """
        nn.utils.vector_to_parameters(weights, self.module.parameters())

    def get_com_history(self):
        # count nb coms
        com_history = []
        for k in range(self.world_size):
            count_k = self.com_history[k]
            com_history.append(count_k.value)
        return self.count_coms_local.value, com_history
