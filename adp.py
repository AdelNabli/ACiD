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
    The 'Asynchronous Data Parallel' wrapper around the model, with added functions to perform asynchronous p2p communications in the background.
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
        """
        Initialize and launch all background processes necessary for the p2p communications.
        If this is worker 0, then additional processes will be launched in the background to coordinate communications.
        
        Parameters:
            - model (nn.Module): the neural net wrapped by ADP.
            - rank (int): our rank id in the distributed setting.
            - local_rank (int): the id of the GPU device the model is loaded on in the cluster's node.
            - world_size (int): the total number of workers.
            - nb_grad_tot_goal (int): The target number of total nb of grads performed by all workers.
            - log (logger): to print messages in the logs if needed.
            - rate_com (float): the rate at which p2p communications are done compared to local grad steps.
            - apply_acid (bool): whether or not to apply ACiD momentum.
            - criterion (nn.Module): the criterion used to optimize model.
            - optimizer (torch Optimizer): the Optimizer to use, only SGD is supported for now.
            - data_iterator (iter of torch DataLoader): iterator over the dataset.
            - momentum (float): the momentum value in SGD.
            - dataset_name (str): one of ['CIFAR10', 'ImageNet'].
            - graph_topology (str): Graph topology to use to make p2p communication (dictates which edges can be used).
                                Currently supports either of ['complete', 'cycle', 'exponential'].
            - deterministic_com (bool): whether or not to schedule to use Poisson Point Processes for the communications.
                                    if True, a random number of p2p communications between 2 grad steps are done, following a poisson law.
            - deterministic_neighbor (bool): whether or not to schedule the p2p communications.
                                         if True, if at the next step, worker i is supposed to communicate with j,
                                         i will wait for j to be available to communicate.
                                         if False, i will communicate faster, by just picking one of its available neighbor. 
        """

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

        # Init mp Variables used in multiple processes
        self.rank_other = mp.Value("i", -1)
        # nb of new grads between two comms to the master process
        self.new_grads = mp.Value("i", 0)
        # count of local grad steps
        self.count_grads_local = mp.Value("i", 0)
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
        """
        Initialize all variables necessary for applying the ACiD momentum.
        Especially, initialize:
            * params_com_tilde (torch.tensor, 1D) : the "duplicate" of the Neural Network's parameters necessary for the momentum.
            * ode_matrix (torch.tensor 2D): the mixing matrix (between params and params_tilde) used in the ODE part of ACiD's dynamic.
            * delta_t_grad (mp.Value): a value keeping track of "how long it takes to perform a grad step" to normalize time in the continuous dynamic.
            * mom_vec (torch.tensor, 1D): the momentum buffer of SGD. Makes it 1D, and makes sure the momentum buffers of the optimizer
                                          points to this 1D tensor, so that when the optimizer updates the momentum buffer, it is
                                          this 1D tensor that is updated.
                                          Used to make sure the "grad step" performed on the model (thus on "params_com") could also be applied on "params_com_tilde".
            * eta, beta_tilde (tuple of floats): ACiD hyperparameters (that have defined theoretical values, depending on the communication rate and graph's topology).
            
        For more details, see our paper https://arxiv.org/pdf/2306.08289.pdf .
        """
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
        Each forward pass is used to increment the local 'count of gradient steps since last communication'
        (used by the master process to count the global number of grad step taken and decide when to stop the training).
        """
        self.new_grads.value += 1
        return self.module(*args, **kwargs)

    def start(self):
        """
        Indicate the beginning of a gradient computation.
        Used for the computation of "delta_t_grad", the time it takes to perform a gradient step.
        """
        self.t_beg_grad = time.time()

    def step(self, optimizer, scheduler, normalize_grads):
        """
        Perform a gradient step on the model.
        * If we apply ACiD momentum, then the gradient step is also performed for the momentum variable,
          and the continuous mixing between the momentum variable and the NN's parameters is done.
          see our paper for details https://arxiv.org/pdf/2306.08289.pdf .
        * Updates the 'delta_t_grad' variable using an Exponential Moving Average.
        * Makes sure that the right ratio of communications/gradients is kept through the use of a mp.Barrier synchronizing
          with the p2p_averaging process.
          
        Parameters:
            - optimizer (torch Optimizer): the Optimizer to use, only SGD is supported for now.
            - lr_scheduler (Scheduler): the lr scheduler to use.
            - normalize_grads (bool): if True, will normalize the grads (prevent training instabilities that might occur).
        """
        # if apply acid, update the momentum variable
        if self.apply_acid:
            # update the time variables used to update the exponential moving average for delta_t_grad and for the spike
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
            # stabilize training
            nn.utils.clip_grad_norm_(self.module.parameters(), 1.0)
        optimizer.step()

        # take a grad step on the momentum var
        if self.apply_acid:
            lr = optimizer.param_groups[0]["lr"]
            # perform the grad step for params tilde
            self.params_com_tilde.add_(self.mom_vec, alpha=-lr)
            # update delta_t_grad using an EMA
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
        # counts the local count of grad steps
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
        This process is used by each worker to communicate with the master process to signal availability for communication
        and know the "peer" with which to communicate next.
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
        This process pairs workers according to the graph's topology and workers current availability for communications,
        and signal to every worker when to stop training.
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
        Wrapper around nn.utils.parameters_to_vector.
        Given a nn.Module, returns a 1D tensor containing all of its parameters.
        """
        return nn.utils.parameters_to_vector(self.module.parameters())

    @torch.no_grad()
    def set_weights(self, weights):
        """
        Wrapper around nn.utils.vector_to_parameters.
        Given a 1D tensor containing a nn.Module parameters,
        loads the parameters into the nn.Module.
        """
        nn.utils.vector_to_parameters(weights, self.module.parameters())

    def get_com_history(self):
        """
        Sends back the content of the logged communication history.
        """
        # count nb coms
        com_history = []
        for k in range(self.world_size):
            count_k = self.com_history[k]
            com_history.append(count_k.value)
        return self.count_coms_local.value, com_history
