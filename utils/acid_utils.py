import time
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_momentum_var(
    model,
    criterion,
    optimizer,
    data_iterator,
    local_rank,
    delta_t_grad,
    momentum,
    dataset_name,
):
    """
    Take a grad step per worker to initialize the momentum in the SGD optimizer.
    Indeed, in SGD, the momentum buffers are initialized at "None" (cf https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html#SGD ).
    So in order to have access to them, we will need to take a first gradient step.
    Use this first grad step to initialize the observed value of "delta_t_grad",
    the time that takes to compute one gradient.

    Parameters:
        - model (nn.Module): the Neural Network model.
        - criterion (nn.Module): the criterion used to optimize model.
        - optimizer (torch Optimizer): the Optimizer to use, only SGD is supported for now.
        - data_iterator (iter of torch DataLoader): iterator over the dataset, use "next" to load next minibatch of data.
        - local_rank (int): the local rank of the worker inside its compute node (to load the data in the right GPU)
        - delta_t_grad (mp.Value storing a double): the variable keeping track of the time that it takes to make a grad step.
        - momentum (float): the momentum value in SGD.
        - dataset_name (str): if 'CIFAR10', will apply log_softmax to the output of the model.

    Returns:
        - mom_vec (torch.tensor): a 1D tensor, the same size as the number of model parameters,
                                  containing the momentum buffers stored in the Optimizer.
    """
    # init time
    t_0 = time.time()
    # load mini-batch of data
    data, target = next(data_iterator)
    data, target = data.to(local_rank), target.to(local_rank)
    # perform a forward pass
    optimizer.zero_grad()
    output = model(data)
    if dataset_name == "CIFAR10":
        output = F.log_softmax(output, dim=1)
    loss = criterion(output, target)
    # backward and optimizer.step()
    loss.backward()
    optimizer.step()
    # initialize the delta_t_grad value
    delta_t_grad.value = time.time() - t_0
    # gather the momentums in a list
    mom_list = []
    for p in model.parameters():
        # if it is a parameter with gradient
        if p.grad is not None:
            if momentum != 0:
                # load the momentum buffer and add it to the list
                state = optimizer.state[p]
                mom_list.append(state["momentum_buffer"])
            else:
                mom_list.append(p.grad)
    # put it in a 1D tensor in shared memory and returns it
    return nn.utils.parameters_to_vector(mom_list).share_memory_()


def load_momentum(mom_vec, model, optimizer, momentum):
    """
    Re-writing of "nn.utils.vector_to_parameters" for the momentum buffer.
    Loads the momentum 1D variable back into the optimizer,
    so that the optimizer buffer and the 1D tensor share the same memory space:
    Each new grad step directly updates the 1D tensor.

    Parameters:
        - mom_vec (torch.tensor): a 1D tensor, the same size as the number of model parameters,
                                  containing the momentum buffers stored in the Optimizer.
        - model (nn.Module): the Neural Network model.
        - optimizer (torch Optimizer): the Optimizer to use, only SGD is supported for now.
        - momentum (float): the momentum value in SGD.

    """
    # perform similarly as in "nn.utils.vector_to_parameters".
    # init a pointer var.
    pointer = 0
    # access all the momentum buffers linked to params with gradients.
    for p in model.parameters():
        if p.grad is not None and momentum != 0:
            # retrieve the momentum buffer in the Optimizer
            state = optimizer.state[p]
            mom = state["momentum_buffer"]
            # The length of the parameter
            num_mom = mom.numel()
            # Slice the vector, reshape it, and replace the old data of the momentum buffer
            optimizer.state[p]["momentum_buffer"] = mom_vec[
                pointer : pointer + num_mom
            ].view_as(mom)
            # Increment the pointer
            pointer += num_mom


@torch.no_grad()
def acid_ode(
    params_com,
    params_com_tilde,
    ode_matrix,
    t_old,
    t_new,
    delta_t_grad,
):
    """
    Integrate the ODE for the continuous momentum, see https://arxiv.org/pdf/2306.08289.pdf for details.
    Update parameters (params_com and params_com_tilde) in-place.

    Parameters:
        - params_com (torch.tensor): 1D tensor containing all of the models learnable parameters.
        - params_com_tilde (torch.tensor): "momentum" variable, same size as params_com, mixing with params_com to obtain acceleration.
        - ode_matrix (torch.tensor): a 2x2 matrix storing the parameters of the linear mixing between params and params_tilde.
        - t_old (float): time of the last local update.
        - t_new (float): time of the current update.
        - delta_t_grad (float): time that it takes to compute a grad step. Used to re-normalize time, as done in the paper.
    """
    # Compute the exponential of the matrix of the ode system
    # between t_old and t_new (we re-normalize time using delta_t_grad as the unit of time)
    exp_M = torch.linalg.matrix_exp(ode_matrix * (t_new - t_old) / delta_t_grad)
    a, b, c, d = exp_M[0][0], exp_M[0][1], exp_M[1][0], exp_M[1][1]
    # Do the mixing in-place, so first remembers the value of params
    params_old = params_com.detach().clone()
    # matrix multiplication
    params_com.mul_(a).add_(params_com_tilde, alpha=b)
    params_com_tilde.mul_(d).add_(params_old, alpha=c)
