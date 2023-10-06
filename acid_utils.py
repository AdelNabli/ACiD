import time
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_momentum_var(model, criterion, optimizer, data_iterator, local_rank, delta_t_grad, momentum, dataset_name):
    """
    Take a grad step per worker to initialize the momentum in the SGD optimizer.
    Use this first grad step to initialize the observed value of "delta_t_grad",
    the time that takes to compute one gradient.
    """
    t_0 = time.time()
    data, target = next(data_iterator)
    data, target = data.to(local_rank), target.to(local_rank)
    optimizer.zero_grad()
    output = model(data)
    if dataset_name == 'CIFAR10':
        output = F.log_softmax(output, dim=1)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    # initialize the delta_t_grad value
    delta_t_grad.value = time.time() - t_0
    # gather the momentums in a list
    mom_list = []
    for p in model.parameters():
        if p.grad is not None:
            if momentum != 0:
                state = optimizer.state[p]
                mom_list.append(state["momentum_buffer"])
            else:
                mom_list.append(p.grad)
    # put it in a 1D tensor in shared memory and returns it
    return nn.utils.parameters_to_vector(mom_list).share_memory_()


def load_momentum(mom_vec, model, optimizer, momentum):
    """
    Loads the momentum 1D variable back into the optimizer,
    so that the optimizer variable and the 1D tensor share the same memory:
    Each new grad step directly updates the 1D tensor.
    """
    pointer = 0
    for p in model.parameters():
        if p.grad is not None and momentum != 0:
            state = optimizer.state[p]
            mom = state["momentum_buffer"]
            # The length of the parameter
            num_mom = mom.numel()
            # Slice the vector, reshape it, and replace the old data of the parameter
            optimizer.state[p]["momentum_buffer"] = mom_vec[
                pointer : pointer + num_mom
            ].view_as(mom)
            # Increment the pointer
            pointer += num_mom
            

@torch.no_grad()
def acid_ode(
    params_com, params_com_tilde, ode_matrix, t_old, t_new, delta_t_grad,
):
    """
    Integrate the ODE for the continuous momentum.
    """
    # Compute the exponential of the matrix of the ode system
    # between t_old and t_new (we re-normalize time using delta_t_grad as the unit of time)
    exp_M = torch.linalg.matrix_exp(ode_matrix * (t_new - t_old) / delta_t_grad)
    a, b, c, d = exp_M[0][0], exp_M[0][1], exp_M[1][0], exp_M[1][1]
    # Do the mixing in-place, so first remembers the value of params
    params_old = params_com.detach().clone()
    params_com.mul_(a).add_(params_com_tilde, alpha=b)
    params_com_tilde.mul_(d).add_(params_old, alpha=c)
