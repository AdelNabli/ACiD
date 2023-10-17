import os
import sys
import time
import torch
import logging
import datetime
import hostlist
import argparse
import multiprocessing as mp
import torch.nn.functional as F
import torch.distributed as dist
from adp import ADP
from utils.data_utils import data_loader, evaluate
from torch.utils.tensorboard import SummaryWriter
from utils.net_utils import create_model, load_optimizer
from utils.logs_utils import (
    create_dict_result,
    save_result,
    create_id_run,
    save_com_logs,
    print_training_evolution,
    log_to_tensorboard,
)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

log = logging.getLogger("distributed_worker")


def get_args_parser():

    parser = argparse.ArgumentParser("Distributed Optimization Script", add_help=False)
    parser.add_argument(
        "--optimizer_name",
        default="SGD",
        type=str,
        help="Name of the optimizer to use. We only support 'SGD' at the moment.",
    )
    parser.add_argument(
        "--dataset_name",
        default="CIFAR10",
        type=str,
        help="Name of the dataset to train on. We support either one of ['CIFAR10','ImageNet'].",
    )
    parser.add_argument(
        "--model_name",
        default="resnet18",
        type=str,
        help="Name of the model to train. We support either one of ['resnet18', 'resnet50'].",
    )
    parser.add_argument(
        "--use_linear_scaling",
        default=False,
        action="store_true",
        help="Whether or not to scale the lr with the world size, as done in https://arxiv.org/pdf/1706.02677.pdf .",
    )
    parser.add_argument(
        "--apply_acid",
        default=False,
        action="store_true",
        help="Whether or not to apply the ACiD momentum.",
    )
    parser.add_argument(
        "--non_iid_data",
        default=False,
        action="store_true",
        help="Whether or not to use non iid data.",
    )
    parser.add_argument(
        "--normalize_grads",
        default=False,
        action="store_true",
        help="Whether or not to normalize gradients before taking the grad step (for stability issues).",
    )
    parser.add_argument(
        "--deterministic_coms",
        default=False,
        action="store_true",
        help="Whether or not to implement Poisson Point Processes for the coms. \
              If True, will make sure there are 'rate_com' communications between EACH gradient steps (not in expectation). \
              Will raise an error if True AND 'rate_com' is not an integer.",
    )
    parser.add_argument(
        "--deterministic_neighbor",
        default=False,
        action="store_true",
        help="Whether or not to pair neighbors for p2p com deterministicaly (thus waiting for it if not available) or look for an available one randomly.",
    )
    parser.add_argument(
        "--batch_size",
        default=256,
        type=int,
        help="Batch size to use for each worker.",
    )
    parser.add_argument(
        "--lr",
        default=0.05,
        type=float,
        help="Learning rate of the optimizer.",
    )
    parser.add_argument(
        "--momentum",
        default=0.9,
        type=float,
        help="Momentum for the optimizer.",
    )
    parser.add_argument(
        "--weight_decay",
        default=5e-4,
        type=float,
        help="Weight decay value of the optimizer.",
    )
    parser.add_argument(
        "--gamma_lr_scheduler",
        default=0.1,
        type=float,
        help="Parameter gamma for the lr scheduler.",
    )
    parser.add_argument(
        "--filter_bias_and_bn",
        default=False,
        action="store_true",
        help="Whether or not to prevent weight decay to be applied to biases and batch norm params in the optimizer.",
    )
    parser.add_argument(
        "--rate_com",
        default=1.0,
        type=float,
        help="The expected ratio between the total number of pairwise communications and the total number of gradients.",
    )
    parser.add_argument(
        "--n_epoch_if_1_worker",
        default=80,
        type=int,
        help="The number of epochs to perform if there was only one worker. Is used to compute the total number of gradient steps to perform.",
    )
    parser.add_argument(
        "--graph_topology",
        default="complete",
        type=str,
        help="Graph topology to use for the communications.",
    )
    parser.add_argument(
        "--path_logs",
        default=os.getcwd(),
        type=str,
        help="Path to the root directory for the logs and results folders.",
    )

    return parser


def run(rank, local_rank, world_size, n_nodes, master_addr, master_port, args):

    #### INITIALIZATION ####
    # Initialize a TCP store for the global run id
    TCP_IP = master_addr
    TCP_port = master_port + 3
    # init  path dir for the logs
    tensorboard_dir = args.path_logs + "/tensorboard/"
    slurm_logs_dir = args.path_logs + "/logs/"
    if rank == 0:
        # initialize the server store for the run id
        filestore = dist.TCPStore(TCP_IP, port=TCP_port, is_master=True)
        # create a unique id_run
        id_run = create_id_run()
        filestore.set("id_run", str(id_run))
        # if a tensorboard dir doesn't exist, makes it
        if not os.path.exists(tensorboard_dir):
            os.mkdir(tensorboard_dir)
        # if the dir for the slurm logs doesn't exist, makes it
        if not os.path.exists(slurm_logs_dir):
            os.mkdir(slurm_logs_dir)
    else:
        # initialize the client stores for the run id
        filestore = dist.TCPStore(TCP_IP, port=TCP_port, is_master=False)
    # get the common id_run
    id_run = filestore.get("id_run")

    # Initialization of constants and data
    train_loader, n_batch_per_epoch = data_loader(
        rank,
        world_size,
        train=True,
        batch_size=args.batch_size,
        dataset_name=args.dataset_name,
        non_iid_data=args.non_iid_data,
    )
    data_iterator = iter(train_loader)
    # n_batch_per_epoch = len(train_loader)
    nb_grad_worker = int(args.n_epoch_if_1_worker * n_batch_per_epoch / world_size)
    nb_grad_tot = nb_grad_worker * world_size
    # Initialize the model, the optimizer, the scheduler
    model, criterion = create_model(args.model_name, args.dataset_name)
    model, criterion = model.to(local_rank), criterion.to(local_rank)
    optimizer, scheduler = load_optimizer(
        model,
        args.lr,
        args.momentum,
        args.weight_decay,
        args.gamma_lr_scheduler,
        args.optimizer_name,
        world_size,
        args.use_linear_scaling,
        nb_grad_tot,
        args.n_epoch_if_1_worker,
        args.batch_size,
        args.filter_bias_and_bn,
        args.dataset_name,
    )
    # Initialize the worker
    adp_model = ADP(
        model,
        rank,
        local_rank,
        world_size,
        nb_grad_tot,
        log,
        args.rate_com,
        args.apply_acid,
        criterion,
        optimizer,
        data_iterator,
        args.momentum,
        args.dataset_name,
        args.graph_topology,
        args.deterministic_coms,
        args.deterministic_neighbor,
    )
    path_tensorboard = tensorboard_dir + id_run.decode()
    writer = SummaryWriter(path_tensorboard)

    #### COMMUNICATIONS & GRAD STEPS ####

    t_begin = time.time()
    t_last_epoch = t_begin
    t0 = time.time()
    count_grads = 0
    epoch = 0
    # switch to train mode
    adp_model.train()

    while adp_model.continue_grad_routine.value:
        adp_model.start()
        # get the next batch of data
        try:
            images, labels = next(data_iterator)
        except StopIteration:
            # When the epoch ends, start a new epoch.
            data_iterator = iter(train_loader)
            images, labels = next(data_iterator)
        # distribution of images and labels to all GPUs
        images = images.to(local_rank, non_blocking=True)
        labels = labels.to(local_rank, non_blocking=True)

        # forward pass
        outputs = adp_model(images)
        if args.dataset_name == "CIFAR10":
            outputs = F.log_softmax(outputs, dim=1)
        loss = criterion(outputs, labels)

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        adp_model.step(optimizer, scheduler, args.normalize_grads)

        # print info every epoch
        count_grads += 1
        epoch, t_last_epoch = print_training_evolution(
            log,
            count_grads,
            adp_model.count_coms_local.value,
            n_batch_per_epoch,
            rank,
            t_begin,
            t_last_epoch,
            loss,
            epoch,
        )
        log_to_tensorboard(writer, count_grads, rank, loss, t0, delta_step_for_log=5)

    t_end = time.time()

    #### END OF TRAINNING ####

    total_time = t_end - t_begin
    # get the com history
    count_com, com_history = adp_model.get_com_history()
    save_com_logs(com_history, args.path_logs, id_run, rank)
    com_message = (
        "\n WORKER {}: Number of grad steps : {} , Number of comm : {}".format(
            rank, count_grads, count_com
        )
    )
    print(com_message)
    # put the info in the filestore in order to make it available to worker 0
    filestore.set(
        "Worker {} #grad / #comm".format(rank), "{} / {}".format(count_grads, count_com)
    )

    # if we are the worker 0, print/saves the logs
    if rank == 0:
        # wait all the workers
        filestore.wait(
            ["Worker {} #grad / #comm".format(rank) for rank in range(world_size)]
        )
        print("Final model: ")
        test_loader, _ = data_loader(
            rank,
            world_size,
            train=False,
            batch_size=args.batch_size,
            dataset_name=args.dataset_name,
        )
        loss, correct, len_data, percent = evaluate(
            adp_model,
            test_loader,
            criterion,
            local_rank=0,
            print_message=True,
            dataset_name=args.dataset_name,
        )
        grad_message = (
            "\n Total training time for {} GPUS : {} minutes, {:.1f} secondes".format(
                world_size, int(total_time // 60), total_time % 60
            )
        )
        print(grad_message)
        # save results
        dict_result = create_dict_result(
            args,
            filestore,
            world_size,
            n_nodes,
            torch.cuda.get_device_name(),
            total_time,
            correct,
            len_data,
            percent,
            id_run,
        )
        save_result(args.path_logs + "/results.csv", dict_result)


if __name__ == "__main__":
    # parse the arguments
    parser = argparse.ArgumentParser(
        "Distributed Optimization Script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    # set the start method for multiprocessing.
    # mp is used to launch independent processes on each worker
    # for the grad steps and communication routine.
    mp.set_start_method("spawn", force=True)
    # get distributed configuration from Slurm environment
    NODE_ID = os.environ["SLURM_NODEID"]
    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    # get node list from slurm
    hostnames = hostlist.expand_hostlist(os.environ["SLURM_JOB_NODELIST"])
    n_nodes = len(hostnames)
    # get IDs of reserved GPU
    gpu_ids = os.environ["SLURM_STEP_GPUS"].split(",")
    # define MASTER_ADD & MASTER_PORT, used to define the distributed communication environment
    master_addr = hostnames[0]
    master_port = 12346 + int(min(gpu_ids))  # to avoid port conflict on the same node
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["MPI4PY_RC_THREADS"] = str(
        0
    )  # to avoid problems with MPI in multi-node setting
    # display info
    if rank == 0:
        print(">>> Training on ", n_nodes, " nodes and ", world_size)
        print("Arguments:")
        print(args)
    print(
        "- Process {} corresponds to GPU {} of node {}".format(
            rank, local_rank, NODE_ID
        )
    )

    run(rank, local_rank, world_size, n_nodes, master_addr, master_port, args)
