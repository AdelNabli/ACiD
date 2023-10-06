import os
import csv
import time
import math
import torch
import random
import datetime


def create_id_run():
    """
    Create a unique id for the current run using the date
    """
    time_now = datetime.datetime.now()
    id_run = "_".join(
        [
            str(time_part)
            for time_part in [
                time_now.year,
                time_now.month,
                time_now.day,
                time_now.hour,
                time_now.minute,
                time_now.second,
            ]
        ]
    )
    # to differentiate runs launched exactly at the same time on the cluster.
    random_number = random.randint(0, 100)
    id_run += "_" + str(random_number)
    return id_run


def create_dict_result(
    args,
    filestore,
    world_size,
    n_nodes,
    cuda_device,
    total_time,
    correct,
    len_data,
    percent,
    id_run,
):
    """
    Put the different logs/metrics into a dict and returns it.
    """
    # starts from the args parsed
    dict_result = vars(args).copy()
    # removes the paths
    dict_result.pop("path_logs")
    # writes it in the dict
    dict_result["0_id_run"] = id_run
    # gather the message from all workers in the shared filestore
    for rank in range(world_size):
        key = "Worker {} #grad / #comm".format(rank)
        dict_result[key] = filestore.get(key)
    # transform the time in a more readable format
    dict_result["Tot_time"] = "{} min {:.1f} s".format(
        int(total_time // 60), total_time % 60
    )
    # put the other values in the dict
    dict_result["N_workers"] = world_size
    dict_result["n_nodes"] = n_nodes
    dict_result["cuda_device"] = cuda_device
    dict_result["Test_acc"] = "{} / {}".format(correct, len_data)
    dict_result["Test_acc_percent"] = percent

    return dict_result


def gather_previous_data(reader):
    """
    Reads the csv of logs and returns the data it contains.
    """
    keys = set()
    rows = []
    for row in reader:
        keys.update([key for key in row.keys()])
        rows.append(row.copy())
    return keys, rows


def write_new_csv(path_to_result_csv, dict_result):
    """
    Creates a csv to stock the results and logs of the run.
    """
    with open(path_to_result_csv, "w", newline="") as csvfile:
        # gather the keys from the dict of result
        fieldnames = list(dict_result.keys())
        fieldnames.sort()
        # init the writer
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # write the header
        writer.writeheader()
        # write the result
        writer.writerow(dict_result)


def update_csv_result(path_to_result_csv, dict_result):
    """
    Updates the csv of results if the csv already exists.
    """
    with open(path_to_result_csv, "r+", newline="") as csvfile:
        # read the previous results
        reader = csv.DictReader(csvfile)
        # gather the previous keys
        fieldnames, rows = gather_previous_data(reader)
        # add the keys from the current dict of result
        fieldnames.update(list(dict_result.keys()))
        # sort the keys
        fieldnames = list(fieldnames)
        fieldnames.sort()
        # clear the file
        csvfile.truncate(0)
        # to prevent weird \x00 appended at the start of the file
        csvfile.seek(0)
        # initialize the writer
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # write the new header
        writer.writeheader()
        # write back the previous data
        for row in rows:
            writer.writerow(row)
        # write the new results
        writer.writerow(dict_result)


def save_result(path_to_result_csv, dict_result):
    """
    Create or updates the csv of results.
    """
    # if a result file already exists
    if os.path.exists(path_to_result_csv):
        # we update it with our new results
        update_csv_result(path_to_result_csv, dict_result)
    # else, we create the file
    else:
        write_new_csv(path_to_result_csv, dict_result)


def save_com_logs(com_history, path_logs, id_run, rank):
    """
    Saves the history of communications so that they can be visualized afterwards.
    """
    path_folder = path_logs + "/com_logs/"
    try:
        os.mkdir(path_folder)
    except:
        pass
    path_com_logs = path_folder + str(id_run) + ".txt"
    with open(path_com_logs, "a+") as file:
        file.write(str(rank) + ' : ' + str(com_history) + "\n")
        
        
def print_training_evolution(log, nb_grad_local, nb_com_local, n_batch_per_epoch, rank, t_beg, t_last_epoch, loss, epoch):
    if nb_grad_local % n_batch_per_epoch == 0:
        epoch += 1
        delta_t = time.time() - t_beg
        log.info(
            " Worker {}. Epoch {} in {:.2f} s. Total time: {} min {:.2f} s. # grad : {} . # com : {}. loss {}".format(
                rank,
                epoch,
                time.time() - t_last_epoch,
                int(delta_t // 60),
                delta_t % 60,
                nb_grad_local,
                nb_com_local,
                float(loss.detach().cpu()),
            )
        )
        t_last_epoch = time.time()
    return epoch, t_last_epoch

def log_to_tensorboard(writer, nb_grad_local, rank, loss, t0): #, output, target, adp_model, optimizer, log):
    if nb_grad_local % 1 == 0:
        with torch.no_grad():
            loss_f = float(loss.detach().cpu())
            writer.add_scalars(
                "loss_t",
                {str(rank): loss_f},
                time.time() - t0,
            )
            writer.add_scalars(
                "loss_step",
                {str(rank): loss_f},
                nb_grad_local,
            )
            """
            pred = output.argmax(dim=1, keepdim=True)
            train_error = (
                1 - pred.eq(target.view_as(pred)).sum().item() / len(pred)
            ) * 100
            writer.add_scalars(
                "train_error",
                {str(rank): float(train_error)},
                nb_grad_local,
            )
            l2_grad = float(torch.linalg.norm(adp_model.mom_vec).detach().cpu())
            writer.add_scalars(
                "grad_norm",
                {str(rank): l2_grad},
                nb_grad_local,
            )
            lr = optimizer.param_groups[0]["lr"]
            writer.add_scalars(
                "lr",
                {str(rank): float(lr)},
                nb_grad_local,
            )"""

def make_checkpoint(rank, model, optimizer, nb_grad_limit, make_checkpoint, path_logs, id_run):
    # save 5 steps before the end to make sure it is saved without a mp bug
    if nb_grads_count == nb_grad_limit - 5 and make_checkpoint:
        # if it does not already exists, creates the folder
        path_folder = path_logs + "/checkpoints/" + id_run.decode()
        try:
            os.mkdir(path_folder)
        except:
            pass
        # saves the model & optimizer of each rank
        path_save_model = path_folder + "/" + str(rank) + "_model.pt"
        path_save_optim = path_folder + "/" + str(rank) + "_optim.pt"
        try:
            torch.save(model.state_dict(), path_save_model)
            torch.save(optimizer.state_dict(), path_save_optim)
        except:
            pass
