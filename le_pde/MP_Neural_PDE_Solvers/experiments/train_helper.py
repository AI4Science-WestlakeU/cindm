import torch
import random
import pdb
from torch import nn, optim
from torch.utils.data import DataLoader

# from equations.PDEs import *
import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from MP_Neural_PDE_Solvers.equations.PDEs import *
from MP_Neural_PDE_Solvers.common.utils import HDF5Dataset, GraphCreator, p

def training_loop(model: torch.nn.Module,
                  unrolling: list,
                  batch_size: int,
                  optimizer: torch.optim,
                  loader: DataLoader,
                  graph_creator: GraphCreator,
                  criterion: torch.nn.modules.loss,
                  device: torch.cuda.device="cpu",
                  is_timing: bool = False,
                  uniform_sample: int = -1,
                 ) -> torch.Tensor:
    """
    One training epoch with random starting points for every trajectory
    Args:
        model (torch.nn.Module): neural network PDE solver
        unrolling (list): list of different unrolling steps for each batch entry
        batch_size (int): batch size
        optimizer (torch.optim): optimizer used for training
        loader (DataLoader): training dataloader
        graph_creator (GraphCreator): helper object to handle graph data
        criterion (torch.nn.modules.loss): criterion for training
        device (torch.cuda.device): device (cpu/gpu)
    Returns:
        torch.Tensor: training losses
    """

    losses = []
    for (u_base, u_super, x, variables) in loader:
        p.print("1.0", precision="millisecond", is_silent=is_timing<1, avg_window=1)
        optimizer.zero_grad()
        # Randomly choose number of unrollings
        unrolled_graphs = random.choice(unrolling)
        steps = [t for t in range(graph_creator.tw,
                                  graph_creator.t_res - graph_creator.tw - (graph_creator.tw * unrolled_graphs) + 1)]
        p.print("2", precision="millisecond", is_silent=is_timing<1, avg_window=1)
        # Randomly choose starting (time) point at the PDE solution manifold
        random_steps = random.choices(steps, k=batch_size)
        data, labels = graph_creator.create_data(u_super, random_steps)  # data/labels: [B:16, tw:25, nx:40]
        p.print("3", precision="millisecond", is_silent=is_timing<1, avg_window=1)
        if f'{model}' == 'GNN':
            graph = graph_creator.create_graph(data, labels, x, variables, random_steps, uniform_sample=uniform_sample).to(device)
        else:
            data, labels = data.to(device), labels.to(device)
        p.print("4", precision="millisecond", is_silent=is_timing<1, avg_window=1)

        # Unrolling of the equation which serves as input at the current step
        # This is the pushforward trick!!!
        with torch.no_grad():
            for _ in range(unrolled_graphs):
                random_steps = [rs + graph_creator.tw for rs in random_steps]
                _, labels = graph_creator.create_data(u_super, random_steps)
                if f'{model}' == 'GNN':
                    pred = model(graph)
                    # graph: Data(x=[640, 25], edge_index=[2, 3648], y=[640, 25], pos=[640, 2], batch=[640], alpha=[640, 1], beta=[640, 1], gamma=[640, 1])  # here graph.x has combined different graph via batch dimension
                    graph = graph_creator.create_next_graph(graph, pred, labels, random_steps).to(device)
                else:
                    data = model(data)
                    labels = labels.to(device)
        p.print("5", precision="millisecond", is_silent=is_timing<1, avg_window=1)

        if f'{model}' == 'GNN':
            pred = model(graph)
            loss = criterion(pred, graph.y)  # pred/graph.y: [640, 25]
        else:
            pred = model(data)
            loss = criterion(pred, labels)
        p.print("6", precision="millisecond", is_silent=is_timing<1, avg_window=1)

        loss = torch.sqrt(loss)
        p.print("7", precision="millisecond", is_silent=is_timing<1, avg_window=1)
        loss.backward()
        p.print("7.1", precision="millisecond", is_silent=is_timing<1, avg_window=1)
        losses.append(loss.detach() / batch_size)
        optimizer.step()
        p.print("7.2", precision="millisecond", is_silent=is_timing<1, avg_window=1)

    losses = torch.stack(losses)
    return losses

def test_timestep_losses(model: torch.nn.Module,
                         steps: list,
                         batch_size: int,
                         loader: DataLoader,
                         graph_creator: GraphCreator,
                         criterion: torch.nn.modules.loss,
                         device: torch.cuda.device = "cpu", uniform_sample=-1) -> None:
    """
    Loss for one neural network forward pass at certain timepoints on the validation/test datasets
    Args:
        model (torch.nn.Module): neural network PDE solver
        steps (list): input list of possible starting (time) points
        batch_size (int): batch size
        loader (DataLoader): dataloader [valid, test]
        graph_creator (GraphCreator): helper object to handle graph data
        criterion (torch.nn.modules.loss): criterion for training
        device (torch.cuda.device): device (cpu/gpu)
    Returns:
        None
    """

    for step in steps:

        if (step != graph_creator.tw and step % graph_creator.tw != 0):
            continue

        losses = []
        for (u_base, u_super, x, variables) in loader:
            with torch.no_grad():
                same_steps = [step]*batch_size
                data, labels = graph_creator.create_data(u_super, same_steps)
                if f'{model}' == 'GNN':
                    graph = graph_creator.create_graph(data, labels, x, variables, same_steps, uniform_sample=uniform_sample).to(device)
                    pred = model(graph)
                    loss = criterion(pred, graph.y)
                else:
                    data, labels = data.to(device), labels.to(device)
                    pred = model(data)
                    loss = criterion(pred, labels)
                losses.append(loss / batch_size)

        losses = torch.stack(losses)
        print(f'Step {step}, mean loss {torch.mean(losses)}')



def test_unrolled_losses(model: torch.nn.Module,
                         steps: list,  # range(25, 250-25+1)
                         batch_size: int,
                         nr_gt_steps: int,  # 2
                         nx_base_resolution: int,
                         loader: DataLoader,
                         graph_creator: GraphCreator,
                         criterion: torch.nn.modules.loss,
                         device: torch.cuda.device = "cpu",
                         uniform_sample = -1,
                        ) -> torch.Tensor:
    """
    Loss for full trajectory unrolling, we report this loss in the paper
    Args:
        model (torch.nn.Module): neural network PDE solver
        steps (list): input list of possible starting (time) points
        nr_gt_steps (int): number of numerical input timesteps
        nx_base_resolution (int): spatial resolution of numerical baseline
        loader (DataLoader): dataloader [valid, test]
        graph_creator (GraphCreator): helper object to handle graph data
        criterion (torch.nn.modules.loss): criterion for training
        device (torch.cuda.device): device (cpu/gpu)
    Returns:
        torch.Tensor: valid/test losses
    """
    losses = []
    losses_base = []
    for (u_base, u_super, x, variables) in loader:
        losses_tmp = []
        losses_base_tmp = []
        with torch.no_grad():
            same_steps = [graph_creator.tw * nr_gt_steps] * batch_size  # [50] * batch_size:16
            data, labels = graph_creator.create_data(u_super, same_steps)  # first time: data: from 25:50, label: from 50:75
            if f'{model}' == 'GNN':
                graph = graph_creator.create_graph(data, labels, x, variables, same_steps, uniform_sample=uniform_sample).to(device)
                pred = model(graph)
                loss = criterion(pred, graph.y) / nx_base_resolution
            else:
                data, labels = data.to(device), labels.to(device)
                pred = model(data)
                loss = criterion(pred, labels) / nx_base_resolution

            losses_tmp.append(loss / batch_size)

            # Unroll trajectory and add losses which are obtained for each unrolling
            for step in range(graph_creator.tw * (nr_gt_steps + 1), graph_creator.t_res - graph_creator.tw + 1, graph_creator.tw):
                same_steps = [step] * batch_size
                _, labels = graph_creator.create_data(u_super, same_steps)
                if f'{model}' == 'GNN':
                    graph = graph_creator.create_next_graph(graph, pred, labels, same_steps).to(device)
                    pred = model(graph)
                    loss = criterion(pred, graph.y) / nx_base_resolution  # pred/graph.y: [640, 25]
                else:
                    labels = labels.to(device)
                    pred = model(pred)
                    loss = criterion(pred, labels) / nx_base_resolution 
                losses_tmp.append(loss / batch_size)  # batch_size = 16

            # # Losses for numerical baseline
            # for step in range(graph_creator.tw * nr_gt_steps, graph_creator.t_res - graph_creator.tw + 1,
            #                   graph_creator.tw):
            #     same_steps = [step] * batch_size
            #     _, labels_super = graph_creator.create_data(u_super, same_steps)
            #     _, labels_base = graph_creator.create_data(u_base, same_steps)
            #     loss_base = criterion(labels_super, labels_base) / nx_base_resolution
            #     losses_base_tmp.append(loss_base / batch_size)

        losses.append(torch.sum(torch.stack(losses_tmp)))
        losses_base.append(torch.tensor(0., dtype=torch.float32))
        # losses_base.append(torch.sum(torch.stack(losses_base_tmp)))

    losses = torch.stack(losses)
    losses_base = torch.stack(losses_base)
    print(f'Unrolled forward losses {torch.mean(losses)}')
    print(f'Unrolled forward base losses {torch.mean(losses_base)}')

    return losses




