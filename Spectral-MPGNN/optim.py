import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np


def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p : p.requires_grad, params)
    if args.opt == 'lbfgs':
        optimizer = optim.LBFGS(filter_fn, lr=float(0.5), max_iter=100, max_eval=500, history_size=150,
                             line_search_fn="strong_wolfe",
                             tolerance_change=1.0 * np.finfo(float).eps, tolerance_grad=1.0 * np.finfo(float).eps)
    elif args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)
    return scheduler, optimizer

# xavier initialization of network parameters
def init_xavier(model):
    """ Initializes the network parameters using xavier initialization.

    To be used with an MLP network with tanh non-linearities

    Parameters
    ----------
    model : torch.nn
        Network to be initialized.
    
    """
    def init_weights(m):
        if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
            g = nn.init.calculate_gain('relu')
            torch.nn.init.xavier_uniform_(m.weight, gain=g)
            m.bias.data.fill_(0)

    model.apply(init_weights)
