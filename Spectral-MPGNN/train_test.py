import pandas as pd
from torch_geometric.loader import DataLoader
from tqdm import trange
import copy
from pathlib import Path
import matplotlib.pyplot as plt

from model_SpectralMeshGraphNet import SpectralMeshGraphNet
from optim import *
from conf import *
from InputNormalize import unnormalize
from mask import lossMask

def train(spt_dataset, spc_dataset, device, stats_list, args):
    '''
    Performs a training loop on the dataset for MeshGraphNets. Also calls
    test and validation functions.
    '''

    df = pd.DataFrame(columns=['epoch','train_loss','test_loss', 'velo_val_loss'])

    #Define the model name for saving 
    model_name='model_spectralGNN_w_X_StressType_nl'+str(args.num_layers)+'_bs'+str(args.batch_size) + \
               '_hd'+str(args.hidden_dim)+'_ep'+str(args.epochs)+'_wd'+str(args.weight_decay) + \
               '_lr'+str(args.lr)+'_shuff_'+str(args.shuffle)+'_tr'+str(args.train_size)+'_te'+str(args.test_size)

    #torch_geometric DataLoaders are used for handling the data of lists of graphs
    spt_loader = DataLoader(spt_dataset[:args.train_size], batch_size=args.batch_size, shuffle=False)
    spt_test_loader = DataLoader(spt_dataset[args.train_size:], batch_size=args.batch_size, shuffle=False)
    spc_loader = DataLoader(spc_dataset[:args.train_size], batch_size=args.batch_size, shuffle=False)
    spc_test_loader = DataLoader(spc_dataset[args.train_size:], batch_size=args.batch_size, shuffle=False)

    #The statistics of the data are decomposed
    [mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_force,std_vec_force,mean_vec_y,std_vec_y] = stats_list
    (mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_force,std_vec_force,mean_vec_y,std_vec_y)=(mean_vec_x.to(device),
        std_vec_x.to(device),mean_vec_edge.to(device),std_vec_edge.to(device),mean_vec_force.to(device),std_vec_force.to(device),
        mean_vec_y.to(device),std_vec_y.to(device))

    # build model
    num_node_features = spt_dataset[0].x.shape[1]
    num_edge_features = spt_dataset[0].edge_attr.shape[1]
    num_ext_force = spt_dataset[0].del_ext_force.shape[1]
    num_classes = spt_dataset[0].y.shape[1] # the dynamic variables have the shape of 1 (displacement)
    num_sp_node_features = spc_dataset[0].x.shape[1]
    num_sp_edge_features = spc_dataset[0].edge_attr.shape[1]
    

    model = SpectralMeshGraphNet(num_node_features, num_edge_features, num_sp_node_features, num_sp_edge_features, 
                            args.hidden_dim, num_ext_force, num_classes, args).to(device)
    init_xavier(model)
    scheduler, opt = build_optimizer(args, model.parameters())

    # train
    losses = []
    test_losses = []
    disp_val_losses = []
    best_test_loss = np.inf
    best_model = None
    for epoch in trange(args.epochs, desc="Training", unit="Epochs"):
        total_loss = 0
        model.train()
        num_loops=0
        for (spt_batch, spc_batch) in zip(spt_loader, spc_loader):
            #Note that normalization must be done before it's called. The unnormalized
            #data needs to be preserved in order to correctly calculate the loss
            loss_mask = lossMask(spt_batch).to(device)
            spt_batch=spt_batch.to(device)
            spc_batch=spc_batch.to(device)
            opt.zero_grad()         #zero gradients each time
            pred = model(spt_batch,spc_batch,mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_force,std_vec_force)
            loss = model.loss(pred,spt_batch,loss_mask,mean_vec_y,std_vec_y)
            loss.backward()         #backpropagate loss
            opt.step()
            total_loss += loss.item()
            num_loops+=1
        total_loss /= num_loops
        losses.append(total_loss)

        #Every tenth epoch, calculate acceleration test loss and disp validation loss
        if epoch % 10 == 0:
            if (args.save_disp_val):
                # save disp evaluation
                test_loss, disp_val_rmse = test(spt_test_loader,spc_test_loader,device,model,mean_vec_x,std_vec_x,mean_vec_edge,
                                 std_vec_edge,mean_vec_force,std_vec_force,mean_vec_y,std_vec_y, args.save_disp_val)
                disp_val_losses.append(disp_val_rmse.item())
            else:
                test_loss, _ = test(spt_test_loader,spc_test_loader,device,model,mean_vec_x,std_vec_x,mean_vec_edge,
                                 std_vec_edge,mean_vec_force,std_vec_force,mean_vec_y,std_vec_y, args.save_disp_val)

            test_losses.append(test_loss.item())

            # saving model
            model_PATH = PATH_ROOT/Path(args.checkpoint_dir)
            model_PATH.mkdir(parents = True, exist_ok = True)
            PATH = model_PATH/Path(model_name+'.csv')
            df.to_csv(PATH,index=False)

            #save the model if the current one is better than the previous best
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_model = copy.deepcopy(model)

        else:
            #If not the tenth epoch, append the previously calculated loss to the
            #list in order to be able to plot it on the same plot as the training losses
            if (args.save_disp_val):
              test_losses.append(test_losses[-1])
              disp_val_losses.append(disp_val_losses[-1])

        if (args.save_disp_val):
            newrow = pd.Series({'epoch': epoch,'train_loss': losses[-1],
                            'test_loss':test_losses[-1],
                           'disp_val_loss': disp_val_losses[-1]})
            df = pd.concat([df, newrow.to_frame().T], ignore_index=True)
        else:
            newrow = pd.Series({'epoch': epoch, 'train_loss': losses[-1], 'test_loss': test_losses[-1]})
            df = pd.concat([df, newrow.to_frame().T], ignore_index=True)
        if(epoch%100==0):
            # if (args.save_disp_val):
            #     print("train loss: ", str(round(total_loss, 2)),
            #           "| test loss: ", str(round(test_loss.item(), 2)),
            #           "| velo loss: ", str(round(disp_val_rmse.item(), 5)))
            # else:
            #     print("train loss: ", str(round(total_loss,2)), "| test loss: ", str(round(test_loss.item(),2)))

            if(args.save_best_model):

                PATH = model_PATH/Path(model_name+'.pt')
                torch.save(best_model.state_dict(), PATH )

    return test_losses, losses, disp_val_losses, best_test_loss




    

def test(spt_loader,spc_loader,device,test_model,
         mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_force,std_vec_force,mean_vec_y,std_vec_y, is_validation):
  
    '''
    Calculates test set losses and validation set errors.
    '''

    loss=0
    disp_rmse = 0
    num_loops=0

    for (spt_data, spc_data) in zip(spt_loader, spc_loader):
        loss_mask = lossMask(spt_data).to(device)
        spt_data=spt_data.to(device)
        spc_data=spc_data.to(device)
        with torch.no_grad():

            #calculate the loss for the model given the test set
            pred = test_model(spt_data, spc_data, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_force, std_vec_force)
            loss += test_model.loss(pred, spt_data, loss_mask, mean_vec_y, std_vec_y)

            #calculate validation error if asked to
            if (is_validation):                
                eval_disp = unnormalize(pred[:], mean_vec_y, std_vec_y )
                
                error = torch.sum((eval_disp - spt_data.y[:]) ** 2, axis=1)
                if loss_mask == torch.tensor([1], device=device):
                    disp_rmse += torch.sqrt(torch.mean(error))
                else:
                    disp_rmse += torch.sqrt(torch.mean(error[loss_mask]))

        num_loops+=1

    return loss/num_loops, disp_rmse/num_loops


def save_plots(args, losses, test_losses):
    model_name='model_spectralGNN_w_X_StressType_nl'+str(args.num_layers)+'_bs'+str(args.batch_size) + \
               '_hd'+str(args.hidden_dim)+'_ep'+str(args.epochs)+'_wd'+str(args.weight_decay) + \
               '_lr'+str(args.lr)+'_shuff_'+str(args.shuffle)+'_tr'+str(args.train_size)+'_te'+str(args.test_size)

    model_PATH = PATH_ROOT/Path(args.checkpoint_dir)
    model_PATH.mkdir(parents = True, exist_ok = True)

    PATH = model_PATH/Path(model_name + '.pdf')

    f = plt.figure()
    plt.title('Losses Plot')
    plt.plot(losses[00:], label="training loss")
    plt.plot(test_losses[00:], label="validation loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    f.savefig(PATH, bbox_inches='tight')
