import torch
from matplotlib import animation
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
from pathlib import Path
from InputNormalize import unnormalize
from conf import *
from mask import updateMask

def make_animation(gs, pred, err, anim_path, skip = 2, save_anim = True, plot_variables = False):
    '''
    input gs is a dataloader and each entry contains attributes of many timesteps.

    '''
    # print('Generating displacement fields...')
    fig, axes = plt.subplots(2, 1, figsize=(20, 16))
    num_steps = len(gs) # for a single trajectory
    num_frames = num_steps // skip
    # print(num_steps)
    def animate(num):
        step = (num*skip) % num_steps
        traj = 0

        bb_min = gs[-1].current_node_pos.min()
        bb_max = gs[-1].current_node_pos.max()
        bb_min_evl = err[-1].current_node_pos.min()
        bb_max_evl = err[-1].current_node_pos.max()
        count = 0

        for ax in axes:
            ax.cla()
            if (count == 0):
                # ground truth
                position_gs = gs[step].current_node_pos
                position_pred = pred[step].current_node_pos
                title = 'Ground truth and predicted current position'
                ax.plot(position_gs, torch.zeros(position_gs.shape), 'ko-', linewidth = 3.0)
                ax.plot(position_pred, torch.zeros(position_pred.shape), 'r*-', linewidth = 2.0)
                ax.set_xlim(bb_min, bb_max)
            else:
                position_gs = gs[step].current_node_pos
                position_err = err[step].current_node_pos
                title = 'Error: (Prediction - Ground truth)'
                ax.plot(position_gs, position_err, 'ko-')
                ax.set_xlim(bb_min, bb_max)
                ax.set_ylim(-0.25, 0.25)
                ax.grid(True, axis = 'y')
            
            ax.set_title('{} Trajectory {} Step {}'.format(title, traj, step), fontsize = '20')
            
            count += 1
        return fig,
    
    if (save_anim):
        gs_anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=1000)
        writergif = animation.PillowWriter(fps=2) 
        gs_anim.save( anim_path, writer=writergif)
        # plt.show(block=True)
    else:
        pass

def visualize(loader, spc_dataset, best_model, anim_path, args, stats_list, skip = 1):

    best_model.eval()
    device = args.device
    viz_data_set = copy.deepcopy(loader)
    gs_data_set = copy.deepcopy(loader)
    err_data_set = copy.deepcopy(loader)
    [mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_force,std_vec_force,mean_vec_y,std_vec_y] = stats_list
    (mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_force,std_vec_force,mean_vec_y,std_vec_y)=(mean_vec_x.to(device),
            std_vec_x.to(device),mean_vec_edge.to(device),std_vec_edge.to(device),mean_vec_force.to(device),std_vec_force.to(device),
            mean_vec_y.to(device),std_vec_y.to(device))

    err_data_set[0].current_node_pos = 0.0*err_data_set[0].current_node_pos
    ref_node_pos = gs_data_set[0].current_node_pos

    total_disp_sq_error = torch.tensor([0])
    total_stress_sq_error = torch.tensor([0])
    n_disp_err = 0
    n_stress_err = 0

    for idx, viz_data in enumerate(viz_data_set[:-1]):
        update_mask = updateMask(viz_data)
        
        gs_data_next = gs_data_set[idx+1]
        viz_data_next = viz_data_set[idx+1]
        err_data_next = err_data_set[idx+1]

        viz_data = viz_data.to(args.device)
        spc_data = spc_dataset[idx].to(args.device)

        with torch.no_grad():
            pred = best_model(viz_data,spc_data,mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_force,std_vec_force)
            # pred gives the learnt normalized output increament
            eval = unnormalize(pred, mean_vec_y, std_vec_y)
            incr_disp = eval[:, 0].reshape(-1, 1)
            if eval.shape[1] > 1:
                pred_stress = eval[:, 1:].reshape(-1, eval.shape[1]-1)

            # error in predicted displacement
            viz_data_next.current_node_pos[update_mask] = viz_data.current_node_pos[update_mask] + incr_disp[update_mask]
            pred_disp = viz_data_next.current_node_pos - ref_node_pos
            disp = gs_data_next.current_node_pos - ref_node_pos
            error_disp = torch.abs(pred_disp - disp)/torch.abs(disp)
            err_data_next.current_node_pos[update_mask] = error_disp[update_mask]
            total_disp_sq_error = total_disp_sq_error + torch.sum(err_data_next.current_node_pos**2)
            n_disp_err = n_disp_err + error_disp.shape[0]

            # updating node features (stress) and calculating error in predicted stress
            if eval.shape[1] > 1:
                viz_data_next.x[:, 0:pred_stress.shape[1]] = pred_stress
                stress = gs_data_next.x[:, 0:pred_stress.shape[1]]
                error_stress = torch.abs(pred_stress - stress)/torch.abs(stress)
                total_stress_sq_error = total_stress_sq_error + torch.sum(error_stress**2)
                n_stress_err = n_stress_err + error_stress.shape[0]

            #updating edge features using the output of gnn
            uC_i = viz_data_next.current_node_pos[viz_data_next.edge_index[0]]
            uC_j = viz_data_next.current_node_pos[viz_data_next.edge_index[1]]
            uC_ij = uC_i-uC_j
            uC_ij_norm = torch.norm(uC_ij, p=2, dim=1, keepdim=True)

            # uncomment the following line if edge feature includes current relative position as well
            # viz_data_next.edge_attr[:, 2:] = torch.cat((uC_ij, uC_ij_norm), dim=-1)
    
    total_disp_error = torch.sqrt(total_disp_sq_error/n_disp_err)
    total_stress_error = torch.sqrt(total_stress_sq_error/n_stress_err)
    
    with open(PATH_ROOT/Path("total_prediction_error.txt"), 'w') as err_file:
        err_file.write(f"Total disp error in prediction: {total_disp_error.item()}")
        err_file.write(f"\nTotal stress error in prediction: {total_stress_error.item()}")

    make_animation(gs_data_set, viz_data_set, err_data_set, anim_path,
                      skip, True, False)

    return err_data_set
