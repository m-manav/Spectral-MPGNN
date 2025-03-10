import torch
import random
import numpy as np
from pathlib import Path

from InputNormalize import InputNormalize
from ObjectView import *
from train_test import *
from visualize import *
from conf import *


dataset = torch.load(PATH_ROOT/Path("data/processed_train_set.pt"))

num_train_graphs = 512

spt_dataset = dataset["spt_data_list"][:num_train_graphs]
spc_dataset = dataset["spc_data_list"][:num_train_graphs]

inputNormalizer = InputNormalize()


for args in [
        {'model_type': 'meshgraphnet',  
         'num_layers': 5,
         'batch_size': 32, 
         'hidden_dim': 8, 
         'epochs': 5000,
         'opt': 'adam', 
         'opt_scheduler': 'none', 
         'opt_restart': 0, 
         'weight_decay': 5e-4, 
         'lr': 0.001,
         'train_size': int(448), 
         'test_size': int(64), 
         'device':'cuda',
         'shuffle': True, 
         'save_disp_val': True,
         'save_best_model': True, 
         'checkpoint_dir': 'best_models/',
         'postprocess_dir': 'postprocess_dir/'},
    ]:
        args = objectview(args)



torch.manual_seed(1)  #Torch
random.seed(1)        #Python
np.random.seed(1)     #NumPy


if(args.shuffle):
  spt_dataset_shuf = []
  spc_dataset_shuf = []
  idx_shuf = list(range(len(spt_dataset)))
  random.shuffle(idx_shuf)
  for i in idx_shuf:
    spt_dataset_shuf.append(spt_dataset[i])
    spc_dataset_shuf.append(spc_dataset[i])
  spt_dataset = spt_dataset_shuf
  spc_dataset = spc_dataset_shuf

stats_list = inputNormalizer.get_stats(spt_dataset)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.device = device


##########################################
# training the model
test_losses, losses, disp_val_losses, best_test_loss = train(spt_dataset, spc_dataset, device, stats_list, args)
save_plots(args, losses, test_losses)


##########################################
# testing the model
test_dataset = torch.load(PATH_ROOT/Path("data/processed_test_set.pt"))
spt_test_dataset = test_dataset["spt_data_list"][:]
spc_test_dataset = test_dataset["spc_data_list"][:]

# load model. 
args.device = torch.device('cpu') # animation function cannot work with data on GPU
num_node_features = spt_test_dataset[0].x.shape[1]
num_edge_features = spt_test_dataset[0].edge_attr.shape[1]
num_sp_node_features = spc_test_dataset[0].x.shape[1]
num_sp_edge_features = spc_test_dataset[0].edge_attr.shape[1]
num_ext_force = spt_test_dataset[0].del_ext_force.shape[1]
num_classes = spt_test_dataset[0].y.shape[1]
model_name='model_spectralGNN_w_X_StressType_nl'+str(args.num_layers)+'_bs'+str(args.batch_size) + \
               '_hd'+str(args.hidden_dim)+'_ep'+str(args.epochs)+'_wd'+str(args.weight_decay) + \
               '_lr'+str(args.lr)+'_shuff_'+str(args.shuffle)+'_tr'+str(args.train_size)+'_te'+str(args.test_size)
model_PATH = PATH_ROOT/Path(args.checkpoint_dir + model_name+'.pt')
model = SpectralMeshGraphNet(num_node_features, num_edge_features, num_sp_node_features, num_sp_edge_features, 
                            args.hidden_dim, num_ext_force, num_classes, args).to(args.device)

model.load_state_dict(torch.load(model_PATH, map_location=args.device))

# visualize predicted displacement
dataset_vis = spt_test_dataset
animation_name = model_name+'_current position_anim.gif'
anim_path = PATH_ROOT/Path(args.postprocess_dir)
anim_path.mkdir(parents = True, exist_ok = True)
anim_path = anim_path/Path(animation_name)
eval_data_loader = visualize(dataset_vis, spc_test_dataset, model, anim_path, args, stats_list, skip = 1)

def main():
    print("Hello from spectral-mpgnn!")


if __name__ == "__main__":
    main()
