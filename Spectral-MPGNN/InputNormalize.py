import torch
import numpy as np


# normalization in order to stabilize training
def normalize(to_normalize, mean_vec, std_vec):
    return (to_normalize-mean_vec)/std_vec

def unnormalize(to_unnormalize, mean_vec, std_vec):
    return to_unnormalize*std_vec+mean_vec


class InputNormalize:
    def __init__(self):
        #Define the maximum number of accumulations to perform such that we do
        #not encounter memory issues
        self.max_accumulations = 10**6
        #Define a very small value for normalizing to 
        self.eps=torch.tensor(np.finfo(float).eps)

    def get_stats(self, data_list):
        '''
        Method for normalizing processed datasets. Given  the processed data_list, 
        calculates the mean and standard deviation for the node features, edge features, 
        and node outputs, and normalizes these using the calculated statistics.
        '''

        #mean and std of the node features are calculated
        mean_vec_x = torch.zeros(data_list[0].x.shape[1:])
        std_vec_x = torch.zeros(data_list[0].x.shape[1:])

        #mean and std of the edge features are calculated
        mean_vec_edge = torch.zeros(data_list[0].edge_attr.shape[1:])
        std_vec_edge = torch.zeros(data_list[0].edge_attr.shape[1:])

        #mean and std of the incremental external force are calculated
        mean_vec_force = torch.zeros(data_list[0].del_ext_force.shape[1:])
        std_vec_force = torch.zeros(data_list[0].del_ext_force.shape[1:])

        #mean and std of the output parameters are calculated
        mean_vec_y = torch.zeros(data_list[0].y.shape[1:])
        std_vec_y = torch.zeros(data_list[0].y.shape[1:])

        #Define counters used in normalization
        num_accs_x = 0
        num_accs_edge=0
        num_accs_force=0
        num_accs_y=0

        #Iterate through the data in the list to accumulate statistics
        for dp in data_list:

            #Add to the 
            mean_vec_x += torch.sum(dp.x,dim=0)
            std_vec_x += torch.sum(dp.x**2,dim=0)
            num_accs_x += dp.x.shape[0]

            mean_vec_edge += torch.sum(dp.edge_attr,dim=0)
            std_vec_edge += torch.sum(dp.edge_attr**2,dim=0)
            num_accs_edge += dp.edge_attr.shape[0]

            mean_vec_force += torch.sum(dp.del_ext_force,dim=0)
            std_vec_force += torch.sum(dp.del_ext_force**2,dim=0)
            num_accs_force += dp.del_ext_force.shape[0]

            mean_vec_y += torch.sum(dp.y,dim=0)
            std_vec_y += torch.sum(dp.y**2,dim=0)
            num_accs_y += dp.y.shape[0]

            if(num_accs_x>self.max_accumulations or num_accs_edge>self.max_accumulations or num_accs_force>self.max_accumulations or num_accs_y>self.max_accumulations):
                break

        mean_vec_x = mean_vec_x/num_accs_x
        std_vec_x = torch.maximum(torch.sqrt(std_vec_x/num_accs_x - mean_vec_x**2), self.eps)

        mean_vec_edge = mean_vec_edge/num_accs_edge
        std_vec_edge = torch.maximum(torch.sqrt(std_vec_edge/num_accs_edge - mean_vec_edge**2), self.eps)

        mean_vec_force = mean_vec_force/num_accs_force
        std_vec_force = torch.maximum(torch.sqrt(std_vec_force/num_accs_force - mean_vec_force**2), self.eps)

        mean_vec_y = mean_vec_y/num_accs_y
        std_vec_y = torch.maximum(torch.sqrt(std_vec_y/num_accs_y - mean_vec_y**2), self.eps)

        mean_std_list=[mean_vec_x, std_vec_x, 
                       mean_vec_edge, std_vec_edge,
                       mean_vec_force, std_vec_force,
                       mean_vec_y, std_vec_y]

        return mean_std_list