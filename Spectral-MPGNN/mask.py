import torch

def lossMask(spt_data):
    # loss_mask = torch.logical_or((torch.argmax(spt_data.x[:, 1:], dim=1) == torch.tensor(1)),
    #                              (torch.argmax(spt_data.x[:, 1:], dim=1) == torch.tensor(2)))
    # loss_mask = torch.argmax(spt_data.x[:, 3:], dim=1) == torch.tensor(1)
    loss_mask = torch.tensor([1])

    return loss_mask

def updateMask(spt_data):
    update_mask = torch.logical_or((torch.argmax(spt_data.x[:, 1:], dim=1) == torch.tensor(1)),
                                 (torch.argmax(spt_data.x[:, 1:], dim=1) == torch.tensor(2)))
    # update_mask = torch.argmax(spt_data.x[:, 1:], dim=1) == torch.tensor(1)
    # update_mask = None

    return update_mask