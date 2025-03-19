from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, spa_data_list, spc_data_list):
        super().__init__()
        # Store spatial and spectral data lists as class variables
        self.spa_data_list = (
            spa_data_list  # List of spatial data (e.g., node/edge features)
        )
        self.spc_data_list = (
            spc_data_list  # List of spectral data (e.g., spectrum of graph Laplacian)
        )
        assert len(self.spa_data_list) == len(self.spc_data_list)

    def __len__(self):
        return len(self.spa_data_list)

    def __getitem__(self, idx):
        # Retrieve the spatial and spectral data pair at the given index
        return self.spa_data_list[idx], self.spc_data_list[idx]
