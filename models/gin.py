from torch_geometric.nn import GINConv

class GINModel(LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, class_weights, learning_rate, dropout_rate):
        super(GINModel, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.class_weights = class_weights
        self.dropout_rate = dropout_rate

        nn1 = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim), torch.nn.ReLU(), torch.nn.Linear(hidden_dim, hidden_dim))
        nn2 = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(), torch.nn.Linear(hidden_dim, hidden_dim))
        
        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(nn2)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = Dropout(dropout_rate)

    def forward(self, batch):
        x, edge_index, batch_indices = batch.x, batch.edge_index, batch.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch_indices)
        x = self.fc(x)
        return x
