from torch_geometric.nn import GATConv

class GATModel(LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, class_weights, learning_rate, dropout_rate):
        super(GATModel, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.class_weights = class_weights
        self.dropout_rate = dropout_rate

        self.conv1 = GATConv(input_dim, hidden_dim, heads=4, concat=True)  # Attention with multiple heads
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=1, concat=False)  # Final single-head attention
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = Dropout(dropout_rate)

    def forward(self, batch):
        x, edge_index, batch_indices = batch.x, batch.edge_index, batch.batch
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = global_mean_pool(x, batch_indices)
        x = self.fc(x)
        return x
