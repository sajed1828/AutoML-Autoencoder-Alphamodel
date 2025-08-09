import h2o
from h2o.automl import H2OAutoML

h2o.init()   

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np

class AutoML:
    def __init__(self, train_data, target_column, max_runtime_secs=3600):
        self.train_data = train_data
        self.target_column = target_column
        self.max_runtime_secs = max_runtime_secs

    def run_automl(self):
        aml = H2OAutoML(max_runtime_secs=self.max_runtime_secs, seed=42)
        aml.train(y=self.target_column, training_frame=self.train_data)
        return aml.leaderboard

    def save_model(self, model, path):
        h2o.save_model(model=model, path=path, force=True)


if __name__ == "__main__":
    models = ["LSTMModel", "GRUModel", "TransformerModel", "CNNModel"]
    
    for model in models.item().get():
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        Model = model().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(Model.parameters(), lr=0.01)
        

