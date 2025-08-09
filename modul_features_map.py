import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel


class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=72):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class genlenDataset(Dataset):
    def __init__(self, df, selected_features, qen_len):
        self.qen_len = qen_len
        self.scaler = MinMaxScaler()

        data = df[selected_features].replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(method="bfill")
        scaled = self.scaler.fit_transform(data)
        self.target = scaled[:, selected_features.index("close")] if "close" in selected_features else scaled[:, 0]
        self.data = scaled

    def __len__(self):
        return len(self.data) - self.qen_len

    def __getitem__(self, idx):
        X = self.data[idx: idx+self.qen_len]
        y = self.target[idx+self.qen_len]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def train_autoencoder(dataloader, model, optimizer, num_epochs=10):
    model.train()
    criterion = nn.MSELoss()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_x, _ in dataloader:
            batch_x_flat = batch_x.view(batch_x.size(0), -1)
            optimizer.zero_grad()
            outputs = model(batch_x_flat)
            loss = criterion(outputs, batch_x_flat)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")


def encode_data(dataloader, model):
    model.eval()
    all_encoded = []
    with torch.no_grad():
        for batch_x, _ in dataloader:
            batch_x_flat = batch_x.view(batch_x.size(0), -1)
            encoded = model.encoder(batch_x_flat)
            all_encoded.append(encoded)
    return torch.cat(all_encoded).numpy()


def run_autoencoder(df, qen_len=72, num_epochs=10, top_n_features=20):
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(method="bfill")

    X = numeric_df.drop(columns=["close"])
    y = numeric_df["close"]

    selector = SelectFromModel(RandomForestRegressor(n_estimators=100), max_features=top_n_features)
    selector.fit(X, y)
    selected_features = list(X.columns[selector.get_support()])
    selected_features.append("close")  

    print(f"✅ test  {len(selected_features)-1} feature encoded: {selected_features}")

    dataset = genlenDataset(numeric_df, selected_features, qen_len)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    model = Autoencoder(input_dim=qen_len * len(selected_features), encoding_dim=72)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_autoencoder(dataloader, model, optimizer, num_epochs)
    encoded_data = encode_data(dataloader, model)

    timestamps = df["timestamp"].values[qen_len:] if "timestamp" in df.columns else list(range(encoded_data.shape[0]))
    encoded_df = pd.DataFrame(encoded_data, columns=[f"encoded_{i}" for i in range(encoded_data.shape[1])])
    encoded_df["timestamp"] = timestamps
    return encoded_df


if __name__ == "__main__":
    path = r"C:\Users\User\Documents\TEST\BITCOIN_PREDICTED.csv"
    df = pd.read_csv(path).head(1000)
    print(df)
    
    # nn.Linear(648, 128)
    
    encoded_df = run_autoencoder(df, qen_len=72, num_epochs=10, top_n_features=20)

    print(encoded_df.head())
    encoded_df.to_csv("encoded_data.csv", index=False)

