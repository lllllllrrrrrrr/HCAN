import numpy as np
from hcan import HCAN, HCANLoss
import torch
import torch.nn as nn
import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import trange

WINDOW_SIZE = 256
LOOKAHEAD = 21
DTYPE = torch.float32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 500


class MyData(Dataset):
    def __init__(self, split: str = 'train'):
        super().__init__()
        df = pd.read_parquet('SPX_sample.parquet')
        df['ret'] = df.close.pct_change().fillna(0)
        df['vola'] = df['ret'].rolling(10).std()
        df = df.drop(columns=['close']).dropna()
        self.y_trues = torch.tensor(df.values, dtype=DTYPE, device=DEVICE)
        l = len(df) - WINDOW_SIZE - LOOKAHEAD
        if split == 'train':
            df = df.iloc[:int(l * .8)]
        else:
            df = df.iloc[int(l * .8):]
        self._df = df
        return

    def __len__(self) -> int:
        return len(self._df) - WINDOW_SIZE - LOOKAHEAD

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        data_window = self._df.iloc[idx: idx + WINDOW_SIZE].values
        target_window = self._df.iloc[idx + WINDOW_SIZE: idx + WINDOW_SIZE + LOOKAHEAD].values
        return (
            torch.tensor(data_window, dtype=DTYPE, device=DEVICE),
            torch.tensor(target_window, dtype=DTYPE, device=DEVICE)
        )


class WithHCAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=64, num_layers=3, batch_first=True)
        self.proj = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        self.hcan = HCAN(pred_len=LOOKAHEAD, channels=2)
        return

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.lstm(x)
        x = self.proj(x[:, -LOOKAHEAD:, :])
        x = self.hcan(x)
        return x


class NoHCAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=64, num_layers=3, batch_first=True)
        self.proj = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        return

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.lstm(x)
        x = self.proj(x[:, -LOOKAHEAD:, :])
        return x


def train_model(model, model_name: str):
    data = MyData('train')
    dataloader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE)
    model = model.to(dtype=DTYPE, device=DEVICE)
    model.train()
    if model_name == 'NoHCAN':
        criterion = nn.MSELoss().to(device=DEVICE)
    else:
        criterion = HCANLoss(y_trues=data.y_trues).to(device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    bar = trange(200, desc='Training ' + model_name, ncols=100)
    for epoch in bar:
        epoch_loss = 0.0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if model_name == 'WithHCAN':
                loss = loss['L_TOTAL']
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
        bar.set_postfix({'loss': epoch_loss / BATCH_SIZE})

    # evaluation
    model.eval()
    dataloader = torch.utils.data.DataLoader(MyData('test'), batch_size=BATCH_SIZE)
    all_y, all_x = [], []
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            if model_name == 'WithHCAN':
                outputs = outputs['y_hat']
            all_y.append(targets.detach().cpu())
            all_x.append(outputs.detach().cpu())

    y = torch.cat(all_y, dim=0).numpy()  # (N, T)
    x = torch.cat(all_x, dim=0).numpy()  # (N, T)
    print(model_name + ' sign accuracy:', np.mean(np.sign(y[..., 0]) == np.sign(x[..., 0])))
    print(model_name + ' returns MSE:', np.mean((y[..., 0] - x[..., 0]) ** 2))
    print(model_name + ' volatility MSE:', np.mean((y[..., 1] - x[..., 1]) ** 2))

    return


if __name__ == '__main__':
    train_model(NoHCAN(), 'NoHCAN')
    train_model(WithHCAN(), 'WithHCAN')