import numpy as np
from hcan import HCAN, HCANLoss
import torch
import torch.nn as nn
import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset

WINDOW_SIZE = 256
LOOKAHEAD = 21
DTYPE = torch.float32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 1000


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


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=64, num_layers=3, batch_first=True)
        self.proj = nn.Sequential(
            nn.Linear(64, 2),
            nn.ReLU(),
        )
        self.hcan = HCAN(pred_len=LOOKAHEAD, channels=2)
        return

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.lstm(x)
        x = self.proj(x[:, -LOOKAHEAD:, :])
        x = self.hcan(x)
        return x


def train_model():
    data = MyData('train')
    dataloader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE)
    model = MyModel().to(dtype=DTYPE, device=DEVICE)
    model.train()
    criterion = HCANLoss(y_trues=data.y_trues).to(device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(50):
        epoch_loss = 0.0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)['L_TOTAL']
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())

        print(f'Epoch {epoch + 1}, Loss: {epoch_loss / BATCH_SIZE}')

    # evaluation
    model.eval()
    dataloader = torch.utils.data.DataLoader(MyData('test'), batch_size=BATCH_SIZE)
    all_y, all_x = [], []
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)['y_hat']
            all_y.append(targets[..., 0].detach().cpu())
            all_x.append(outputs[..., 0].detach().cpu())

    y = np.sign(torch.cat(all_y, dim=0).numpy())  # (N, T)
    x = np.sign(torch.cat(all_x, dim=0).numpy())  # (N, T)
    print('Accuracy:', np.mean(y == x))
    return


if __name__ == '__main__':
    train_model()
