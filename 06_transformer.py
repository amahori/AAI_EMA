#transformerモデルのエンコーダを使って、音声特徴量から調音データを推定する。
＃ベストモデルと学習曲線を保存する。後から学習曲線のスケールを変更するときのために、損失値データも保存しておく。

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# データセットクラス
class ArticulatoryDataset(Dataset):
    def __init__(self, audio_feature_folder, articulatory_data_folder):
        self.audio_feature_files = sorted(
            [os.path.join(audio_feature_folder, f) for f in os.listdir(audio_feature_folder) if f.endswith(".pt")]
        )
        self.articulatory_data_files = sorted(
            [os.path.join(articulatory_data_folder, f) for f in os.listdir(articulatory_data_folder) if f.endswith(".csv")]
        )

    def __len__(self):
        return len(self.audio_feature_files)

    def __getitem__(self, idx):
        audio_features = torch.load(self.audio_feature_files[idx], weights_only=True)  # 音声特徴量
        articulatory_data = torch.tensor(pd.read_csv(self.articulatory_data_files[idx]).values, dtype=torch.float32)  # 調音位置データ

        # マスクを作成（ゼロ以外の部分を1、ゼロの部分を0に設定）
        mask = (articulatory_data.sum(dim=-1) != 0).float()  # 各フレームの全次元が0でない部分を有効とする

        return audio_features.squeeze(0), articulatory_data, mask

# Transformerモデル定義（Encoderのみ）
class SpeechToArticulationTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_layers, dropout_rate):
        super(SpeechToArticulationTransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)  # 入力特徴量を埋め込み次元に変換
        self.positional_encoding = PositionalEncoding(hidden_dim, dropout_rate)

        # Transformer Encoderのみを使用
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dropout=dropout_rate, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(hidden_dim, output_dim)  # 出力を調音位置次元に変換

    def forward(self, x, mask=None, key_padding_mask=None):
        """
        Parameters:
        - x: 入力テンソル (batch_size, seq_len, input_dim)
        - mask: ソースシーケンス用のマスク (seq_len, seq_len)
        - key_padding_mask: パディング位置のマスク (batch_size, seq_len)
        """
        x = self.input_proj(x)  # 入力次元変換
        x = self.positional_encoding(x)  # 位置エンコーディングを追加
        x = self.transformer_encoder(x, mask=mask, src_key_padding_mask=key_padding_mask)  # エンコーダ処理
        output = self.fc(x)  # 全結合層で出力次元に変換
        return output

# Positional Encodingの定義
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout_rate, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# カスタム損失関数（マスク処理付き）
class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, predictions, targets, mask):
        squared_diff = (predictions - targets) ** 2
        masked_loss = (squared_diff * mask.unsqueeze(-1)).sum() / mask.sum()
        return masked_loss

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, save_path="increase_M1-2/trans_2.pt"):
        """
        Parameters:
        - patience: 検証損失が改善しない場合に許容されるエポック数
        - verbose: Trueの場合、進捗を表示
        - save_path: ベストモデルを保存するパス
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.save_path = save_path

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss:
            # 検証損失が改善した場合
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)  # モデルを保存
            if self.verbose:
                print(f"Validation loss improved to {val_loss:.4f}. Model saved.")
        else:
            # 検証損失が改善しない場合
            self.counter += 1
            if self.verbose:
                print(f"Validation loss did not improve. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

if __name__ == "__main__":
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ハイパーパラメータ
    input_dim = 768
    hidden_dim = 512
    num_heads = 16
    num_layers = 4
    output_dim = 18
    dropout_rate = 0.3
    learning_rate = 0.0005
    batch_size = 8
    num_epochs = 1000
    patience = 30
    save_path = "increase_M1-2/trans_head16.pt"
    loss_curve_path = "increase_M1-2/trans_head16.png"

    # Early Stoppingの初期化
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # データローダーの準備
    train_dataset = ArticulatoryDataset("increase_M1-2/split/training/audio", "increase_M1-2/split/training/articulatory")
    dev_dataset = ArticulatoryDataset("increase_M1-2/split/development/audio", "increase_M1-2/split/development/articulatory")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    # モデル、損失関数、最適化手法
    model = SpeechToArticulationTransformer(input_dim, hidden_dim, output_dim, num_heads, num_layers, dropout_rate).to(device)
    criterion = MaskedMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 学習曲線用リスト
    train_losses = []
    dev_losses = []

    # 学習ループ
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        train_loss = 0.0
        for audio_features, articulatory_data, mask in tqdm(train_loader, desc="Training", leave=False):
            audio_features, articulatory_data, mask = audio_features.to(device), articulatory_data.to(device), mask.to(device)
            optimizer.zero_grad()
            outputs = model(audio_features)
            loss = criterion(outputs, articulatory_data, mask)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)  # 損失値を記録
        print(f"Train Loss: {train_loss:.4f}")

        # 検証
        model.eval()
        dev_loss = 0.0
        with torch.no_grad():
            for audio_features, articulatory_data, mask in tqdm(dev_loader, desc="Validation", leave=False):
                audio_features, articulatory_data, mask = audio_features.to(device), articulatory_data.to(device), mask.to(device)
                outputs = model(audio_features)
                loss = criterion(outputs, articulatory_data, mask)
                dev_loss += loss.item()

        dev_loss /= len(dev_loader)
        dev_losses.append(dev_loss)  # 損失値を記録
        print(f"Dev Loss: {dev_loss:.4f}")

        # Early Stopping
        early_stopping(dev_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    import json

    # 損失データを保存するファイルのパス
    loss_data_path = "increase_M1-2/loss_data_head16.json"

    # 学習損失と検証損失を保存
    with open(loss_data_path, "w") as f:
        json.dump({"train_losses": train_losses, "dev_losses": dev_losses}, f)

    print(f"Loss data saved to {loss_data_path}")


    # 学習曲線のプロットと保存
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(dev_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curve")
    plt.legend()
    plt.savefig(loss_curve_path)  # 学習曲線を保存
    plt.close()

    print(f"Loss curve saved to {loss_curve_path}")
