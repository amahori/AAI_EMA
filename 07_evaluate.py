#ベストモデルを評価し、テスト損失を算出する。推定した調音データを保存する。

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import os
from tqdm import tqdm  # 進捗表示ライブラリ

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

# 保存したモデルのパスとテストデータのパスを指定
model_path = "increase_M1-2/trans_head16.pt"
test_audio_folder = "increase_M1-2/split/evaluation/audio"
test_articulatory_folder = "increase_M1-2/split/evaluation/articulatory"
output_folder = "increase_M1-2/predictions_2"

# テストデータのロード
test_dataset = ArticulatoryDataset(test_audio_folder, test_articulatory_folder)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # 1サンプルずつ処理

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルのロード
model = SpeechToArticulationTransformer(input_dim=768, hidden_dim=512, output_dim=18, num_heads=16, num_layers=4, dropout_rate=0.3)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# 損失関数を定義
criterion = MaskedMSELoss()

# 出力フォルダの作成
os.makedirs(output_folder, exist_ok=True)

# テストデータでの予測と損失計算
test_loss = 0.0
print("Evaluating on test data and saving predictions...")
with torch.no_grad():
    for idx, (audio_features, articulatory_data, mask) in enumerate(tqdm(test_loader, desc="Predicting")):
        # 入力データをデバイスに転送
        audio_features, articulatory_data, mask = (
            audio_features.to(device),
            articulatory_data.to(device),
            mask.to(device),
        )
        # モデルで予測
        predictions = model(audio_features)
        predictions = predictions.squeeze(0)  # バッチ次元を除去してnumpy配列に変換
        predictions *= mask.unsqueeze(-1).squeeze(0)

        # 損失を計算
        loss = criterion(predictions=torch.tensor(predictions, device=device), targets=articulatory_data, mask=mask)
        test_loss += loss.item()

        # ファイル名を決定し、保存
        output_filename = os.path.join(output_folder, f"prediction_{idx + 1:04d}.csv")
        pd.DataFrame(predictions.cpu().numpy()).to_csv(output_filename, index=False, header=False)

# テスト損失の平均を計算
test_loss /= len(test_loader)
print(f"Test Loss (Masked MSE): {test_loss:.4f}")
print(f"Test loss saved to {test_loss_file}")
