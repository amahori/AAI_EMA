#503×5=2515個のデータのフレーム数を揃えないといけないので、ゼロパディングで揃える。

import os
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import pandas as pd

# 入力データと出力データのフォルダ、およびフレーム数の記録ファイルを指定
input_folder = "data_M1-2/audio/20180911hpf_wav_16k"           # 入力wavデータのフォルダ
output_folder = "increase_M1-2/audio/20ms" # 出力特徴量保存先フォルダ
frame_counts_file = "increase_M1-2/audio/frame_counts.csv"  # フレーム数記録ファイル
os.makedirs(output_folder, exist_ok=True)

# GPUが使用可能か確認し、デバイスを設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用するデバイス: {device}")

# Wav2Vec2モデルとプロセッサのロード
model_name = "facebook/wav2vec2-base"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2Model.from_pretrained(model_name).to(device)  # モデルをGPUに送る

# スタートオフセットを定義（サンプリングレートが16kHzの場合、4ms = 64サンプル）
start_offsets = [0, 4, 8, 12, 16]
sample_rate = 16000
offset_samples = [int(offset * sample_rate / 1000) for offset in start_offsets]

# フレーム数記録用リスト
frame_counts = []

# 入力フォルダ内のwavファイルを処理
for file_name in sorted(os.listdir(input_folder)):
    if file_name.endswith(".wav"):
        # 入力wavファイルをロード
        file_path = os.path.join(input_folder, file_name)
        waveform, sr = torchaudio.load(file_path)
        
        # サンプリングレートが異なる場合はリサンプリング
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
            waveform = resampler(waveform)
        
        # 各オフセットに基づいて特徴量を抽出
        for offset, samples in zip(start_offsets, offset_samples):
            if waveform.size(1) > samples:  # オフセットが波形長より小さい場合のみ処理
                # 波形をオフセットに応じて切り取る
                trimmed_waveform = waveform[:, samples:]
                
                # 特徴量を抽出
                inputs = processor(trimmed_waveform.squeeze(0),
                                   sampling_rate=sample_rate, 
                                   return_tensors="pt", 
                                   padding=True)
                inputs = {key: val.to(device) for key, val in inputs.items()}  # 入力をGPUに送る
                
                with torch.no_grad():
                    features = model(**inputs).last_hidden_state
                
                # 出力ファイル名を生成
                output_file = os.path.join(output_folder, f"{file_name[:-4]}_{offset}ms.pt")
                
                # 特徴量を保存
                torch.save(features, output_file)
                print(f"特徴量を {output_file} に保存しました。")
                
                # フレーム数を記録
                frame_counts.append(features.size(1))  # フレーム数（時間次元）

# フレーム数をCSVに保存
frame_counts_df = pd.DataFrame(frame_counts, columns=["frame_count"])
frame_counts_df.to_csv(frame_counts_file, index=False)

print(f"処理が完了しました。すべてのファイルは {output_folder} に保存されました。フレーム数は {frame_counts_file} に記録されています。")
