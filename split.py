#音声特徴量と調音データを訓練データ、検証データ、開発データに分ける。

import os
import shutil

# データフォルダの指定
articulatory_folder = "increase_M1-2/articulatory/padded"  # 調音位置データフォルダ
audio_folder = "increase_M1-2/audio/padded"               # 音声特徴量フォルダ

# 分割後の保存先フォルダ
output_base = "increase_M1-2/split"
evaluation_folder = os.path.join(output_base, "evaluation")  # テストデータ
development_folder = os.path.join(output_base, "development")  # 検証データ
training_folder = os.path.join(output_base, "training")  # 訓練データ

# 必要ならフォルダを作成
os.makedirs(evaluation_folder, exist_ok=True)
os.makedirs(development_folder, exist_ok=True)
os.makedirs(training_folder, exist_ok=True)

# 分割範囲を定義
ranges = {
    "evaluation": range(1, 51),  # 001~050
    "development": range(51, 101),  # 051~100
    "training": range(101, 504)  # 101~503
}

# 分割処理
for data_type, file_range in ranges.items():
    # 出力先フォルダ
    articulatory_output = os.path.join(output_base, data_type, "articulatory")
    audio_output = os.path.join(output_base, data_type, "audio")
    os.makedirs(articulatory_output, exist_ok=True)
    os.makedirs(audio_output, exist_ok=True)
    
    # 指定された範囲のデータをコピー
    for file_id in file_range:
        for offset in [0, 4, 8, 12, 16]:  # 各オフセットについて処理
            articulatory_file = os.path.join(articulatory_folder, f"{file_id:03d}_{offset}ms.csv")
            audio_file = os.path.join(audio_folder, f"{file_id:03d}_{offset}ms.pt")
            
            # ファイルが存在する場合にコピー
            if os.path.exists(articulatory_file):
                shutil.copy(articulatory_file, articulatory_output)
            if os.path.exists(audio_file):
                shutil.copy(audio_file, audio_output)

print("データ分割が完了しました。")
