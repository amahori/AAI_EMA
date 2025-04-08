#csvファイルのほうが音声特徴量よりもフレーム数が多いので、多い部分をカットする。

import pandas as pd
import os

# 調音位置データと音声特徴量のフレーム数のCSVファイル
articulatory_frame_counts_file = "increase_M1-2/articulatory/frame_counts.csv"  # 調音位置データのフレーム数
audio_frame_counts_file = "increase_M1-2/audio/frame_counts.csv"  # 音声特徴量のフレーム数

# データフォルダ
articulatory_folder = "increase_M1-2/articulatory/20ms"  # 調音位置データのフォルダ
output_folder = "increase_M1-2/articulatory/20ms_cut"  # フレーム数を合わせた調音データの出力先
os.makedirs(output_folder, exist_ok=True)

# フレーム数データを読み込む
articulatory_frame_counts = pd.read_csv(articulatory_frame_counts_file, header=None).squeeze("columns")
audio_frame_counts = pd.read_csv(audio_frame_counts_file, header=None).squeeze("columns")

# データの整合性を確認
if len(articulatory_frame_counts) != len(audio_frame_counts):
    raise ValueError("調音位置データと音声特徴量のフレーム数のリストが一致していません！")

# 001_0ms.csv から 503_16ms.csv まで処理
for file_id in range(1, 504):
    for offset in [0, 4, 8, 12, 16]:  # オフセットをループ
        # 入力ファイル名を生成
        input_file = os.path.join(articulatory_folder, f"{file_id:03d}_{offset}ms.csv")
        
        # 対応する音声特徴量のフレーム数を取得
        audio_frames = audio_frame_counts[(file_id - 1) * 5 + offset // 4]
        
        # 調音位置データを読み込む
        articulatory_data = pd.read_csv(input_file)
        
        # フレーム数が一致しない場合、後ろをカット
        if len(articulatory_data) > audio_frames:
            articulatory_data = articulatory_data.iloc[:audio_frames]
        
        # 出力ファイル名を生成
        output_file = os.path.join(output_folder, f"{file_id:03d}_{offset}ms.csv")
        
        # 修正後のデータを保存
        articulatory_data.to_csv(output_file, index=False)

print(f"処理が完了しました。すべてのファイルは {output_folder} に保存されました。")
