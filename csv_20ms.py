#調音データを20ms間隔で取り出す。スタートを4msずつずらす。

import pandas as pd
import os

# 調音位置データのフォルダと保存先フォルダを指定
input_folder = "data_M1-2/articulatory/alignment"  # 入力データフォルダ
output_folder = "increase_M1-2/articulatory/20ms"  # 出力データフォルダ
frame_counts_file = "increase_M1-2/articulatory/frame_counts.csv"  # フレーム数記録ファイル

# 必要なら出力フォルダを作成
os.makedirs(output_folder, exist_ok=True)

# スタート位置を設定
start_offsets = [0, 1, 2, 3, 4]  # オフセットを4ms単位に対応
step = 5  # 20ms ÷ 4ms = 5

# フレーム数記録用リスト
frame_counts = []

# 001.csv から 503.csv まで処理
for file_id in range(1, 504):
    # 入力ファイル名を生成
    input_file = os.path.join(input_folder, f"{file_id:03d}.csv")
    
    # データを読み込む
    data = pd.read_csv(input_file)
    
    # 各スタート位置でサンプリングして保存
    for offset in start_offsets:
        # 修正: オフセットを4ms単位に直接対応
        sampled_data = data.iloc[offset::step]  # offset から step 間隔で抽出
        
        # 出力ファイル名を生成
        output_file = os.path.join(output_folder, f"{file_id:03d}_{offset * 4}ms.csv")
        sampled_data.to_csv(output_file, index=False)
        
        # フレーム数を記録
        frame_counts.append(len(sampled_data))

# フレーム数をCSVに保存（列名を省略）
frame_counts_df = pd.DataFrame(frame_counts)
frame_counts_df.to_csv(frame_counts_file, index=False, header=False)

print(f"処理が完了しました。すべてのファイルは {output_folder} に保存されました。フレーム数は {frame_counts_file} に記録されています。")
