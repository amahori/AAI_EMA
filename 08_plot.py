#推定した調音データと正解の調音データの推移をグラフにする。

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# サンプリング周波数（調音位置データが1秒間に250フレームの場合）
sampling_rate = 50  # Hz

# 調音位置の名前
articulatory_points = ["ND", "NA", "UI", "UL", "LL", "T1", "T2", "T3", "LJ"]

# CSVファイルからデータをロード
# 以下のファイルパスを実際の予測データと正解データのCSVファイルパスに置き換えてください
predictions_file = "increase_M1-2/predictions_2/prediction_0001.csv"  # 予測データのCSVファイル
ground_truth_file = "001_0ms.csv"  # 正解データのCSVファイル
output_folder = "articulatory_graphs_2"
os.makedirs(output_folder, exist_ok=True) 

# データを読み込む
predictions = pd.read_csv(predictions_file, header=None).values  # 予測データ
ground_truth = pd.read_csv(ground_truth_file, header=None).values  # 正解データ

# ゼロパディングを無視するためのマスクを作成
# ground_truth のゼロフレームを基準にする（フレーム内のすべての次元がゼロである場合、そのフレームは無視）
mask = ~(ground_truth == 0).all(axis=1)  # 全次元がゼロのフレームを検出
filtered_predictions = predictions[mask]  # マスクを適用した予測データ
filtered_ground_truth = ground_truth[mask]  # マスクを適用した正解データ

# 時間軸を再計算
num_frames = filtered_ground_truth.shape[0]
time_axis = np.linspace(0, num_frames / sampling_rate, num_frames)

# 縦軸の範囲を固定（-120～120）
ymin_global = -120
ymax_global = 0

# 縦軸間隔を設定（例: 10mm間隔）
step_size = 20

# 横軸の範囲を固定（0～5秒）
xmin_global = 0
xmax_global = 5

# 各調音位置について y と z をプロットし、保存
for i, point in enumerate(articulatory_points):
    # Y座標のグラフ
    plt.figure(figsize=(8, 4))
    plt.plot(time_axis, filtered_predictions[:, 2 * i], label=f"Predicted {point}_y", linestyle='--', color='black')
    plt.plot(time_axis, filtered_ground_truth[:, 2 * i], label=f"True {point}_y", linestyle='-', color='black')
    plt.title(f"{point} Y-Coordinate")
    plt.xlabel("Time [s]")
    plt.ylabel("y [mm]")
    plt.legend()
    plt.grid(True)

    # 固定した縦軸と横軸の範囲を設定
    plt.ylim(ymin_global, ymax_global)
    plt.yticks(np.arange(ymin_global, ymax_global + step_size, step_size))
    plt.xlim(xmin_global, xmax_global)

    plt.tight_layout()
    y_graph_path = os.path.join(output_folder, f"{point}_y_coordinate.png")
    plt.savefig(y_graph_path)  # グラフを保存
    plt.close()

    # Z座標のグラフ
    plt.figure(figsize=(8, 4))
    plt.plot(time_axis, filtered_predictions[:, 2 * i + 1], label=f"Predicted {point}_z", linestyle='--', color='black')
    plt.plot(time_axis, filtered_ground_truth[:, 2 * i + 1], label=f"True {point}_z", linestyle='-', color='black')
    plt.title(f"{point} Z-Coordinate")
    plt.xlabel("Time [s]")
    plt.ylabel("z [mm]")
    plt.legend()
    plt.grid(True)

    # 固定した縦軸と横軸の範囲を設定
    plt.ylim(ymin_global, ymax_global)
    plt.yticks(np.arange(ymin_global, ymax_global + step_size, step_size))
    plt.xlim(xmin_global, xmax_global)

    plt.tight_layout()
    z_graph_path = os.path.join(output_folder, f"{point}_z_coordinate.png")
    plt.savefig(z_graph_path)  # グラフを保存
    plt.close()

print(f"Graphs with fixed axes (y: -120 to 120, x: 0 to 5 seconds) saved in folder: {output_folder}")
articulatory point saved in folder: {output_folder}")
