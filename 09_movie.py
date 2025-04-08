#推定した調音運動を映像にする。

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 2つのデータのパス
file_path1 = "increase_M1-2/articulatory/padded/001_0ms.csv"  # 1つ目のCSVファイル
file_path2 = "test_predictions/001_0ms.csv"  # 2つ目のCSVファイル

# データの読み込み
data1 = pd.read_csv(file_path1)
data2 = pd.read_csv(file_path2)

# y座標とz座標を抽出
y_coords1 = data1.iloc[:, ::2].values  # 偶数列: y座標
z_coords1 = data1.iloc[:, 1::2].values  # 奇数列: z座標
y_coords2 = data2.iloc[:, ::2].values  # 偶数列: y座標
z_coords2 = data2.iloc[:, 1::2].values  # 奇数列: z座標

# アニメーションの設定
fig, ax = plt.subplots()
sc1 = ax.scatter([], [], c='blue', s=50, label='Dataset 1')  # データセット1
sc2 = ax.scatter([], [], c='red', s=50, label='Dataset 2')  # データセット2

# FPS 設定
fps = 50  # フレームレート（1秒あたりのフレーム数）

# 軸の範囲設定
ax.set_xlim(min(y_coords1.min(), y_coords2.min()) - 1, max(y_coords1.max(), y_coords2.max()) + 1)
ax.set_ylim(min(z_coords1.min(), z_coords2.min()) - 1, max(z_coords1.max(), z_coords2.max()) + 1)
ax.set_xlabel("Y Coordinate")
ax.set_ylabel("Z Coordinate")

# 時間表示
time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')
ax.legend()

# フレーム更新関数
def update(frame):
    time_in_seconds = frame * 0.02  # フレームごとに 20ms (0.02秒) 進む
    time_text.set_text(f"Time: {time_in_seconds:.2f} s")
    sc1.set_offsets(np.c_[y_coords1[frame], z_coords1[frame]])
    sc2.set_offsets(np.c_[y_coords2[frame], z_coords2[frame]])
    return sc1, sc2, time_text

# アニメーションの作成
ani = FuncAnimation(fig, update, frames=min(len(y_coords1), len(y_coords2)), interval=1000/fps, blit=True)

# 動画を保存
video_path = "trans8_01_0.mp4"  # 保存先
ani.save(video_path, writer="ffmpeg", fps=fps)
plt.close()

print(f"動画が保存されました: {video_path}")
