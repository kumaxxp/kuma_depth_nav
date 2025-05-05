# KUMA_DEPTH_NAV

単眼カメラの情報を深度に変換し、障害物を回避するソフトを実験的に実装します。
m5stack LLMモジュールで実装します。

## ファイル構成

```

c:\work\kuma_depth_nav\
  |- fast_camera_streaming.py (メインアプリケーション - 更新)
  |- depth_processor/
     |- __init__.py (更新)
     |- depth_model.py (既存)
     |- visualization.py (既存)
     |- point_cloud.py (新規) - 点群・トップビュー関連の処理

```

1. depth_processor/point_cloud.py (新規)
点群処理と占有グリッド生成に関連する関数を実装します。主な関数は：

+ depth_to_point_cloud: 深度マップから3D点群を生成
+ create_top_down_occupancy_grid: 点群から天頂視点の占有グリッドを生成
+ visualize_occupancy_grid: 占有グリッドをカラー画像として可視化

## 機能

1. Linuxに接続したUSBカメラの画像を高速で取り込む
2. 画像にdepth anythingの処理を施して、深度を得る
3. 深度情報を点群に変換して可視化する
4. 通過可能な経路を計算する
5. 過去のデータと現在のデータから、どのようなスロットル・ステアリングを行うべきかを計算する

現在、2を実装完了。ライブラリ化して3の機能を追加する。

