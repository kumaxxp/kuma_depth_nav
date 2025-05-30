# KUMA_DEPTH_NAV

単眼カメラの情報を深度に変換し、障害物を回避するソフトを実験的に実装します。
m5stack LLMモジュールで実装します。
m5stack LLMはLinuxモジュールですので、本システムはLinuxのみで動作します。

## ファイル構成

kuma_depth_nav/
├── config.json                 # 設定ファイル
├── fast_camera_streaming.py    # メインアプリケーション
├── utils.py                    # ユーティリティ関数（ロギング、設定読み込み）
├── linux_optimization.py       # Linuxシステム最適化機能
├── depth_processor/
│   ├── __init__.py
│   ├── depth_model.py          # 深度推定モデル関連
│   ├── point_cloud.py          # 点群処理関連
│   └── visualization.py        # 可視化関連
├── run.sh                      # 起動スクリプト
├── tests/                      # テストコード（オプショナル）
│   └── test_depth_processor.py
├── logs/                       # ログ出力ディレクトリ
│   └── .gitkeep
└── systemd/                    # systemdサービス設定
    └── kuma-depth-nav.service

```
```
kuma_depth_nav/
├── config.json                 # 設定ファイル
├── fast_camera_streaming.py    # メインアプリケーション
├── utils.py                    # ユーティリティ関数（ロギング、設定読み込み）
├── linux_optimization.py       # Linuxシステム最適化機能
├── depth_processor/
│   ├── __init__.py
│   ├── depth_model.py          # 深度推定モデル関連
│   ├── point_cloud.py          # 点群処理関連
│   └── visualization.py        # 可視化関連
├── calibration/
│   ├── camera_calibration.py   # キャリブレーション用ライブラリ
│   ├── calibration_app.py      # キャリブレーション実行アプリ
│   ├── run_calibration.sh      # 実行用シェルスクリプト
│   └── __init__.py
├── calibration_images/         # キャリブレーション画像（jpg等）を配置
├── calibration_data/           # キャリブレーション結果保存先
├── run.sh                      # 起動スクリプト
├── tests/                      # テストコード（オプショナル）
│   └── test_depth_processor.py
├── logs/                       # ログ出力ディレクトリ
│   └── .gitkeep
└── systemd/                    # systemdサービス設定
    └── kuma-depth-nav.service
```

## キャリブレーションの実行方法

1. `calibration_images/` ディレクトリにキャリブレーション画像（jpg等）を配置してください。
2. 以下のコマンドでキャリブレーションを実行します（Linux/Macの場合）:

```sh
cd calibration
sh run_calibration.sh
```

Windowsの場合は、`python calibration/calibration_app.py` を実行してください。

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


----

## 設定

プロジェクトの設定は`config.json`ファイルで管理されています。各設定項目は以下のとおりです。

### カメラ設定

```json
"camera": {
  "device_index": 0,     // カメラデバイスのインデックス（通常、内蔵カメラは0）
  "width": 640,          // キャプチャ画像の幅（ピクセル）
  "height": 480,         // キャプチャ画像の高さ（ピクセル）
  "fps": 30,             // フレームレート（毎秒フレーム数）
  "use_v4l2": true       // Linux環境でのV4L2対応（Windowsでは無視される）
}
```

### 深度推定設定

```json
"depth": {
  "model_type": "DINOv2",      // 使用する深度推定モデル
  "width": 640,                // 深度マップの出力幅
  "height": 480,               // 深度マップの出力高さ
  "use_gpu": true,             // GPUを使用するかどうか
  "visualization_min": 0.1,    // 可視化時の最小深度（相対値）
  "visualization_max": 0.9     // 可視化時の最大深度（相対値）
}
```

### サーバー設定

```json

"server": {
  "host": "0.0.0.0",    // サーバーがバインドするIPアドレス（0.0.0.0はすべてのネットワークインターフェースを意味する）
  "port": 8000,         // サーバーが使用するポート番号
  "debug": false        // デバッグモードの有効/無効
}
```

### 起動方法
プロジェクトを起動するには以下のコマンドを使用します。

```bash
# 実行権限を付与（初回のみ）
chmod +x run.sh

# 起動
./run.sh
```

----

[![Depth Anything: Accelerating Monocular Depth Perception](https://tse1.mm.bing.net/th?id=OIP.SbIO7cLsmvgcyXZhDMs9mQHaCo\&cb=iwc1\&pid=Api)](https://learnopencv.com/depth-anything/)

以下は、Depth Anythingの出力形式、その意味、そして絶対距離を計算するための方法について整理した資料です。

---

# Depth Anything 出力と絶対距離の計算方法

## 📌 出力形式と意味

* **出力形式**: Depth Anythingは、入力画像と同じ解像度の1チャンネルのfloat32型テンソルを出力します。
* **値の意味**: 各ピクセルの値は **視差（disparity）** を表しており、これは「1 / 深度（距離）」に相当します。&#x20;
* **正規化**: 学習時には、深度値を視差空間に変換し、各深度マップ内で0から1の範囲に正規化しています。&#x20;

## ⚠️ 注意点

* **相対値である**: 出力は相対的な「近さ」を示すものであり、絶対的な距離を直接得ることはできません。
* **スケーリングが必要**: 絶対距離を求めるには、既知の距離を持つ物体を基準にスケーリングを行う必要があります。
* **逆数変換の注意**: 出力が視差であるため、単純に逆数を取るだけでは正確な距離を得ることはできません。スケーリングとシフトの調整が必要です。

## 🧮 絶対距離を計算する方法

Depth Anythingの出力を絶対距離（メートル単位）に変換するには、以下の手順を踏む必要があります。

1. **既知の距離を持つ参照点を選定**: 画像内で実際の距離が分かっている点（例：特定の物体までの距離が5メートル）を選びます。

2. **スケーリング係数の計算**: 選定した参照点の出力値（視差）を用いて、スケーリング係数を計算します。

   例えば、参照点の出力値が3で、実際の距離が5メートルの場合：

   ```
   スケーリング係数 = 実際の距離 × 出力値 = 5 × 3 = 15
   ```

3. **全体の距離マップの計算**: スケーリング係数を用いて、全体の距離マップを計算します。

   ```
   実際の距離マップ = スケーリング係数 / 出力値マップ
   ```

   この計算により、各ピクセルの絶対距離を推定できます。

## 📝 まとめ

* **出力値の意味**: Depth Anythingの出力は視差（1 / 距離）を表す相対的な値です。
* **絶対距離の取得**: 既知の距離を持つ参照点を用いてスケーリング係数を計算し、全体の距離マップを推定します。
* **注意点**: 出力は相対値であり、単純な逆数変換では正確な距離を得ることはできません。スケーリングとシフトの調整が必要です。

この方法により、Depth Anythingの出力を実際の距離情報として活用することが可能になります。
