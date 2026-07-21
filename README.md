# Object Counter

Webカメラの映像を YOLOv8 でリアルタイムに解析し、指定した種類の物体を検出・カウントする Python アプリケーションです。人物だけでなく、COCO データセットに含まれる 80 クラスから対象を選択できます。

## 1. 主な機能

- Webカメラ映像のリアルタイム物体検出
- COCO 80クラスからカウント対象を選択
- 検出枠、クラス名、信頼度、検出数を画面に表示
- ONNX 形式の YOLOv8n モデルを同梱

## 2. 必要環境

- Windows、macOS、または Linux
- Python 3.10 以降
- Webカメラ
- 仮想環境の利用を推奨

GPU は必須ではありません。同梱の `yolov8n.onnx` は CPU でも動作しますが、カメラの解像度やPCの性能によって処理速度は変わります。

## 3. セットアップ

### 3.1. リポジトリへ移動

PowerShell の例：

```powershell
cd C:\path\to\object-counter
```

### 3.2. 依存パッケージをインストール

このアプリの実行に必要な主要パッケージをインストールします。

```powershell
python -m pip install ultralytics onnxruntime opencv-python numpy
```

> [!NOTE]
> `requirements.txt` は開発環境全体から出力されており、生成元PC固有の `file:///...` パスを含みます。そのため、別のPCでは上記の最小構成でのインストールを推奨します。

### 3.3. YOLOv8 モデルを ONNX 形式へ変換

`src` ディレクトリへ移動し、YOLOv8n の PyTorch モデルから推論用の ONNX モデルを生成します。

```powershell
cd src
yolo export model=yolov8n.pt format=onnx imgsz=416
```

変換が完了すると、同じディレクトリに `yolov8n.onnx` が作成されます。すでに同名の ONNX モデルがある場合、この手順は省略できます。

## 4. 実行方法

モデルファイルが相対パスで指定されているため、`src` ディレクトリへ移動して実行します。

```powershell
cd src
python main.py
```

起動すると、ターミナルに COCO のクラス一覧が表示されます。カウントしたい対象のクラスIDを入力してください。

例：人物をカウントする場合

```text
Enter the class ID to count (0-79): 0
```

代表的なクラスID：

| ID | クラス |
|---:|---|
| 0 | person（人物） |
| 1 | bicycle（自転車） |
| 2 | car（自動車） |
| 3 | motorcycle（オートバイ） |
| 5 | bus（バス） |
| 7 | truck（トラック） |
| 15 | cat（猫） |
| 16 | dog（犬） |

映像ウィンドウで `Esc` キーを押すと終了します。

## 5. チューニング

設定値は [`src/main.py`](src/main.py) の `ObjectCounter.__init__()` にまとまっています。

```python
self.target_resolution = (1920, 1080)
self.target_fps = 30
self.model_input_size = 416
self.confidence_threshold = 0.25
self.window_size = (800, 600)
```

### 5.1. 検出精度と速度

#### `confidence_threshold`

検出結果として採用する最低信頼度です。

- 値を上げる：誤検出が減る一方、見逃しが増える
- 値を下げる：見逃しが減る一方、誤検出が増える
- 調整例：`0.15` ～ `0.50`

誤検出が多い場合は、まず `0.35` 前後を試してください。

#### `model_input_size`

推論に使用する画像サイズです。

- 小さくする：高速になるが、小さい物体を検出しにくくなる
- 大きくする：精度が上がりやすいが、処理が重くなる
- 調整例：`320`、`416`、`640`

処理が遅い場合は `320`、遠くの人物や小さい物体を検出したい場合は `640` が目安です。

### 5.2. カメラ設定

#### `target_resolution`

カメラへ要求する撮影解像度です。カメラが対応していない値は、利用可能な解像度へ自動的に変更される場合があります。

```python
self.target_resolution = (1280, 720)
```

解像度を下げると、カメラからの読み込み負荷を軽減できます。

#### `target_fps`

カメラへ要求するフレームレートです。

```python
self.target_fps = 15
```

CPU 使用率を抑えたい場合は、`15` または `20` に下げてください。実際のFPSはカメラ性能に依存します。

#### `カメラ番号`

複数のカメラが接続されている場合は、`initialize_camera()` 内の番号を変更します。

```python
self.cap = cv2.VideoCapture(1)
```

通常は `0` が内蔵または最初のカメラ、`1` 以降が追加のカメラです。

### 5.3. モデルの変更

現在は軽量な ONNX モデルを使用しています。

```python
counter = ObjectCounter(model_path="yolov8n.onnx")
```

同梱の PyTorch モデルを使用する場合は、次のように変更できます。

```python
counter = ObjectCounter(model_path="yolov8n.pt")
```

独自学習モデルを使う場合は、その `.pt` または `.onnx` ファイルを `src` に置き、同じ箇所のファイル名を変更してください。独自モデルではクラス数やクラスIDが COCO と異なることがあるため、入力可能範囲の処理もモデルに合わせて変更する必要があります。

## 6. 参考

- 人物検出モデル: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)

## 7. ディレクトリ構成

```text
object-counter/
├── README.md
├── requirements.txt
└── src/
    ├── main.py
    ├── yolov8n.onnx
    └── yolov8n.pt
```
