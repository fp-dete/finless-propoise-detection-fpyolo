# YOLOv5 + DeepSORT

This project uses YOLOv5 for object detection and DeepSORT for continuous object tracking. Please download the source code from the official GitHub repositories to avoid interface differences or compatibility issues associated with third-party implementations.

## 1. Download the Source Code

Run the following commands in a terminal:

```bash
git clone https://github.com/ultralytics/yolov5.git
git clone https://github.com/nwojke/deep_sort.git
```

The resulting directory structure should be:

```text
project/
└──yolov5/
└── deep_sort/
```

If necessary, create a project directory before cloning the repositories:

```bash
mkdir project
cd project

git clone https://github.com/ultralytics/yolov5.git
git clone https://github.com/nwojke/deep_sort.git
```

## 2. Create the Environment

It is recommended to create an independent Python environment using Conda:

```bash
conda create -n yolov5_deepsort python=3.8 -y
conda activate yolov5_deepsort
```

## 3. Install the YOLOv5 Dependencies

```bash
cd yolov5
pip install -r requirements.txt
cd ..
```

## 4. Install the DeepSORT Dependencies

To run the tracker using existing detection results:

```bash
cd deep_sort
pip install -r requirements.txt
cd ..
```

To generate appearance features using a GPU, follow the official DeepSORT instructions:

```bash
cd deep_sort
pip install -r requirements-gpu.txt
cd ..
```

> Note: The official DeepSORT implementation was released several years ago, and its appearance-feature model depends on an older version of TensorFlow. Compatibility issues may occur when using recent versions of Python, TensorFlow, CUDA, or cuDNN. If DeepSORT is only used with YOLOv5 detection results, the Kalman filter, matching cascade, and track-management modules in `deep_sort/` can be retained while the appearance-feature extraction component is adapted to the current environment.

## 5. CBAM Integration

The CBAM implementation was adapted from the official code released by the original authors:

```bash
https://github.com/Jongchan/attention-module
```

In the modified YOLOv5 backbone, the CBAM modules are embedded into the early downsampling stages:
```bash
backbone:
  [[-1, 1, Focus, [64, 6, 2, 2]],       # 0-P1/2
   [-1, 1, Conv_CBAM, [128, 3, 2]],     # 1-P2/4
   [-1, 3, C3, [128]],
   [-1, 1, Conv_CBAM, [256, 3, 2]],     # 3-P3/8
   [-1, 6, C3, [256]],
   [-1, 1, Conv, [512, 3, 2]],          # 5-P4/16
   [-1, 9, C3, [512]],
   [-1, 1, Conv, [1024, 3, 2]],         # 7-P5/32
   [-1, 3, C3, [1024]],
   [-1, 1, SPPF, [1024, 5]],            # 9
  ]
   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 14], 1, Concat, [1]],  # cat head P4
   [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)
   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, Concat, [1]],  # cat head P5
   [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)
   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
```

