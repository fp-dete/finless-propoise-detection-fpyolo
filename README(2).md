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

## 5. Verify the YOLOv5 Installation

Enter the YOLOv5 directory and run inference using the official pretrained weights:

```bash
cd yolov5
python detect.py --weights yolov5s.pt --source data/images
```

The detection results are saved by default in:

```text
yolov5/runs/detect/exp/
```

## 6. Official Repositories

- YOLOv5: https://github.com/ultralytics/yolov5
- DeepSORT: https://github.com/nwojke/deep_sort

## 7. Update the Source Code

To retrieve the latest changes, enter each repository and run `git pull`:

```bash
cd yolov5
git pull

cd ../deep_sort
git pull
```
