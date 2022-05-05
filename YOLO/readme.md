## YOLOv5
Implementation of state-of-the-art model YOLOv5 by ultralytics. YOLO is known for its speed and lightness, while getting the similar results as other high-performing models.

YOLOv5 comes in 5 sizes:
- `YOLOv5n`: Nano
- `YOLOv5s`: Small
- `YOLOv5m`: Medium
- `YOLOv5l`: Large
- `YOLOv5x`: XLarge
One of these are passed in with the `weights` flag during training.

![Models](https://github.com/ultralytics/yolov5/releases/download/v1.0/model_comparison.png)

&nbsp;

## Data structure and preprocessing
- Clone [*ultralytics/YOLOv5*](https://github.com/ultralytics/yolov5)
- Create folder `datasets` on same level
    - In this folder we would like a folder for the name of the dataset
    - Inside the folder of each dataset, we would like two folders `images` and `labels`, which both have a folder for train and val data
- Labels need to be converted into YOLO format. See provided notebook for this conversion.
- Inside `Yolov5/data`, a `.yaml`-file need to be defined with the following information
```Py
"""
Dataset.yaml
"""
# Paths
path: ..
train: datasets/tdt4265/images
val: datasets/tdt4265/images

# Classes
nc: 9
names: [
        "background",
        "car",
        "truck",
        "bus",
        "motorcycle",
        "bicycle",
        "scooter",
        "person",
        "rider"
        ]
```

**TLDR; The following data structure is needed:**
```
.
├── yolov5
│   ├── data
│   |   └── dataset.yaml
│   ├── train.py
│   └── detect.py
└── datasets
    └── tdt4265
        ├── images
        |   ├── train
        |   |   └── trip007_glos_Video00000_0.png
        |   └── val
        |       └── trip007_glos_Video00003_0.png
        └── labels
            ├── train
            |   └── trip007_glos_Video00000_0.txt
            └── val
                └── trip007_glos_Video00003_0.txt
```
Label format:
```
1 0.5135 0.8156 0.0431 0.3688
7 0.5233 0.5652 0.0029 0.0676
8 0.2825 0.5411 0.0089 0.1101
```
Which is defined as:

`class x_center y_center width height`

With normalized values for flexibility regarding change of height and width of image.

More information is available at the following [guide](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data), in addition to the notebook `coco2yolov5.ipynb`.



&nbsp;

## How to train the YOLO network

```
torchrun train.py --rect --img 1024 --batch 32 --epochs 100 --data dataset.yaml --weights yolov5s.pt --optimizer Adam --workers 2
```
- **rect**: define that the images are non-square
- **img**: define input image size of largest side
- **batch**: determine batch size
- **epochs**: define the number of training epochs.
- **data**: Our dataset locaiton is saved in the dataset.location
- **weights**: specify a path to weights to start transfer learning from. Options are the sizes:
-- ""
- **optimizer**: Select optimizer {SGD, Adam, AdamW}
- **workers**: define amount of workers, default 8 but recommended 2 in project
- **cache**: cache images for faster training. Did not have enough RAM available on cluster for this, but will make training faster.
- **resume**: Resume training where the former run left off, for the given amount of epochs when first ran.


Entire list is defined above the main method in `train.py`.

<details>
<summary>Fix for "ValueError: Error initializing torch.distributed"</summary>
Fix this by running either of the following options:

```torchrun train.py``` 

```python -m torch.distributed.launch train.py```

</details>

&nbsp;

## Detection/inference with trained model

```
python detect.py --weights runs/train/exp/weights/best.pt --img 1024 --source ../datasets/tdt4265/images/val --data data/dataset.yaml --conf 0.5 --iou-thres 0.2 --line-thickness 1
```

- **weights**: specify path of weights as .pt-file
- **img**: define input image size
- **source**: define path to test images
- **conf**: confidence needed for classification
- **iou-thres**: define iou threshold for removing unnecessary bounding box
- **line-thickness**: define line thickness for bounding boxes. Default 3px
- **hide-conf**: remove confidence from drawn boxes
- **hide-labels**: remove label from drawn boxes

Entire list is defined above the main method in `detect.py`.

