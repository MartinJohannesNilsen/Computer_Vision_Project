python -m torch.distributed.launch train.py --rect --img 1024 --batch 32 --epochs 100 --data dataset.yaml --weights yolov5s.pt --optimizer Adam --cache

python detect.py --weights runs/train/exp/weights/best.pt --img 1024 --source ../datasets/tdt4265/images/val --data data/dataset.yaml --conf 0.1 --iou-thres 0.2
--line-thickness 1