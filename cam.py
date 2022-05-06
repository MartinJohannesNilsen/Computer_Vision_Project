import os
from pyexpat import model
import shutil
import cv2
from matplotlib.pyplot import gray
import torch
from torch.autograd import Variable
from tqdm import tqdm
from dataset_exploration.save_images_with_annotations import get_dataloader
from performance_assessment.save_comparison_images import (
    convert_image_to_hwc_byte,
    get_config,
    get_trained_model,
    visualize_model_predictions_on_image,
)
from ssd import utils
from pathlib import Path
import click
import numpy as np
import tops
from tops.config import instantiate
from PIL import Image
from pytorch_grad_cam import EigenCAM
import torchvision
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image

# Most of the code is used from https://github.com/jacobgil/pytorch-grad-cam/blob/master/tutorials/Class%20Activation%20Maps%20for%20Object%20Detection%20With%20Faster%20RCNN.ipynb
# There are some changes to fit our dataset


class ScoreTarget:
    """For every original detected bounding box specified in "bounding boxes",
    assign a score on how the current bounding boxes match it,
        1. In IOU
        2. In the classification score.
    If there is not a large enough overlap, or the category changed,
    assign a score of 0.

    The total score is the sum of all the box scores.
    """

    def __init__(self, labels, bounding_boxes, iou_threshold=0.5):
        self.labels = labels
        self.bounding_boxes = bounding_boxes
        self.iou_threshold = iou_threshold

    def __call__(self, model_outputs):
        output = torch.Tensor([0])
        pred_boxes, pred_labels, pred_scores = model_outputs
        if torch.cuda.is_available():
            output = output.cuda()

        if len(pred_boxes) == 0:
            return output

        for box, label in zip(self.bounding_boxes, self.labels):
            box = torch.Tensor(box[None, :])
            if torch.cuda.is_available():
                box = box.cuda()

            ious = torchvision.ops.box_iou(box, pred_boxes)
            index = ious.argmax()
            if ious[0, index] > self.iou_threshold and pred_labels[index] == label:
                score = ious[0, index] + pred_scores[index]
                output = output + score
        return output


def convert_relative_boxes_to_absolute(boxes, height=128, width=1024):
    boxes[:, [0, 2]] *= width
    boxes[:, [1, 3]] *= height
    return boxes


def predict(input_tensor, model, device, detection_threshold):
    model_output_boxes, model_output_labels, model_output_scores = model(input_tensor)[
        0
    ]
    pred_classes = [categories[i] for i in model_output_labels.cpu().numpy()]
    pred_labels = model_output_labels.cpu().numpy()
    pred_scores = model_output_scores.detach().cpu().numpy()
    pred_bboxes = model_output_boxes.detach().cpu().numpy()

    pred_bboxes = convert_relative_boxes_to_absolute(
        pred_bboxes,
    )

    boxes, classes, labels, indices = [], [], [], []
    for index in range(len(pred_scores)):
        if pred_scores[index] >= detection_threshold:
            boxes.append(pred_bboxes[index].astype(np.int32))
            classes.append(pred_classes[index])
            labels.append(pred_labels[index])
            indices.append(index)
    boxes = np.int32(boxes)

    return boxes, classes, labels, indices


def draw_boxes(boxes, labels, classes, image):
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        image = np.ascontiguousarray(image, dtype=np.uint8)
        cv2.rectangle(
            image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 1
        )
        cv2.putText(
            image,
            classes[i],
            (int(box[0]), int(box[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            lineType=cv2.LINE_AA,
        )
    return image


categories = [
    "__background__",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "scooter",
    "person",
    "rider",
]


# This will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(categories), 3))


def renormalize_cam_in_bounding_boxes(boxes, image_float_np, grayscale_cam):
    """Normalize the CAM to be in the range [0, 1]
    inside every bounding boxes, and zero outside of the bounding boxes."""
    renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
    images = []
    for x1, y1, x2, y2 in boxes:
        if x1 < 0:
            x1 = 0
        if x2 < 0:
            x2 = 0
        if y1 < 0:
            y1 = 0
        if y2 < 0:
            y2 = 0
        img = renormalized_cam * 0
        img[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
        images.append(img)
    renormalized_cam = np.max(np.float32(images), axis=0)
    renormalized_cam = scale_cam_image(renormalized_cam)
    eigencam_image_renormalized = show_cam_on_image(
        image_float_np, renormalized_cam, use_rgb=True
    )
    return eigencam_image_renormalized


def reshape_transform(x):
    # x is a tuple of 6 features (which makes sense :) )
    target_size = x[4].size()[
        -2:
    ]  # Select the feature map of highest resolution (index 0)
    activations = []

    for value in x:
        activations.append(
            torch.nn.functional.interpolate(
                torch.abs(value), target_size, mode="bilinear", align_corners=False
            )
        )
    activations = torch.cat(activations, axis=1)
    return activations


def create_cam_image(
    cam, batch, model, img_transform, device, renormalized, score_threshold=0.5
):
    image = convert_image_to_hwc_byte(batch["image"])
    image_float_np = np.float32(image) / 255
    # define the torchvision image transforms
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
        ]
    )

    input_tensor = transform(image)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = input_tensor.to(device)
    input_tensor = input_tensor.unsqueeze(0)

    transformed_image = img_transform({"image": input_tensor})["image"]
    boxes, classes, labels, indices = predict(
        transformed_image, model, device, score_threshold
    )

    targets = [ScoreTarget(labels=labels, bounding_boxes=boxes)]
    grayscale_cam = cam(input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    if renormalized:
        cam_image = renormalize_cam_in_bounding_boxes(
            boxes, image_float_np, grayscale_cam
        )
    else:
        cam_image = show_cam_on_image(image_float_np, grayscale_cam, use_rgb=False)
    image_width_predicted_boxes = draw_boxes(boxes, labels, classes, cam_image)

    return image_width_predicted_boxes


@click.command()
@click.argument(
    "config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option("-n", "--n-images", default=100, type=int)
@click.option("-r", "--renormalized", is_flag=True, default=False)
@click.option("-c", "--threshold", default=0.5, type=float)
def main(config_path: Path, n_images: int, renormalized, threshold):
    try:
        cfg = get_config(config_path)
    except TypeError:
        cfg = get_config(str(config_path, "utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_trained_model(cfg)
    model.eval()
    dataset_to_visualize = "val"
    dataloader = get_dataloader(cfg, dataset_to_visualize)
    num_images_to_save = min(len(dataloader), n_images)
    img_transform = instantiate(cfg.data_val.gpu_transform)
    dataloader = iter(dataloader)
    count = 0
    save_folder = "cam_results"
    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)
    os.makedirs(save_folder)

    target_layers = [model.feature_extractor]
    cam = EigenCAM(
        model,
        target_layers,
        use_cuda=torch.cuda.is_available(),
        reshape_transform=reshape_transform,
    )
    cam.uses_gradients = False
    for i in tqdm(range(num_images_to_save)):
        batch = next(dataloader)
        cam_image = create_cam_image(
            cam,
            batch,
            model,
            img_transform,
            cfg.label_map,
            renormalized,
            threshold,
        )
        cv2.imwrite(f"{save_folder}/{count}.jpg", cam_image)
        count += 1


if __name__ == "__main__":
    main()
