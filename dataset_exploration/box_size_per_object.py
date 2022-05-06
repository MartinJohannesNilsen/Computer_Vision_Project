import collections
from dataset_exploration.utils import create_histogram, read_annotation
from tops.config import instantiate, LazyConfig
from ssd import utils
import sys
from tabulate import tabulate
import numpy as np


def get_config(config_path):
    cfg = LazyConfig.load(config_path)
    cfg.train.batch_size = 1
    return cfg


def get_dataloader(cfg, dataset_to_visualize):
    if dataset_to_visualize == "train":
        # Remove GroundTruthBoxesToAnchors transform
        cfg.data_train.dataset.transform.transforms = (
            cfg.data_train.dataset.transform.transforms[:-1]
        )
        data_loader = instantiate(cfg.data_train.dataloader)
    else:
        cfg.data_val.dataloader.collate_fn = utils.batch_collate
        data_loader = instantiate(cfg.data_val.dataloader)

    return data_loader


def create_box_size_per_table(annotations, object, type):
    category = next(
        item for item in annotations["categories"] if item["name"] == object
    )
    category_id = category["id"]
    filtered_annotations = list(
        filter(lambda x: x["category_id"] == category_id, annotations["annotations"])
    )
    if type == "avg":
        total_width = 0
        total_height = 0
        count = 0
        for annotation in filtered_annotations:
            total_width += annotation["bbox"][2]
            total_height += annotation["bbox"][3]
            count += 1

        avg_width = 0 if count == 0 else total_width / count
        avg_height = 0 if count == 0 else total_height / count

        return [object, f"({round(avg_width,1)}, {round(avg_height,1)})"]
    elif type == "max":
        highest = {"width": 0, "height": 0, "area": 0}

        for annotation in filtered_annotations:
            width = annotation["bbox"][2]
            height = annotation["bbox"][3]
            area = annotation["area"]
            if area > highest["area"]:
                highest["width"] = width
                highest["height"] = height
                highest["area"] = area
        return [object, f"({round(highest['width'],1)}, {round(highest['height'],1)})"]
    else:
        lowest = {"width": 0, "height": 0, "area": np.inf}

        for annotation in filtered_annotations:
            width = annotation["bbox"][2]
            height = annotation["bbox"][3]
            area = annotation["area"]
            if area < lowest["area"]:
                lowest["width"] = width
                lowest["height"] = height
                lowest["area"] = area
        return [object, f"({round(lowest['width'],1)}, {round(lowest['height'],1)})"]


def analyze_something(dataloader, cfg):
    f = open("dataset_exploration/tables/object_width_height.txt", "w")
    annotations_train = read_annotation("data/tdt4265_2022/train_annotations.json")
    min = []
    max = []
    avg = []
    for category in annotations_train["categories"]:
        max.append(
            create_box_size_per_table(
                annotations_train, object=category["name"], type="max"
            )
        )

        min.append(
            create_box_size_per_table(
                annotations_train, object=category["name"], type="min"
            )
        )
        avg.append(
            create_box_size_per_table(
                annotations_train, object=category["name"], type="avg"
            )
        )

    f.write("------max (width, height)------\n")
    f.write(tabulate(max, headers=["Object", "(width, height)"]))
    f.write("\n\n")
    f.write("------min (width, height)------\n")
    f.write(tabulate(min, headers=["Object", "(width, height)"]))
    f.write("\n\n")
    f.write("------avg (width, height)------\n")
    f.write(tabulate(avg, headers=["Object", "(width, height)"]))


def main():
    config_path = "configs/tdt4265.py"
    cfg = get_config(config_path)
    dataset_to_analyze = "train"  # or "val"

    print("Label map is:", cfg.label_map)

    dataloader = get_dataloader(cfg, dataset_to_analyze)
    analyze_something(dataloader, cfg)


if __name__ == "__main__":
    main()
