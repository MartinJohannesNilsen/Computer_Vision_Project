import collections
from dataset_exploration.utils import create_histogram, read_annotation
from tops.config import instantiate, LazyConfig
from ssd import utils
import sys


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


def create_object_aspect_ratio_histogram(annotations, object):
    aspect_ratios = {}
    print(f"Object is: {object}")
    category = next(
        item for item in annotations["categories"] if item["name"] == object
    )
    category_id = category["id"]
    filtered_annotations = list(
        filter(lambda x: x["category_id"] == category_id, annotations["annotations"])
    )

    for annotation in filtered_annotations:
        height = annotation["bbox"][3]
        width = annotation["bbox"][2]
        aspect_ratio = round(height / width, 1)
        try:
            aspect_ratios[aspect_ratio] += 1
        except KeyError:
            aspect_ratios[aspect_ratio] = 1
    max_value = None
    if bool(aspect_ratios):
        max_value = max(aspect_ratios.values())

    if bool(max_value):
        aspect_ratios = {
            k: v for (k, v) in aspect_ratios.items() if v > max_value * 0.1
        }
    create_histogram(
        collections.OrderedDict(sorted(aspect_ratios.items())),
        x_label="Aspect ratios",
        y_label="Objects",
        title=f"Aspect ratios histogram for {object}",
        savefig_location=f"dataset_exploration/histograms/{object}_aspect_ratio.png",
    )


def analyze_something(dataloader, cfg):
    annotations_train = read_annotation("data/tdt4265_2022/train_annotations.json")
    for category in annotations_train["categories"]:
        create_object_aspect_ratio_histogram(annotations_train, object=category["name"])


def main():
    config_path = "configs/tdt4265.py"
    cfg = get_config(config_path)
    dataset_to_analyze = "train"  # or "val"

    print("Label map is:", cfg.label_map)

    dataloader = get_dataloader(cfg, dataset_to_analyze)
    analyze_something(dataloader, cfg)


if __name__ == "__main__":
    main()
