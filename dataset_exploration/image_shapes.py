from dataset_exploration.utils import create_histogram
import collections
import operator
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


def analyze_something(dataloader, cfg):
    shapes = {}
    for batch in dataloader:
        # Remove the two lines below and start analyzing :D
        # print("The keys in the batch are:", batch.keys())
        shape = (batch["width"][0].item(), batch["height"][0].item())
        try:
            shapes[str(shape)] += 1
        except KeyError:
            shapes[str(shape)] = 1

    create_histogram(
        shapes,
        x_label="Image shape",
        y_label="Count",
        title=f"Image shape histogram",
        savefig_location=f"dataset_exploration/histograms/image_shape.png",
        rotation=0,
    )


def main():
    config_path = "configs/tdt4265.py"
    cfg = get_config(config_path)
    dataset_to_analyze = "train"  # or "val"

    print("Label map is:", cfg.label_map)

    dataloader = get_dataloader(cfg, dataset_to_analyze)
    analyze_something(dataloader, cfg)


if __name__ == "__main__":
    main()
