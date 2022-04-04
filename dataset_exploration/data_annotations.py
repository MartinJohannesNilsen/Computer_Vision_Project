from dataset_exploration.utils import create_histogram
from tops.config import instantiate, LazyConfig
from ssd import utils
import sys
from tabulate import tabulate


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
    annotations = {}
    for batch in dataloader:
        for label in batch["labels"].detach().tolist()[0]:
            try:
                annotations[cfg.label_map[label]] += 1
            except KeyError:
                annotations[cfg.label_map[label]] = 1

    total = sum(annotations.values())
    sorted_annotations = sorted(annotations.items(), key=lambda x: x[1], reverse=True)

    table = list(
        map(
            lambda x: [x[0], x[1], f"{round((x[1]/total)*100, 2)}%"],
            sorted_annotations,
        )
    )
    f = open("dataset_exploration/tables/data_annotations.txt", "w")
    f.write(tabulate(table, headers=["Object", "Occurences", "In %"]))

    print("Saved data_annotations.txt")

    create_histogram(
        annotations,
        x_label="Objects",
        y_label="Count",
        title=f"Labels distribution",
        rotation=0,
        savefig_location=f"dataset_exploration/histograms/labels_distribution.png",
        fontsize=10,
    )
    print("Saved labels_distribution.png")


def main():
    config_path = "configs/tdt4265.py"
    cfg = get_config(config_path)
    dataset_to_analyze = "train"  # or "val"

    print("Label map is:", cfg.label_map)

    dataloader = get_dataloader(cfg, dataset_to_analyze)
    analyze_something(dataloader, cfg)


if __name__ == "__main__":
    main()
