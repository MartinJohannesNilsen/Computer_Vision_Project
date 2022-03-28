import os
import cv2
import json
import matplotlib.pyplot as plt


def read_images(path):
    images = []
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename))
        if img is not None:
            images.append(img)
    return images


def read_annotation(path):
    with open(path) as f:
        data = json.load(f)
    return data


def create_histogram(
    dictionary,
    title,
    savefig_location=None,
    x_label="",
    y_label="",
    rotation=40,
    fontsize=5,
):
    names = list(dictionary.keys())
    values = list(dictionary.values())
    plt.clf()
    plt.bar(range(len(dictionary)), values, tick_label=names)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.xticks(rotation=rotation, fontsize=fontsize)
    if savefig_location is not None:
        plt.savefig(savefig_location)
