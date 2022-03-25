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


def create_histogram(dictionary, title, savefig_location=None):
    names = list(dictionary.keys())
    values = list(dictionary.values())

    plt.bar(range(len(dictionary)), values, tick_label=names)
    plt.title(title)
    if savefig_location is not None:
        plt.savefig(savefig_location)
