from dis import show_code
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
    x_label="",
    y_label="",
    xtick_rotation=None,
    xtick_fontsize=None,
    ytick_rotation=None,
    ytick_fontsize=None,
    show=False,
    savefig_location=None,
    figsize=None,
    rotation=10,
    fontsize=10,
):
    names = list(dictionary.keys())
    values = list(dictionary.values())
    plt.clf()
    if figsize:
        plt.figure(figsize=figsize)
    plt.bar(range(len(dictionary)), values, tick_label=names)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()

    if xtick_fontsize:
        plt.xticks(fontsize=xtick_fontsize)
    if xtick_rotation:
        plt.xticks(rotation=xtick_rotation)
    if ytick_fontsize:
        plt.yticks(fontsize=ytick_fontsize)
    if ytick_rotation:
        plt.yticks(rotation=ytick_rotation)
    if savefig_location:
        plt.savefig(savefig_location)
    if show:
        plt.show()


def create_matrix():
    pass
