from utils import read_images, read_annotation, create_histogram
import collections


def create_image_shape_histogram(images):

    shapes = {}
    for image in images:
        try:
            shapes[str(image.shape)] += 1
        except KeyError:
            shapes[str(image.shape)] = 1
    create_histogram(shapes)


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

    create_histogram(
        collections.OrderedDict(sorted(aspect_ratios.items())),
        title=f"Aspect ratios histogram for {object}",
        savefig_location=f"utils/histograms/{object}_aspect_ratio.png",
    )


if __name__ == "__main__":
    """images = read_images("data/tdt4265_2022/images/train/")
    create_image_shape_histogram(images)"""
    annotations_train = read_annotation("data/tdt4265_2022/train_annotations.json")
    create_object_aspect_ratio_histogram(annotations_train, object="car")
