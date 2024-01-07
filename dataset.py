from PIL import Image, ImageDraw
from random import randint
import os
import json
import argparse

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
CIRCLE_RADIUS = 20;

def create_image(x, y):
    # create image
    base_image = Image.new("L",(224, 224))
    draw = ImageDraw.Draw(base_image)
    
    draw.ellipse((x,y, x + CIRCLE_RADIUS,y + CIRCLE_RADIUS), fill=(255))

    return base_image


def create_training_sample(idx: int):

    cx = randint(0,IMAGE_WIDTH)
    cy = randint(0,IMAGE_HEIGHT)
    image = create_image(cx, cy)

    image_path = f"samples/{idx}.png"
    image.save(image_path)

    sample = {
        "x": cx,
        "y": cy,
        "width": CIRCLE_RADIUS,
        "height": CIRCLE_RADIUS,
        "image_path": image_path
    }

    return sample

def build_dataset(training_samples: int):
    # generate samples

    print("Generating samples...")

    samples = []
    for sample in range(training_samples):
        samples.append(create_training_sample(sample))

    # save sample csv

    with open("samples/dataset.json", "w") as f:
        json_ = json.dumps(samples)
        f.write(json_)

    print("Samples saved to file..")

    
if __name__ == "__main__":
    build_dataset(100)
