import net
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


from PIL import Image

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
OUTPUT_NEURONS = 4

model = net.ConvNet(OUTPUT_NEURONS)
model.load_state_dict(torch.load("./conv_model.pth"))


def plot_bounding_box(image):
    image_base = Image.open(image)
    image_base = image_base.convert("L")

    image = torch.from_numpy(np.array(image_base)).to(torch.float).unsqueeze(0).unsqueeze(0)
    
    bbox = model(image).detach().numpy()[0]

    fig, ax = plt.subplots()
    ax.imshow(image_base)

    x, y, width, height = bbox
    rect = patches.Rectangle(
        (x, y),
        width,
        height,
        linewidth=2,
        edgecolor="r",
        facecolor="none"
    )

    ax.add_patch(rect)
    plt.show()

plot_bounding_box("samples/30.png")
