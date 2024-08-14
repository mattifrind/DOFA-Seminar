import numpy as np
from torchgeo.datasets import BigEarthNet
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

dataset_path = "data/bigearthnet/"
dataset = BigEarthNet(
    root=dataset_path,
    split="train",  # or "val", "test"
    bands="s1",
    num_classes=43,  # or 19 depending on the task
    download=False,
    checksum=False
)

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def getRGBImageFromDataset():
    i = np.random.randint(len(dataset))
    data = dataset.__getitem__(i)
    img = data["image"]
    vv = normalize(img[0])
    vh = normalize(img[1])
    vvhv = normalize(vv * vh)
    rgb = np.stack([vv, vh, vvhv], axis=-1)
    return rgb

columns = 10
rows = 6

images = [getRGBImageFromDataset() for i in range(columns * rows)]

fig = plt.figure(figsize=(16., 16.))
grid = ImageGrid(fig, 111,
                 nrows_ncols=(rows, columns),
                 axes_pad=0.1,
                 )

for ax, im in zip(grid, images):
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(im)
plt.savefig("example-images.png")
plt.show()