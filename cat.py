from pathlib import Path

import numpy as np
from PIL import Image

images = [np.array(Image.open(img)) for img in sorted(Path("figs").iterdir())]
img = np.concatenate(images, axis=1)
Image.fromarray(img).save("training.png")
