import os
from pathlib import Path

root = Path(
    "/home/charatan/projects/NeuStyle/outputs/2023-05-14/03-01-36/wandb/latest-run/files/media/images"
)

for item in root.iterdir():
    index = int(item.name.split("_")[1])
    os.system(f'cp "{item}" animation/chair/frame_{index:0>2}.png')
