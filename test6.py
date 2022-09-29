from pathlib import Path
import os
pa = Path(__file__).parent / "results/VOC8classAlexnet_v3"
print(pa)
print(Path(__file__).parent / "results/VOC8classAlexnet_v3")
if not os.path.isdir(pa):
    os.mkdir(pa)
