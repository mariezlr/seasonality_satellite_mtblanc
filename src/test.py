from utils import *
from data_exploration import *
from main import *
import numpy as np
import matplotlib.pyplot as plt

print("Pixels slope < 10° :", np.sum((slope_arr < 10) & mask_arr))
print("Pixels slope ≥ 15° :", np.sum((slope_arr >= 15) & mask_arr))

def plot_slope_classes(slope_arr, mask_arr):
    """
    Quick visual check of slope classes.
    Green  : slope < 10°
    Red    : slope ≥ 15°
    Black  : masked / ignored
    """

    # Init map with NaNs
    slope_class = np.full(slope_arr.shape, np.nan)

    # Classes
    slope_class[(slope_arr < 10) & mask_arr] = 0
    slope_class[(slope_arr >= 15) & mask_arr] = 1

    plt.figure(figsize=(6, 6))
    im = plt.imshow(slope_class, cmap="RdYlGn_r", interpolation="none")
    plt.colorbar(im, label="Slope class")

    plt.axis("off")
    plt.tight_layout()
    plt.show()

plot_slope_classes(slope_arr, mask_arr)


plt.imshow(((slope_arr < 10) - (slope_arr >= 15)) * mask_arr, cmap="bwr")
plt.colorbar()
plt.show()


print("Pixels slope < 10° :", np.sum((slope_arr < 10) & mask_arr))
print("Pixels slope ≥ 15° :", np.sum((slope_arr >= 15) & mask_arr))
