import tifffile as tiff
import numpy as np
a = tiff.imread(r'../GuiTorch/segmentation_data/confuse/gth/benxi375.TIF')

print(np.unique(a))
print(len((np.where(a==15))[0]))
