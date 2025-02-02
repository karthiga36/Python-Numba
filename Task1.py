import numpy as np
from skimage import data
from skimage.transform import resize
from line_profiler import LineProfiler

# Load dataset
imgs = np.uint8(data.lfw_subset() * 255)

# Define the function
def res_skimage(imgs):
    res_im = []
    new_size = (imgs[0].shape[0] // 2, imgs[0].shape[1] // 2)  # Ensure correct indexing

    for im in imgs:
        image_resized = resize(im, new_size, anti_aliasing=True)
        res_im.append(image_resized)

    return np.asarray(res_im)

# Create profiler instance
lp = LineProfiler()
lp.add_function(res_skimage)  # Add function AFTER defining it

# Start profiling
lp.enable()
res_im = res_skimage(imgs)  # Run the function
lp.disable()

# Print profiling results
lp.print_stats()
