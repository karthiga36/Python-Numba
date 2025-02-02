import numpy as np
import cv2  # OpenCV for fast image resizing
from skimage import data
from joblib import Parallel, delayed
from line_profiler import LineProfiler
from numba import jit

# Load dataset
imgs = np.uint8(data.lfw_subset() * 255)

@jit(nopython=True, parallel=True)  # Just-In-Time Compilation for speed
def resize_image(im, new_size):
    """Resize a single image using OpenCV (high-speed interpolation)."""
    return cv2.resize(im, new_size, interpolation=cv2.INTER_AREA)

def res_skimage(imgs):
    """Resize all images in parallel using OpenCV batch processing."""
    new_size = (imgs.shape[2] // 2, imgs.shape[1] // 2)  # OpenCV uses (width, height)

    # **FASTEST METHOD: OpenCV batch processing (avoids Python loops)**
    resized_images = cv2.dnn.blobFromImages(imgs, scalefactor=1.0, size=new_size, swapRB=False, crop=False)

    return resized_images.squeeze()  # Remove extra batch dimension

# Create profiler instance
lp = LineProfiler()
lp.add_function(res_skimage)

# Start profiling
lp.enable()
res_im = res_skimage(imgs)  # Run the function
lp.disable()

# Print profiling results
lp.print_stats()
