import os
import random

import matplotlib.image as mpimg
import matplotlib

import numpy as np

if __name__ == "__main__":

    img_dir = ""

    imsize = 256
    width = 5
    height = 3

    grid = np.zeros((imsize * height, imsize * width, 3))

    full_paths = []
    class_paths = os.listdir(img_dir)
    for class_path in class_paths:
        file_paths = os.listdir(os.path.join(img_dir, class_path))
        for file_path in file_paths:
            full_paths.append(os.path.join(img_dir, class_path, file_path))

    for i in range(height):
        for j in range(width):
            index = random.randint(0, len(full_paths) - 1)
            full_path = full_paths.pop(index)
            img = mpimg.imread(full_path)
            grid[i*imsize : (i+1)*imsize, j*imsize : (j+1)*imsize] = img

    matplotlib.image.imsave('image_grid.png', grid)



    print(full_paths[0])
    print(len(full_paths))