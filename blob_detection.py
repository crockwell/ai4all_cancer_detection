import os

import numpy as np

from common import (find_maxima, read_img, visualize_maxima,
                    visualize_scale_space)
# ~~START DELETE~~
from filters import convolve

# ~~END DELETE~~


def gaussian_filter(image, sigma):
    # Given an image, apply a Gaussian filter with the input kernel size
    # and standard deviation
    # Input
    #   image: image of size HxW
    #   sigma: scalar standard deviation of Gaussian Kernel
    #
    # Output
    #   Gaussian filtered image of size HxW
    H, W = image.shape
    # -- good heuristic way of setting kernel size
    kernel_size = int(2 * np.ceil(2 * sigma) + 1)

    # Ensure that the kernel size isn't too big and is odd
    kernel_size = min(kernel_size, min(H, W) // 2)
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1

    # TODO implement gaussian filtering of size kernel_size x kernel_size
    # You can use your convolution function or scipy's convolution function
    output = None
    # ~~START DELETE~~
    k_h = kernel_size // 2
    g_1d = np.arange(-k_h, k_h + 1)[:, None]**2
    g_2d = -1 * (g_1d + g_1d.T)
    g_kern = np.exp(g_2d / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    output = convolve(image, g_kern)
    # ~~END DELETE~~
    return output


def scale_space(image, min_sigma, k=np.sqrt(2), S=8):
    # Calcualtes a DoG scale space of the image
    # Input
    #   image: image of size HxW
    #   min_sigma: smallest sigma in scale space
    #   k: scalar multiplier for scale space
    #   S: number of scales considers
    #
    # Output
    #   Scale Space of size HxWx(S-1)
    output = None
    # ~~START DELETE~~
    sigs = [min_sigma * (k**i) for i in range(S)]
    gaus = [gaussian_filter(image, sigma) for sigma in sigs]
    scales = [gaus[i+1] - gaus[i] for i in range(S - 1)]
    output = np.stack(scales, axis=2)
    # ~~END DELETE~~
    return output


def main():
    image = read_img('polka.png')

    # Create directory for polka_detections
    if not os.path.exists("./polka_detections"):
        os.makedirs("./polka_detections")

    # -- Detecting Polka Dots
    print("Detect small polka dots")
    # -- Detect Small Circles
    sigma_1, sigma_2 = None, None
    gauss_1 = None  # to implenent
    gauss_2 = None  # to implement

    # calculate difference of gaussians
    DoG_small = None  # to implement
    # ~~START DELETE~~
    image = image / 255.
    sigma_1 = 10
    sigma_2 = 14
    gauss_1 = gaussian_filter(image, sigma_1)
    gauss_2 = gaussian_filter(image, sigma_2)
    DoG_small = gauss_2 - gauss_1
    # ~~END DELETE~~

    # visualize maxima
    maxima = find_maxima(DoG_small, k_xy=int(sigma_1))
    visualize_scale_space(DoG_small, sigma_1, sigma_2 / sigma_1,
                          './polka_detections/polka_small_DoG.png')
    visualize_maxima(image, maxima, sigma_1, sigma_2 / sigma_1,
                     './polka_detections/polka_small.png')

    # -- Detect Large Circles
    print("Detect large polka dots")
    sigma_1, sigma_2 = None, None
    gauss_1 = None
    gauss_2 = None

    # calculate difference of gaussians
    DoG_large = None
    # ~~START DELETE~~
    sigma_1 = 30
    sigma_2 = 40
    gauss_1 = gaussian_filter(image, sigma_1)
    gauss_2 = gaussian_filter(image, sigma_2)
    DoG_large = gauss_2 - gauss_1
    # ~~END DELETE~~

    # visualize maxima
    # Value of k_xy is a sugguestion; feel free to change it as you wish.
    maxima = find_maxima(DoG_large, k_xy=10)
    visualize_scale_space(DoG_large, sigma_1, sigma_2 / sigma_1,
                          './polka_detections/polka_large_DoG.png')
    visualize_maxima(image, maxima, sigma_1, sigma_2 / sigma_1,
                     './polka_detections/polka_large.png')

    # TODO Implement scale_space() and try to find both polka dots
    # ~~START DELETE~~
    min_sigma = 12
    k = np.sqrt(1.8)
    ss = scale_space(image, min_sigma, k=k, S=8)
    visualize_scale_space(ss, min_sigma, k, './polka_detections/scale_space.png')
    # ~~END DELETE~~

    # TODO Detect the cells in any one (or more) image(s) from vgg_cells
    # Create directory for polka_detections
    if not os.path.exists("./cell_detections"):
        os.makedirs("./cell_detections")
    # ~~START DELETE~~
    # ~~END DELETE~~


if __name__ == '__main__':
    main()
