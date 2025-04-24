import numpy as np
import sys

#https://medium.com/@pasanSK/understanding-and-implementing-canny-edge-detection-in-native-python-52c296255601

def apply_gaussian_blur(image, kernel_size):
    def gaussian_kernel(size, sigma):
        kernel = np.fromfunction(
            lambda x, y: (1/ (2 * np.pi * sigma ** 2)) *
                          np.exp(- ((x - (size-1)/2) ** 2 + (y - (size-1)/2) ** 2) / (2 * sigma ** 2)),
            (size, size)
        )
        return kernel / np.sum(kernel)

    # Ensure the kernel size is odd for symmetry
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Generate the Gaussian kernel
    kernel = gaussian_kernel(kernel_size, sigma=1.0)

    # Get the dimensions of the image
    rows, cols = image.shape

    # Get the half-size of the kernel for padding
    k_half = kernel_size // 2

    # Create an output image with the same dimensions as the input
    output = np.zeros_like(image)

    # Apply the Gaussian blur by convolving the image with the kernel
    for i in range(k_half, rows - k_half):
        for j in range(k_half, cols - k_half):
            output[i, j] = np.sum(image[i - k_half: i + k_half + 1, j - k_half: j + k_half + 1] * kernel)

    # get the image without the padding
    return output[k_half:rows-k_half, k_half:cols-k_half]

def compute_gradient_magnitude_and_orientation(image, sobel_kernel_size):
    """applique un filtre de sobel horizontal et vertical sur l'image"""

    if sobel_kernel_size == 3:
        # Define Sobel kernels (3x3)
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    elif sobel_kernel_size == 5:
        sobel_x = np.array([[-1, -2, 0, 2, 1], [-2, -3, 0, 3, 2], [-3, -5, 0, 5, 3], [-2, -3, 0, 3, 2], [-1, -2, 0, 2, 1]])
        sobel_y = np.array([[-1, -2, -3, -2, -1], [-2, -3, -5, -3, -2], [0, 0, 0, 0, 0], [2, 3, 5, 3, 2], [1, 2, 3, 2, 1]])
    else:
        sys.exit("Sobel kernel size should be 3 or 5!")

    # Get the dimensions of the image
    rows, cols = image.shape

    # Initialize arrays for gradient magnitude and orientation
    gradient_x = np.zeros_like(image, dtype=np.float64)
    gradient_y = np.zeros_like(image, dtype=np.float64)

    # Compute gradient using Sobel operators
    half_size = sobel_kernel_size // 2
    for i in range(half_size, rows - half_size):
        for j in range(half_size, cols - half_size):
            window = image[i - half_size:i + half_size + 1, j - half_size:j + half_size + 1]
            gradient_x[i, j] = np.sum(window * sobel_x)
            gradient_y[i, j] = np.sum(window * sobel_y)

    # Compute gradient magnitude and orientation
    magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    #save_image(magnitude, "3-gradient_magnitude.jpg")
    orientation = np.arctan2(gradient_y, gradient_x)

    return magnitude, orientation


def apply_non_max_suppression(magnitude, orientation):
    # Apply non-maximum suppression to the gradient magnitude
    # This will thin the edges by keeping only the local maxima
    suppressed_magnitude = np.copy(magnitude)
    rows, cols = magnitude.shape

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            angle = orientation[i][j]
            q = [0, 0]
            if (-np.pi / 8 <= angle < np.pi / 8) or (7 * np.pi / 8 <= angle):
                q[0] = magnitude[i][j + 1]
                q[1] = magnitude[i][j - 1]
            elif (np.pi / 8 <= angle < 3 * np.pi / 8):
                q[0] = magnitude[i + 1][j + 1]
                q[1] = magnitude[i - 1][j - 1]
            elif (3 * np.pi / 8 <= angle < 5 * np.pi / 8):
                q[0] = magnitude[i + 1][j]
                q[1] = magnitude[i - 1][j]
            else:
                q[0] = magnitude[i - 1][j + 1]
                q[1] = magnitude[i + 1][j - 1]

            if magnitude[i][j] < max(q[0], q[1]):
                suppressed_magnitude[i][j] = 0

    return suppressed_magnitude


def apply_edge_tracking_by_hysteresis(magnitude, low_threshold, high_threshold):
    # Apply edge tracking by hysteresis to detect strong and weak edges
    rows, cols = magnitude.shape
    edge_map = np.zeros((rows, cols), dtype=np.uint8)

    strong_edge_i, strong_edge_j = np.where(magnitude >= high_threshold)
    weak_edge_i, weak_edge_j = np.where((magnitude >= low_threshold) & (magnitude < high_threshold))

    # mark strong edges as white (255)
    edge_map[strong_edge_i, strong_edge_j] = 255

    # mark weak edges as white if they are connected to strong edges
    for i, j in zip(weak_edge_i, weak_edge_j):
        if (edge_map[i - 1:i + 2, j - 1:j + 2] == 255).any():
            edge_map[i, j] = 255

    return edge_map