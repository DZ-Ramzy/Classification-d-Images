import cv2
import numpy as np
from sklearn.cluster import KMeans

def calculate_local_distances(image, window_size=5):
    height, width = image.shape[:2]
    distances = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            i_start = max(0, i - window_size)
            i_end = min(height, i + window_size + 1)
            j_start = max(0, j - window_size)
            j_end = min(width, j + window_size + 1)
            window = image[i_start:i_end, j_start:j_end]
            center_value = image[i, j]
            distances[i, j] = np.mean(np.abs(window - center_value))
    return distances

def apply_kmeans(image, n_clusters=2):
    pixels = image.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(pixels)
    clustered = labels.reshape(image.shape)
    return clustered, kmeans

def remove_background(image, clustered, background_label=0):
    mask = (clustered != background_label).astype(np.uint8) * 255
    return cv2.bitwise_and(image, image, mask=mask)

def detect_edges(image, low_threshold=50, high_threshold=150):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    return cv2.Canny(blurred, low_threshold, high_threshold)

def merge_horizontal_lines(edges, min_line_length=100, max_line_gap=10):
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                            minLineLength=min_line_length, maxLineGap=max_line_gap)
    if lines is None:
        return edges
    y_coords = []
    for line in lines:
        _, y1, _, y2 = line[0]
        y_coords.append((y1 + y2) / 2)
    y_coords = np.array(y_coords).reshape(-1, 1)
    kmeans = KMeans(n_clusters=min(len(y_coords), 20), random_state=42, n_init='auto')
    kmeans.fit(y_coords)
    merged = np.zeros_like(edges)
    for center in kmeans.cluster_centers_:
        y = int(center[0])
        cv2.line(merged, (0, y), (edges.shape[1], y), 255, 1)
    return merged

def filter_small_lines(edges, min_length=50):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = np.zeros_like(edges)
    for contour in contours:
        if cv2.arcLength(contour, False) > min_length:
            cv2.drawContours(filtered, [contour], -1, 255, 1)
    return filtered

def detect_stair_steps(edge_image, original_image, rho=1, theta=np.pi/180, threshold=50):
    lines = cv2.HoughLines(edge_image, rho, theta, threshold)
    if lines is None:
        return 0, original_image.copy()

    y_coords = []
    for line in lines:
        rho_val, theta_val = line[0]
        b = np.sin(theta_val)
        y0 = b * rho_val
        y_coords.append(y0)

    y_coords = np.array(y_coords).reshape(-1, 1)
    kmeans = KMeans(n_clusters=min(len(y_coords), 20), random_state=42, n_init='auto')
    kmeans.fit(y_coords)

    result = original_image.copy()

    # Nouveau filtrage : supprimer les lignes trop proches entre elles
    y_sorted = sorted([int(center[0]) for center in kmeans.cluster_centers_])
    final_lines = []
    min_spacing = 10  # espacement minimum en pixels

    for y in y_sorted:
        if not final_lines or abs(y - final_lines[-1]) >= min_spacing:
            final_lines.append(y)

    # Dessiner les lignes détectées
    for y in final_lines:
        cv2.line(result, (0, y), (result.shape[1], y), (0, 0, 255), 2)

    return len(final_lines), result

def resize_for_display(img, max_width=1000):
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        return cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read image")

    original_color = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clustered, _ = apply_kmeans(gray)
    no_background = remove_background(gray, clustered)
    edges = detect_edges(no_background)
    merged = merge_horizontal_lines(edges)
    filtered = filter_small_lines(merged)
    num_steps, result = detect_stair_steps(filtered, original_image=original_color)
    return result, num_steps

if __name__ == "__main__":
    img_path = r"C:\Users\alger\OneDrive\Desktop\projet_image\images\47.jpg"
    try:
        result, num_steps = process_image(img_path)
        print(f"Number of detected stair steps: {num_steps}")
        display = resize_for_display(result)
        cv2.imshow("Detected Stairs", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error processing image: {str(e)}")
