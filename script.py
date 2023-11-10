import pyrealsense2 as rs
import numpy as np
import cv2
import math
from sklearn.cluster import DBSCAN
from pynput.mouse import Button, Controller
import time

# Constants for RealSense stream configuration
DEPTH_WIDTH = 640
DEPTH_HEIGHT = 480
DEPTH_FPS = 30

# Constants for vertex filtering
Y_MIN = -0.01
Y_MAX = 0.01
Z_MIN = 0.2
Z_MAX = 0.8
X_MIN = -1.0
X_MAX = 1.0

# Constants for clustering
EPS = 0.1
MIN_SAMPLES = 200

# Constants for mouse control
MOUSE_SCALING = 1800
MOUSE_OFFSET_X = 900
MOUSE_OFFSET_Y = -200

# Constants for FOV and visualization
FOV_DEGREES = 87
VISUALIZATION_WIDTH = 500
VISUALIZATION_HEIGHT = 500
LINE_COLOR = 150  # Grayscale intensity for drawing lines
LINE_THICKNESS = 2

# Constants for the FOV lines
START_POINT_X = VISUALIZATION_WIDTH // 2
START_POINT_Y = 0
MAX_Z_FOR_FOV_LINES = VISUALIZATION_HEIGHT

# Constants for visualization
SCALING = 400
OFFSET_X = START_POINT_X
OFFSET_Y = 0

MASK_RADIUS = 2


def create_mask_from_vertices(vertices, radius):
    """
    Create a binary mask image from the filtered vertices.
    A circle is drawn for each vertex with the given radius, and the area covered by the circles is considered foreground.
    """
    # Initialize an empty image
    mask_image = np.zeros((VISUALIZATION_HEIGHT, VISUALIZATION_WIDTH), dtype=np.uint8)

    # Convert vertex positions to image coordinates and draw circles
    for vertex in vertices:
        x_int = int(vertex[0] * SCALING) + OFFSET_X
        z_int = int(vertex[1] * SCALING) + OFFSET_Y
        cv2.circle(mask_image, (x_int, z_int), radius, 255, -1)  # -1 fills the circle

    # Convert the mask image to a boolean array
    mask = mask_image.astype(bool)

    return mask

def apply_mask_to_vertices(vertices, mask):
    """
    Applies a binary mask to the vertices. Only vertices within the mask's foreground are kept.
    """
    try:
        masked_vertices = []
        height, width = mask.shape  # maskの高さと幅を取得
        for vertex in vertices:
            x_int = int(vertex[0] * SCALING) + OFFSET_X
            z_int = int(vertex[1] * SCALING) + OFFSET_Y
            # 範囲チェックを追加
            if 0 <= z_int < height and 0 <= x_int < width:
                if not mask[z_int, x_int]:  # Check if the vertex is within the mask's foreground
                    masked_vertices.append(vertex)
        return np.array(masked_vertices)
    except Exception as e:
        print(f"Error in apply_mask_to_vertices: {e}")
        return None


# Initialize the RealSense camera
def initialize_realsense():
    """
    Initialize RealSense pipeline and configure the depth stream.
    """
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.z16, DEPTH_FPS)
        pipeline.start(config)
        return pipeline
    except Exception as e:
        print(f"Failed to initialize RealSense: {e}")
        exit(1)

# Filter vertices based on pre-defined limits
def filter_vertices(vertices):
    """
    Filter vertices based on pre-defined X, Y, Z limits.
    """
    try:
        return np.where(
            (vertices[:, 1] > Y_MIN) & (vertices[:, 1] < Y_MAX) &
            (vertices[:, 2] > Z_MIN) & (vertices[:, 2] < Z_MAX) &
            (vertices[:, 0] > X_MIN) & (vertices[:, 0] < X_MAX)
        )[0]
    except IndexError as e:
        print(f"Index error in filter_vertices: {e}")
        return np.array([])



# Perform clustering on filtered vertices
def perform_clustering(filtered_vertices):
    """
    Perform DBSCAN clustering on the filtered vertices.
    """
    try:
        return DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(filtered_vertices)
    except Exception as e:
        print(f"Failed to perform clustering: {e}")
        return None

# Find the closest cluster
def closest_cluster(mean_x, mean_z, cluster_ids):
    """
    Find the closest cluster based on Euclidean distance.
    """
    try:
        min_distance = float('inf')
        closest_id = None
        for coords, cluster_id in cluster_ids.items():
            distance = math.hypot(mean_x - coords[0], mean_z - coords[1])
            if distance < min_distance:
                min_distance = distance
                closest_id = cluster_id
        return closest_id
    except Exception as e:
        print(f"Error in closest_cluster: {e}")
        return None

# Visualize filtered vertices on the image
def visualize_filtered_vertices(vis_image, filtered_vertices):
    """
    Visualize the filtered vertices on the image.
    """
    for vertex in filtered_vertices:
        x_int = int(vertex[0] * SCALING) + OFFSET_X
        z_int = int(vertex[1] * SCALING) + OFFSET_Y
        cv2.circle(vis_image, (x_int, z_int), 2, 255)

# Visualize cluster IDs on the image
def visualize_cluster_ids(vis_image, new_cluster_ids):
    """
    Visualize the cluster IDs on the image.
    """
    for (mean_x, mean_z), current_id in new_cluster_ids.items():
        mean_x_int = int(mean_x * SCALING) + OFFSET_X
        mean_z_int = int(mean_z * SCALING) + OFFSET_Y
        cv2.circle(vis_image, (mean_x_int, mean_z_int), 10,(255, 255, 255), 2)
        cv2.putText(vis_image, str(current_id), (mean_x_int + 20, mean_z_int - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))


def overlay_mask_on_image(vis_image, mask, mask_color=(0, 255, 0), alpha=0.3):
    """
    Overlays a binary mask on top of the visualization image.

    :param vis_image: The original image on which to overlay the mask.
    :param mask: The binary mask to overlay.
    :param mask_color: The color to use for the mask overlay (default is green).
    :param alpha: The transparency factor for the mask overlay.
    """
    # Ensure vis_image is in color
    if len(vis_image.shape) == 2:  # Check if grayscale
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)

    # Create a color image from the mask to overlay
    color_mask = np.zeros_like(vis_image, dtype=np.uint8)

    # We need to make sure that the mask is broadcastable to the color mask
    # This means we need to index where mask is True, and for these indices, set the color
    color_mask[mask] = mask_color
    # Blend the color mask with the image
    vis_image = cv2.addWeighted(color_mask, alpha, vis_image, 1 - alpha, 1)
    return vis_image


# Update cluster IDs based on the current frame
def update_cluster_ids(filtered_vertices, cluster_ids, next_id=0):
    """
    Update cluster IDs based on the current frame.
    """
    clustering = perform_clustering(filtered_vertices)
    if clustering is None:
        return cluster_ids, next_id
    
    labels = clustering.labels_
    unique_labels = set(labels)
    new_cluster_ids = {}
    
    for label in unique_labels:
        if label == -1:  # Skip noise points
            continue
        cluster_indices = np.where(labels == label)[0]
        cluster_points = filtered_vertices[cluster_indices]
        mean_x, mean_z = np.mean(cluster_points, axis=0)
        closest_id = closest_cluster(mean_x, mean_z, cluster_ids)
        current_id = closest_id if closest_id is not None else next_id
        if closest_id is None:
            next_id += 1
        new_cluster_ids[(mean_x, mean_z)] = current_id
    
    return new_cluster_ids, next_id

# Main function to run the application
def main():
    """
    Main function to run the application.
    """
    try:
        mouse = Controller()
    except Exception as e:
        print(f"Failed to initialize mouse controller: {e}")
        exit(1)

    pipeline = initialize_realsense()
    pc = rs.pointcloud()
    cluster_ids = {}
    next_id = 1
    mouse_state = 'up'

    initial_mask = None

    try:   
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue  # Skip if the frame is not ready
            
            points = pc.calculate(depth_frame)
            vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)          

            # Convert FOV from degrees to radians for calculation
            fov_rad = math.radians(FOV_DEGREES)
            
            # Calculate end points for the FOV lines
            end_point_x_right = int(math.tan(fov_rad / 2) * MAX_Z_FOR_FOV_LINES) + START_POINT_X
            end_point_x_left = int(-math.tan(fov_rad / 2) * MAX_Z_FOR_FOV_LINES) + START_POINT_X

            # Drawing the FOV lines on the visualization image
            start_point = (START_POINT_X, START_POINT_Y)
            vis_image = np.zeros((VISUALIZATION_HEIGHT, VISUALIZATION_WIDTH), dtype=np.uint8)
            cv2.line(vis_image, start_point, (end_point_x_right, MAX_Z_FOR_FOV_LINES), LINE_COLOR, LINE_THICKNESS)
            cv2.line(vis_image, start_point, (end_point_x_left, MAX_Z_FOR_FOV_LINES), LINE_COLOR, LINE_THICKNESS)
            
            # Filter vertices and update clusters
            filtered_indices = filter_vertices(vertices)
            filtered_vertices = vertices[filtered_indices][:, [0, 2]]               

            
            # マスクが存在しない場合にのみ生成
            if initial_mask is None:
                filtered_vertices = vertices[filtered_indices][:, [0, 2]]
                initial_mask = create_mask_from_vertices(filtered_vertices, 10)

                
            masked_vertices = apply_mask_to_vertices(filtered_vertices, initial_mask)

            visualize_filtered_vertices(vis_image, filtered_vertices)
            
            # Usage example:
            # Assuming 'vis_image' is the image you want to overlay the mask on,
            # and 'mask' is the binary mask array with the same size as 'vis_image'.
            if initial_mask is not None:                    
                vis_image = overlay_mask_on_image(vis_image, initial_mask)

            if masked_vertices.size > 0:               

                cluster_ids, next_id = update_cluster_ids(masked_vertices, cluster_ids, next_id)
                visualize_cluster_ids(vis_image, cluster_ids)
                # Mouse control logic
                if cluster_ids:
                    min_id = min(cluster_ids.values())
                    mean_x, mean_z = next(k for k, v in cluster_ids.items() if v == min_id)
                    screen_x = int(mean_x * MOUSE_SCALING) + MOUSE_OFFSET_X
                    screen_z = int(mean_z * MOUSE_SCALING) + MOUSE_OFFSET_Y
                    mouse.position = (screen_x, screen_z)
                    if mouse_state == 'up':
                        mouse_state = 'down'
                        time.sleep(0.1)  # Wait for the OS to catch up
                        mouse.click(Button.left, 1)
            else:
                mouse_state = 'up'

            # Display the visualization
            cv2.imshow('Top View with Average', vis_image)

            # キーボード入力を待つ
            k = cv2.waitKey(1)

            # 'q'キーまたは'esc'キーが押されたらループを抜ける
            if k == ord('q') or k == 27:
                print("User pushed the end key")
                break

            # ウィンドウが閉じられたかチェック
            if cv2.getWindowProperty('Top View with Average', cv2.WND_PROP_VISIBLE) < 1:
                print("User pushed the close button")
                break

    except KeyboardInterrupt:
        print("User interrupted the program.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Close all windows")

if __name__ == "__main__":
    main()
