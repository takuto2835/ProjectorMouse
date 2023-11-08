import pyrealsense2 as rs
import numpy as np
import cv2
import math
from sklearn.cluster import DBSCAN
from pynput.mouse import Button, Controller
import time


#test

def initialize_realsense():
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        pipeline.start(config)
        return pipeline
    except Exception as e:
        print(f"Failed to initialize RealSense: {e}")
        exit(1)

def filter_vertices(vertices, y_min=-0.01, y_max=0.01, z_min=0.2, z_max=0.8, x_min = -1.0, x_max = 1.0):
    try:
        return np.where(
            (vertices[:, 1] > y_min) & (vertices[:, 1] < y_max) &  # 縦のフィルタ
            (vertices[:, 2] > z_min) & (vertices[:, 2] < z_max) & # 奥行きのフィルタ
            (vertices[:, 0] > x_min) & (vertices[:, 0] < x_max)
        )[0]
    except IndexError as e:
        print(f"Index error in filter_vertices: {e}")
        return np.array([])

def perform_clustering(filtered_vertices):
    try:
        return DBSCAN(eps=0.1, min_samples=20).fit(filtered_vertices)
    except Exception as e:
        print(f"Failed to perform clustering: {e}")
        return None

def closest_cluster(mean_x, mean_z, cluster_ids):
    try:
        min_distance = float('inf')
        closest_id = None
        for coords, cluster_id in cluster_ids.items():
            distance = np.sqrt((mean_x - coords[0]) ** 2 + (mean_z - coords[1]) ** 2)
            if distance < 0.1:
                min_distance = distance
                closest_id = cluster_id
        return closest_id
    except Exception as e:
        print(f"Error in closest_cluster: {e}")
        return None

def visualize_filtered_vertices(vis_image, filtered_vertices, scaling, offset_x, offset_y):
    for vertex in filtered_vertices:
        x_int = int(vertex[0] * scaling) + offset_x
        z_int = int(vertex[1] * scaling) + offset_y
        cv2.circle(vis_image, (x_int, z_int), 2, 255)

def visualize_cluster_ids(vis_image, new_cluster_ids, scaling, offset_x, offset_y):
    for (mean_x, mean_z), current_id in new_cluster_ids.items():
        mean_x_int = int(mean_x * scaling) + offset_x
        mean_z_int = int(mean_z * scaling) + offset_y
        cv2.circle(vis_image, (mean_x_int, mean_z_int), 10, 150, 2)
        cv2.putText(vis_image, str(current_id), (mean_x_int + 20, mean_z_int - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 150)

def update_cluster_ids(filtered_vertices, cluster_ids, next_id=0):
    clustering = perform_clustering(filtered_vertices)
    if clustering is None:
        return cluster_ids, next_id
    
    labels = clustering.labels_
    unique_labels = np.unique(labels)
    new_cluster_ids = {}
    
    for label in unique_labels:
        if label == -1:
            continue
        cluster_indices = np.where(labels == label)[0]
        cluster_points = filtered_vertices[cluster_indices]
        mean_x, mean_z = np.mean(cluster_points, axis=0)
        closest_id = closest_cluster(mean_x, mean_z, cluster_ids)
        current_id = closest_id if closest_id else next_id
        if closest_id is None:
            next_id += 1
        new_cluster_ids[(mean_x, mean_z)] = current_id
    
    return new_cluster_ids, next_id


def main():
    try:
        mouse = Controller()
    except Exception as e:
        print(f"Failed to initialize mouse controller: {e}")
        exit(1)

    pipeline = initialize_realsense()
    pc = rs.pointcloud()
    cluster_ids = {}
    next_id = 1
    scaling = 400
    mouse_state = 'up'

    mouse_scaling = 1800
    mouse_offsetX = 900
    mouse_offsetY = -200
    y_min = -0.05
    y_max = -0.04
    z_min = 0.2
    z_max = 0.7
    x_min = -1.0
    x_max = 1.0
    
    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            points = pc.calculate(depth_frame)
            vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
            vis_image = np.zeros((500, 500), dtype=np.uint8)
            fov_rad = math.radians(87)  # FOVをラジアンに変換
            max_z = 500  # zの最大値。この値は任意。

            # 線を描く起点と終点を計算
            start_point = (250, 0)
            cv2.line(vis_image,start_point, (int(math.tan(fov_rad / 2) * max_z) + 250, max_z), 150, 2)
            cv2.line(vis_image,start_point, (int(-math.tan(fov_rad / 2) * max_z) + 250, max_z), 150, 2)
            
            filtered_indices = filter_vertices(vertices, y_min, y_max, z_min, z_max, x_min, x_max)
            
            if filtered_indices.size > 0:

                filtered_vertices = vertices[filtered_indices][:, [0, 2]]
                visualize_filtered_vertices(vis_image, filtered_vertices, scaling, 250, 0)
                # 関数呼び出し
                cluster_ids, next_id = update_cluster_ids(filtered_vertices, cluster_ids, next_id)
                visualize_cluster_ids(vis_image, cluster_ids, scaling, 250, 0)

                if len(cluster_ids) > 0:

                    min_id = min(cluster_ids.values())
                    # 最小IDに対応する座標を取得して適用
                    mean_x, mean_z = [k for k, v in cluster_ids.items() if v == min_id][0]
                    screen_x = int(mean_x * mouse_scaling) + mouse_offsetX
                    screen_z = int(mean_z * mouse_scaling) + mouse_offsetY
                    mouse.position = (screen_x, screen_z)
                    if mouse_state == 'up':
                        mouse_state = 'down'
                        # Wait for the OS to catch up
                        time.sleep(0.1)
                        mouse.click(Button.left, 1)
            else:
                mouse_state = 'up'

            # Display the image
            cv2.imshow('Top View with Average', vis_image)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    except KeyboardInterrupt:
        print("User interrupted the program.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
