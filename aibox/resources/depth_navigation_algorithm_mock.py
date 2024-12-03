import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue
import cv2

# Constants
DEPTH_THRESHOLD = 1500  # Depth value for free space
KERNEL_SIZE = 5  # Kernel size for obstacle checking
INF = float('inf')
ANGLE_RESOLUTION = 5  # Resolution of angles in degrees for movement

def add_safety_buffer(depth_map, obstacle_value, buffer_size):
    """Expand obstacles' areas by the buffer size."""
    obstacle_mask = depth_map < obstacle_value
    expanded_obstacle_mask = cv2.dilate(obstacle_mask.astype(np.uint8), np.ones((buffer_size, buffer_size), np.uint8))
    depth_map[expanded_obstacle_mask > 0] = obstacle_value

def is_path_clear(depth_map, bbox1, bbox2, kernel_size, safety_distance):
    """Check if there's a clear path between two bounding boxes with a safety buffer."""
    min_x = min(bbox1[0], bbox2[0])
    max_x = max(bbox1[2], bbox2[2])
    min_y = min(bbox1[1], bbox2[1])
    max_y = max(bbox1[3], bbox2[3])

    path_points = []
    
    for y in range(min_y, max_y):
        for x in range(min_x, max_x):
            kernel = depth_map[max(0, y - kernel_size // 2):min(depth_map.shape[0], y + kernel_size // 2 + 1),
                               max(0, x - kernel_size // 2):min(depth_map.shape[1], x + kernel_size // 2 + 1)]
            if np.all(kernel >= DEPTH_THRESHOLD):
                if (depth_map[y, x] >= DEPTH_THRESHOLD) and (np.mean(depth_map[max(0, y - safety_distance):min(depth_map.shape[0], y + safety_distance + 1),
                                                                               max(0, x - safety_distance):min(depth_map.shape[1], x + safety_distance + 1)] >= DEPTH_THRESHOLD)):
                    path_points.append((x, y))

    return path_points

def astar(start, goal, depth_map, depth_threshold):
    """A* pathfinding algorithm for continuous angles."""
    queue = PriorityQueue()
    queue.put((0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    all_directions = [(np.cos(np.radians(angle)), np.sin(np.radians(angle))) for angle in range(0, 360, ANGLE_RESOLUTION)]

    while not queue.empty():
        current = queue.get()[1]

        if current == goal:
            break

        for direction in all_directions:
            neighbor = (int(current[0] + direction[0]), int(current[1] + direction[1]))

            if (0 <= neighbor[0] < depth_map.shape[1] and 
                0 <= neighbor[1] < depth_map.shape[0] and 
                depth_map[neighbor[1], neighbor[0]] >= depth_threshold):

                new_cost = cost_so_far[current] + np.linalg.norm(np.array(direction))

                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + np.linalg.norm(np.array(goal) - np.array(neighbor))
                    queue.put((priority, neighbor))
                    came_from[neighbor] = current

    # Reconstruct path
    current = goal
    path = []
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    
    return path

def smooth_path(path, step_size):
    """Smooth the path by creating waypoints based on the desired step size."""
    smoothed_path = []

    start = np.array(path[0])
    
    for i in range(len(path) - 1):
        #start = np.array(path[i])
        #end = np.array(path[i + 1])
        #end = np.array(path[i + step_size + 1])
        end = np.array(path[i+1])
        direction = end - start
        distance = np.linalg.norm(direction)
        direction_normalized = direction / distance if distance > 0 else direction
        
        #smoothed_path.append(tuple(start.astype(int)))

        #print(end)
        #print(start)
        #print(distance)

        if distance > step_size:
            smoothed_path.append(tuple(start.astype(int)))
            smoothed_path.append(tuple(end.astype(int)))
            return smoothed_path
            start = np.array(path[i+1])

# Add points towards the end point until we reach it
        #while np.linalg.norm(start - end) > step_size:
        #    smoothed_path.append(tuple(start.astype(int)))
        #    start += direction_normalized * step_size
            
    smoothed_path.append(tuple(end.astype(int)))  # Ensure the last point is included
    return smoothed_path

def compute_directions(path):
    """Compute angles of movement at each step in the smoothed path."""
    directions = []
    for i in range(1, len(path)):
        start = np.array(path[i - 1])
        end = np.array(path[i])
        direction = end - start
        angle = np.degrees(np.arctan2(direction[1], direction[0]))  # Convert to degrees
        directions.append(angle)
    
    return directions

def visualize_results(depth_map, path, smoothed_path, bbox1, bbox2, angles):
    """Visualize the depth map and the path with angles."""
    annotated_depth = np.clip(depth_map / 2550, 0, 1)

    plt.figure(figsize=(12, 8))
    
    # Show the depth map
    plt.subplot(1, 2, 1)
    plt.title('Depth Map Area')
    plt.imshow(annotated_depth, cmap='gray')
    plt.colorbar(label='Depth Value')

    # Show the smoothed path with angles
    plt.subplot(1, 2, 2)
    plt.title('Smoothed Path with Angles')
    plt.imshow(annotated_depth, cmap='gray')
    plt.plot([p[0] for p in smoothed_path], [p[1] for p in smoothed_path], color='yellow', linewidth=2, marker='o')
    plt.scatter([bbox1[0], bbox2[0]], [bbox1[1], bbox2[1]], color='red', label='Object 1')
    plt.scatter([bbox1[2], bbox2[2]], [bbox1[3], bbox2[3]], color='blue', label='Object 2')

    # Annotate angles along the smoothed path
    for i, angle in enumerate(angles):
        if i < len(smoothed_path) - 1:
            plt.annotate(f'{angle:.1f}°', (smoothed_path[i][0], smoothed_path[i][1]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='orange')

    plt.colorbar(label='Depth Value')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Create a depth map with obstacles
depth_map_height = 600
depth_map_width = 800

# Initialize depth map with high depth
depth_map = np.full((depth_map_height, depth_map_width), 2000)

# Add obstacles (low depth)
obstacle_value = 500
cv2.rectangle(depth_map, (300, 300), (400, 400), obstacle_value, -1)  # Obstacle 1
cv2.rectangle(depth_map, (200, 450), (350, 500), obstacle_value, -1)  # Obstacle 2
cv2.rectangle(depth_map, (580, 200), (730, 400), obstacle_value, -1)  # Obstacle 3

# Add safety buffer
safety_buffer_size = 10
add_safety_buffer(depth_map, obstacle_value, safety_buffer_size)

# Define bounding boxes for two objects
bbox1 = (150, 150, 200, 200)  # Object 1 bounding box
bbox2 = (650, 450, 700, 500)  # Object 2 bounding box

# Analyze the area to find clear path points
clear_path_points = is_path_clear(depth_map, bbox1, bbox2, KERNEL_SIZE, safety_buffer_size)

# Run pathfinding between the centers of the bounding boxes
if clear_path_points:
    start = ((bbox1[0] + bbox1[2]) // 2, (bbox1[1] + bbox1[3]) // 2)
    goal = ((bbox2[0] + bbox2[2]) // 2, (bbox2[1] + bbox2[3]) // 2)
    
    path = astar(start, goal, depth_map, DEPTH_THRESHOLD)
    
    # Smooth the path for continuous movement
    smoothed_path = smooth_path(path, step_size=5)  # Example step size for smoother path
    
    # Compute directions (angles) for the smoothed path
    angles = compute_directions(smoothed_path)

    print("Original path:", path)
    print("Smoothed path:", smoothed_path)
    print("Angles for each step:", angles)
    visualize_results(depth_map, path, smoothed_path, bbox1, bbox2, angles)
else:
    print("No clear path found between objects.")