import cv2
import numpy as np
from queue import PriorityQueue

def map_obstacles(handBB, targetBB, depth_map):

    bbs_dilation = 5

    # Get BB information
    hand_x, hand_y, hand_w, hand_h = handBB[:4]
    target_x, target_y, target_w, target_h = targetBB[:4]

    # Calculate BB bounds to check for overlap
    hand_right = int(hand_x + hand_w/2)
    hand_left = int(hand_x - hand_w/2)
    hand_top = int(hand_y - hand_h/2)
    hand_bottom = int(hand_y + hand_h/2)

    target_right = int(target_x + target_w/2)
    target_left = int(target_x - target_w/2)
    target_top = int(target_y - target_h/2)
    target_bottom = int(target_y + target_h/2)

    print(f'hand BB: bot {hand_bottom}, top {hand_top}, left {hand_left}, right {hand_right}')

    # Object closer to camera relative to hand -> values are larger
    hand_depth = handBB[7]
    #obstacle_mask = depth_map < hand_depth # binary mask
    obstacle_mask = depth_map < hand_depth - 300
    print(f'depth_map {depth_map.shape}, {depth_map.min()} - {depth_map.max()}')
    print(f'hand_depth {hand_depth.shape}, {hand_depth.min()} - {hand_depth.max()}')
    print(f'obstacle_mask {obstacle_mask.shape}, {obstacle_mask.min()} - {obstacle_mask.max()}')

    # Mask hand BB
    obstacle_mask[hand_top-bbs_dilation:hand_bottom+bbs_dilation, hand_left-bbs_dilation:hand_right+bbs_dilation] = True

    # Mask target BB
    obstacle_mask[target_top-bbs_dilation:target_bottom+bbs_dilation, target_left-bbs_dilation:target_right+bbs_dilation] = True

    # Dilate obstacles
    expanded_obstacle_mask = cv2.dilate(obstacle_mask.astype(np.uint8), np.ones((5, 5), np.uint8))
    #depth_map[expanded_obstacle_mask > 0] = hand_depth - 5

    mask_rgb = cv2.cvtColor(obstacle_mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    cv2.imshow("expanded_obstacle_mask", mask_rgb)
    pressed_key = cv2.waitKey(1)

    return expanded_obstacle_mask

def check_obstacles_between_points(handBB, targetBB, depth_map, depth_threshold):
    """
    Check if there are any obstacles between two points in a depth map.

    Parameters:
    start (tuple): Starting point (x1, y1).
    end (tuple): Ending point (x2, y2).
    depth_map (np.array): 2D numpy array representing the depth map.
    depth_threshold (float): Threshold below which the pixel is considered an obstacle.

    Returns:
    bool: True if an obstacle is found, False otherwise.
    """
    
    # Get BB information
    hand_x, hand_y = handBB[:2]
    target_x, target_y = targetBB[:2]

    # Compute the difference in x and y
    dx = hand_x - target_x
    dy = hand_y - target_y
    
    # Check for line direction
    steps = int(max(abs(dx), abs(dy)))
    
    # Prevent division by zero and determine step changes
    if steps == 0:
        return depth_map[int(hand_y), int(hand_x)] < depth_threshold

    x_increment = dx / steps
    y_increment = dy / steps

    x, y = hand_x, hand_y

    for _ in range(steps + 1):
        xi, yi = int(round(x)), int(round(y))
        
        # Check bounds
        if 0 <= xi < depth_map.shape[1] and 0 <= yi < depth_map.shape[0]:
            if depth_map[yi, xi] < depth_threshold:
                return True  # Found an obstacle
        
        x += x_increment
        y += y_increment
    
    return False  # No obstacles found


def astar(handBB, targetBB, depth_map, depth_threshold, stop_condition):
    """A* pathfinding algorithm for continuous angles, stopping after finding x steps in the optimal trajectory."""

    ANGLE_RESOLUTION = 45

    xc_hand, yc_hand = handBB[:2]
    xc_target, yc_target = targetBB[:2]

    start = (int(xc_hand), int(yc_hand))
    goal = (int(xc_target), int(yc_target))

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
    
    if len(path) > stop_condition:
        path = path[:stop_condition]

    return path

def dijkstra(handBB, targetBB, depth_map, depth_threshold, stop_condition):
    """Dijkstra's pathfinding algorithm for continuous angles."""

    ANGLE_RESOLUTION = 5

    xc_hand, yc_hand = handBB[:2]
    xc_target, yc_target = targetBB[:2]

    start = (int(xc_hand), int(yc_hand))
    goal = (int(xc_target), int(yc_target))

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

            # Check for valid neighbor
            if (0 <= neighbor[0] < depth_map.shape[1] and 
                0 <= neighbor[1] < depth_map.shape[0] and 
                depth_map[neighbor[1], neighbor[0]] >= depth_threshold):

                new_cost = cost_so_far[current] + np.linalg.norm(np.array(direction))

                # Only consider this new path if it's better
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost  # Dijkstra's uses path cost as priority
                    queue.put((priority, neighbor))
                    came_from[neighbor] = current

    # Reconstruct path
    current = goal
    path = []
    while current is not None:
        path.append(current)
        current = came_from[current]

    path.reverse()  # Reverse the path to get it from start to goal

    return path[:stop_condition]

def fast_pathfinding(handBB, targetBB, depth_map, depth_threshold, x):
    """Pathfinding algorithm that stops after finding the first x optimal steps."""

    ANGLE_RESOLUTION = 5

    xc_hand, yc_hand = handBB[:2]
    xc_target, yc_target = targetBB[:2]

    start = (int(xc_hand), int(yc_hand))
    goal = (int(xc_target), int(yc_target))

    queue = PriorityQueue()
    queue.put((0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    all_directions = [(np.cos(np.radians(angle)), np.sin(np.radians(angle))) for angle in range(0, 360, ANGLE_RESOLUTION)]
    
    steps_found = 0  # Counter to track found optimal steps

    while not queue.empty():
        current = queue.get()[1]

        # If we reach the goal, we can start to count path steps
        if current == goal:
            steps_found += 1
            current_node = current
            # Reconstruct path immediately
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = came_from[current_node]
            path.reverse()  # Reverse to get start to goal

            if steps_found >= x:
                return path[:x]  # Return first x steps

        for direction in all_directions:
            neighbor = (int(current[0] + direction[0]), int(current[1] + direction[1]))

            # Check for valid neighbor
            if (0 <= neighbor[0] < depth_map.shape[1] and 
                0 <= neighbor[1] < depth_map.shape[0] and 
                depth_map[neighbor[1], neighbor[0]] >= depth_threshold):

                new_cost = cost_so_far[current] + np.linalg.norm(np.array(direction))

                # Only consider this new path if it's better
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost  # For Dijkstra's, use cumulative cost as priority
                    queue.put((priority, neighbor))
                    came_from[neighbor] = current

    # If we reach the end of search without finding the goal
    return []  # Return an empty path if goal is not reached


def smooth_path(path, step_size):
    """Smooth the path by creating waypoints based on the desired step size."""
    smoothed_path = []

    start = np.array(path[0])
    
    for i in range(len(path) - 1):
        end = np.array(path[i+1])
        direction = end - start
        distance = np.linalg.norm(direction)
        direction_normalized = direction / distance if distance > 0 else direction

        if distance > step_size:
            smoothed_path.append(tuple(start.astype(int)))
            smoothed_path.append(tuple(end.astype(int)))

            return smoothed_path
            #smoothed_path.append(tuple(start.astype(int)))
            #start = np.array(path[i+1])
            
    smoothed_path.append(tuple(end.astype(int)))  # Ensure the last point is included
    return smoothed_path

def find_obstacle_target_point(handBB, targetBB, obstacle_map):

    xc_hand, yc_hand = handBB[:2]
    xc_target, yc_target = targetBB[:2]

    # Create region of interest from obstacle map
    
    print(obstacle_map)
    print(obstacle_map.shape)
    print(type(obstacle_map))

    roi_general = obstacle_map[:,int(min(xc_hand, xc_target)):int(max(xc_hand, xc_target))]

    print(obstacle_map.shape)

    #roi_rgb = cv2.cvtColor(roi_general.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    #cv2.imshow("ROI", roi_rgb)
    #pressed_key = cv2.waitKey(1)
    
    # Determine general direction of movement
    
    angle_radians = np.arctan2(yc_hand - yc_target, xc_target - xc_hand) # inverted y-axis
    angle = np.degrees(angle_radians) % 360

    if 90 < angle < 270:
        direction = 'left'
    else:
        direction = 'right'

    # Find closest obstacle point in x axis which is at least as high as hand center

    roi_target_point = obstacle_map[:int(yc_hand),int(min(xc_hand, xc_target)):int(max(xc_hand, xc_target))]

    min_values = np.min(roi_target_point, axis=1)
    min_indices = np.argmin(roi_target_point, axis=1)

    min_indices = []
    max_indices = []

    for i in range(roi_target_point.shape[0]):  # Iterate over rows
        min_value = min_values[i]
        # Get the indices of the maximum value in the current row
        indices_for_min_value = np.where(roi_target_point[i, :] == min_value)[0]
        
        # Find the minimum index for the current maximum value
        min_index = indices_for_min_value.min()
        max_index = indices_for_min_value.max()
        min_indices.append(min_index)
        max_indices.append(max_index)

    if direction == 'left': # moving from right side of the image
        obstacle_x = max(max_indices)
    else:
        obstacle_x = min(min_indices)

    print(f'Obstacle x within ROI: {obstacle_x}, within whole map {obstacle_x + int(min(xc_hand, xc_target))}')

    # Repeat procedure for finding point in y axis for the same ROI

    min_values = np.min(roi_target_point, axis=0)
    min_indices = np.argmin(roi_target_point, axis=0)

    min_indices = []
    max_indices = []

    for i in range(roi_target_point.shape[1]):  # Iterate over rows
        min_value = min_values[i]
        # Get the indices of the maximum value in the current row
        indices_for_min_value = np.where(roi_target_point[:, i] == min_value)[0]
        
        # Find the minimum index for the current maximum value
        min_index = indices_for_min_value.min()
        max_index = indices_for_min_value.max()
        min_indices.append(min_index)
        max_indices.append(max_index)

    obstacle_y = min(min_indices)

    print(f'Obstacle y within ROI: {obstacle_y}, within whole map {obstacle_y}')

    roi_rgb = cv2.cvtColor(roi_target_point.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    cv2.circle(roi_rgb, (obstacle_x, obstacle_y), radius=5, color=(0, 0, 255), thickness=-1)
    cv2.imshow("ROI", roi_rgb)
    pressed_key = cv2.waitKey(1)

    return [obstacle_x + int(min(xc_hand, xc_target)), obstacle_y]