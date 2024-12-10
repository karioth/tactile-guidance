import cv2
import numpy as np
from queue import PriorityQueue

def map_obstacles(handBB, targetBB, depth_map, metric):

    bbs_dilation = 10
    obstacles_dilation = 3

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

    # Object closer to camera relative to hand -> values are larger
    hand_depth = handBB[7]

    # relative hand depth is actually disparity, i.e. 1/estimate
    scale = 5000
    hand_depth = scale / hand_depth if not metric else hand_depth
    depth_map = scale / depth_map if not metric else depth_map

    # Create binary obstacle mask
    obstacle_mask = depth_map < hand_depth

    # Mask hand BB
    #obstacle_mask[hand_top-bbs_dilation:hand_bottom+bbs_dilation, hand_left-bbs_dilation:hand_right+bbs_dilation] = True
    obstacle_mask[hand_top-bbs_dilation:hand_bottom+bbs_dilation, hand_left-bbs_dilation:hand_right+bbs_dilation] = False

    # Mask target BB
    #obstacle_mask[target_top-bbs_dilation:target_bottom+bbs_dilation, target_left-bbs_dilation:target_right+bbs_dilation] = True
    obstacle_mask[target_top-bbs_dilation:target_bottom+bbs_dilation, target_left-bbs_dilation:target_right+bbs_dilation] = False

    # Dilate obstacles
    expanded_obstacle_mask = cv2.dilate(obstacle_mask.astype(np.uint8), np.ones((obstacles_dilation, obstacles_dilation), np.uint8))
    #depth_map[expanded_obstacle_mask > 0] = hand_depth - 5

    mask_rgb = cv2.cvtColor(expanded_obstacle_mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    cv2.imshow("expanded_obstacle_mask", mask_rgb)
    cv2.setWindowProperty("expanded_obstacle_mask", cv2.WND_PROP_TOPMOST, 1)
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
    #hand_x, hand_y = handBB[:2]
    #target_x, target_y = targetBB[:2]
    xc_hand, yc_hand = handBB[:2]
    xc_target, yc_target = targetBB[:2]

    # If hand and target are aligned either horizontally or vertically navigate normally - TO EXPAND
    if xc_hand == xc_target or yc_hand == yc_target:
        return False

    roi_depth_map = depth_map[int(min(yc_hand, yc_target)):int(max(yc_hand, yc_target)),int(min(xc_hand, xc_target)):int(max(xc_hand, xc_target))]

    try:
        mask_rgb = cv2.cvtColor(roi_depth_map.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
        cv2.imshow("roi_depth_map", mask_rgb)
        cv2.setWindowProperty("roi_depth_map", cv2.WND_PROP_TOPMOST, 1)
        pressed_key = cv2.waitKey(1)
    except:
        pass

    # Check if there is any value above the threshold
    if np.any(roi_depth_map >= depth_threshold):
        return True
    else:
        return False


def find_obstacle_target_point(handBB, targetBB, obstacle_map, leeway=10):

    xc_hand, yc_hand = handBB[:2]
    xc_target, yc_target = targetBB[:2]

    # Create region of interest from obstacle map
    
    
    # Determine general direction of movement
    angle_radians = np.arctan2(yc_hand - yc_target, xc_target - xc_hand) # inverted y-axis
    angle = np.degrees(angle_radians) % 360
    #print(f'Angle: {angle}')

    if 90 < angle < 270:
        direction = 'left'
    else:
        direction = 'right'

    # Find closest obstacle point in x axis which is at least as high as hand center
    #roi_target_point = obstacle_map[:int(yc_hand),int(min(xc_hand, xc_target)):int(max(xc_hand, xc_target))]
    roi_target_point = obstacle_map[int(min(yc_hand, yc_target)):int(max(yc_hand, yc_target)),int(min(xc_hand, xc_target)):int(max(xc_hand, xc_target))]

    roi_min_y = np.min(np.argwhere(roi_target_point)[:, 0])
    print(roi_min_y)

    #if roi_min_y <= 5:
    #    return targetBB[:2], roi_min_y

    # Find corners of obstacles
    dst = cv2.cornerHarris(roi_target_point, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    
    # Find the x-coordinates of the corners that are above a threshold
    corner_indices = np.argwhere(dst > 0.01 * dst.max())
    
    #if corner_indices.size == 0:
    #    return [None, None]  # Return None if no corners are found
    if corner_indices.size == 0:
        return targetBB[:2], roi_min_y

    min_y = np.min(corner_indices[:, 0]) + leeway
    min_x = np.min(corner_indices[:, 1]) + leeway
    max_x = np.max(corner_indices[:, 1]) + leeway

    #min_x = corner_indices[corner_indices[:, 0] == min_y, 1].min()
    #max_x = corner_indices[corner_indices[:, 0] == min_y, 1].max()

    # Determine target point based on direction
    target_point = [max_x, min_y] if direction == 'left' else [min_x, min_y]
    roi_rgb = cv2.cvtColor(roi_target_point.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)

    for candidate in corner_indices:
        cv2.circle(roi_rgb, (candidate[1], candidate[0]), radius=1, color=(0, 255, 0), thickness=-1)
    cv2.circle(roi_rgb, (target_point[0], target_point[1]), radius=5, color=(0, 0, 255), thickness=-1)
    cv2.imshow("ROI", roi_rgb)
    #cv2.setWindowProperty("ROI", cv2.WND_PROP_TOPMOST, 1)
    pressed_key = cv2.waitKey(1)

    angle_radians = np.arctan2(yc_hand - target_point[1], target_point[0] - xc_hand) # inverted y-axis
    angle = np.degrees(angle_radians) % 360

    return target_point, roi_min_y


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
