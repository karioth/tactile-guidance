import numpy as np
import cv2

if __name__ == "__main__":

    mock_depth_map_height = 480
    mock_depth_map_width = 640

    mock_depth_map = np.full((mock_depth_map_height, mock_depth_map_width), 1)

    print(mock_depth_map)

    cv2.imshow("ROI", mock_depth_map)

    cv2.waitKey(0)