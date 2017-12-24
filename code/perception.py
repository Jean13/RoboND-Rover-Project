import numpy as np
import cv2

'''
Identify pixels above the threshold
Threshold of RGB > 160 does a nice job of identifying ground pixels only
'''
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])

    '''
    Require that each pixel be above all three threshold values in RGB
    above_thresh will now contain a boolean array with "True"
    where threshold was met
    '''
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select


'''
Identifies the obstacles
Identify pixels below the threshold
Threshold of RGB < 160 identifies obstacles
'''
def obstacles_thresh(img, rgb_thresh=(150, 150, 150)):
    # Create an array of zeros same xy size as img, but single channel
    obstacles = np.zeros_like(img[:,:,0])

    # Obstacles; contains "True" where threshold was met; original was <, not >
    below_thresh = (img[:,:,0] < rgb_thresh[0]) \
                & (img[:,:,1] < rgb_thresh[1]) \
                & (img[:,:,2] < rgb_thresh[2])

    # Index the array of zeros with the boolean array and set to 1
    obstacles[below_thresh] = 1

    # Return the binary image
    return obstacles


# Identifies the rocks to pick up
def rocks_thresh(img):
    # Convert RGB to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Define range of yellow color in HSV
    lower_yellow = np.array([0, 100, 100])
    higher_yellow = np.array([55, 255, 255])

    # Threshold the HSV image to only get yellow colors
    masked_rocks = cv2.inRange(hsv, lower_yellow, higher_yellow)

    # Return the image
    return masked_rocks


# Define a function to convert to rover-centric coordinates
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()

    '''
    Calculate pixel positions with reference to the rover position being at the
    center bottom of the image.
    '''
    x_pixel = np.absolute(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[0]).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    '''
    Convert (x_pixel, y_pixel) to (distance, angle)
    in polar coordinates in rover space
    Calculate distance to each pixel
    '''
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles


# Define a function to apply a rotation to pixel positions
def rotate_pix(xpix, ypix, yaw):

    # Yaw angle is recorded in degrees so first convert to radians
    yaw_rad = yaw * np.pi / 180

    xpix_rotated = xpix * np.cos(yaw_rad) - ypix * np.sin(yaw_rad)
    ypix_rotated = xpix * np.sin(yaw_rad) + ypix * np.cos(yaw_rad)
    # Return the result
    return xpix_rotated, ypix_rotated


# Performs a translation
def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale):

    # Perform translation and convert to integer since
    # pixel values can't be float
    xpix_translated = np.int_(xpos + (xpix_rot / scale))
    ypix_translated = np.int_(ypos + (ypix_rot / scale))

    # Return the result
    return xpix_translated, ypix_translated


# Applies rotation and translation (and clipping)
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world


# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image

    return warped


# Applies above functions in succession and updates the Rover state accordingly
def perception_step(Rover):

    '''
    Define calibration box in source (actual) and destination (desired) coordinates
    These source and destination points are defined to warp the image
    to a grid where each 10x10 pixel square represents 1 square meter
    The destination box will be 2 * dst_size on each side
    '''
    dst_size = 5

    '''
    Set a bottom offset to account for the fact that the bottom of the image
    is not the position of the rover but a bit in front of it
    this is just a rough guess, feel free to change it!
    '''
    bottom_offset = 6

    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - bottom_offset],
                      [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - bottom_offset],
                      [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - 2 * dst_size - bottom_offset],
                      [Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - 2 * dst_size - bottom_offset],
                      ])


    # Applies perspective transform
    warped = perspect_transform(Rover.img, source, destination)


    # Applies color threshold to identify navigable terrain/obstacles/rock samples
    ground_thresh = color_thresh(warped)
    obstacle_thresh = obstacles_thresh(warped)
    rock_thresh = rocks_thresh(warped)
    

    # Updates Rover.vision_image (this will be displayed on left side of screen)
    Rover.vision_image[:,:,0] = obstacle_thresh * 155
    Rover.vision_image[:,:,1] = rock_thresh
    Rover.vision_image[:,:,2] = ground_thresh * 255


    # Converts map image pixel values to rover-centric coords
    xpix, ypix = rover_coords(ground_thresh)
    obstacles_xpix, obstacles_ypix = rover_coords(obstacle_thresh)
    rocks_xpix, rocks_ypix = rover_coords(rock_thresh)


    # Converts rover-centric pixel values to world coordinates
    rover_xpos, rover_ypos, rover_yaw = Rover.pos[0], Rover.pos[1], Rover.yaw

    scale = dst_size * 2

    # Gets navigable pixel positions in world coords
    navigable_x_world, navigable_y_world = pix_to_world(xpix, ypix,
                                    rover_xpos, rover_ypos, rover_yaw,
                                    Rover.worldmap.shape[0], scale)

    obstacles_x_world, obstacles_y_world = pix_to_world(obstacles_xpix, obstacles_ypix,
                                    rover_xpos, rover_ypos, rover_yaw,
                                    Rover.worldmap.shape[0], scale)

    rocks_x_world, rocks_y_world = pix_to_world(rocks_xpix, rocks_ypix,
                                    rover_xpos, rover_ypos, rover_yaw,
                                    Rover.worldmap.shape[0], scale)


    # Updates Rover worldmap (to be displayed on right side of screen)

    # Limit roll and pitch to increase fidelity
    roll_limit = .6
    pitch_limit = .6

    pitch = min(abs(Rover.pitch), abs(Rover.pitch - 359.5))
    roll = min(abs(Rover.roll), abs(Rover.roll - 359.5))

    # Updates worldmap only when within the roll and pitch bounds
    if (abs(pitch) < pitch_limit and abs(roll) < roll_limit):
        Rover.worldmap[obstacles_y_world, obstacles_x_world, 0] += 1
        Rover.worldmap[rocks_y_world, rocks_x_world, 1] += 1
        Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1


    # Converts rover-centric pixel positions to polar coordinates
    navigable_distance, navigable_angles = to_polar_coords(xpix, ypix)

    Rover.nav_dists = navigable_distance
    Rover.nav_angles = navigable_angles

    # Updates rock pixel distances and angles
    rock_distance, rock_angles = to_polar_coords(rocks_xpix, rocks_ypix)

    Rover.rock_dists = rock_distance
    Rover.rock_angles = rock_angles


    return Rover
