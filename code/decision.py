import numpy as np
import time


'''
This is where a decision tree is built for determining throttle, brake and steer
commands based on the output of the perception_step() function
'''


'''
Translate arctan angle in range of (0, 2pi)
Calculate angle rror between current yaw and target angle.
x1:     target point
x2:     current point
yaw:    rover yaw in degrees
'''
def calculate_angle_error(x1, x2, yaw):
    starting_point = np.array(x1)
    rover_point = np.array(x2)
    angle_rad = np.arctan2(starting_point[1] - rover_point[1], 
                           starting_point[0] - rover_point[0])
    
    # Transfer the negative angle_rad to (pi, 2pi)
    if angle_rad < 0:
        angle_rad = angle_rad + 2 * np.pi
    # Transfer radians to degrees
    angle_degree = angle_rad * 180 / np.pi
    angle_error = angle_degree - yaw
    
    return angle_error
    
    
'''
Return home once tasks are completed successfully.
'''
def return_home(Rover):
    # Calculate angle and distance from current point to starting point.
    Rover.angle_error = calculate_angle_error(Rover.start_pos, 
                                              Rover.pos, Rover.yaw)
    
    if np.absolute(Rover.angle_error) > 0.5 and not Rover.turn_home:
        if Rover.vel > 0.2:
            Rover.throttle = 0
            Rover.brake = Rover.brake_set
            Rover.steer = 0
        
        # Turn the rover towards the starting point
        elif Rover.vel <= 0.2:
            if np.absolute(Rover.angle_error) > 1:
                # Stop and acquire vision data to determine path forward
                Rover.throttle = 0
                # Release brakes to start turning
                Rover.brake = 0
                '''
                Turning range is +/- 15 degrees
                When stopped this will induce a 4-wheel turn
                '''
                Rover.steer = np.clip(Rover.angle_error, -15, 15)
                # Update status
                Rover.angle_error = calculate_angle_error(Rover.start_pos,
                                    Rover.pos, Rover.yaw)
            
            else:
                Rover.steer = 0
                Rover.turn_home = True
    
    # If angle error is smaller than 0.5, move back home
    else:
        if len(Rover.nav_angles) >= Rover.stop_forward:
            '''
            Move towards the starting point with obstacle avoidance
            Using 30% and 70% larger navigation angles as steer boundaries
            Rover steer angle is the angle error between yaw and home point
            '''
            nav_angle_b70 = np.percentile(np.array(Rover.nav_angles *
                            180 / np.pi), 70)
            nav_angle_b30 = np.percentile(np.array(Rover.nav_angles *
                            180 / np.pi), 30)
            nav_angle_lower = np.maximum(nav_angle_b30, -15)
            nav_angle_upper = np.maximum(nav_angle_b70, 15)
            
            Rover.steer = np.clip(Rover.angle_error, nav_angle_lower,
                                  nav_angle_upper)
            
            Rover.angle_error = calculate_angle_error(Rover.start_pos, 
                                Rover.pos, Rover.yaw)
            
            Rover.brake = 0
            
            # Use a larger throttle to go back home
            if Rover.vel < Rover.max_vel:
                Rover = stuck_mode(Rover)
            else:
                Rover.throttle = Rover.throttle_set
        
        elif len(Rover.nav_angles) < Rover.stop_forward:
            Rover.throttle = 0
            Rover.brake = Rover.brake_set
            Rover.steer = 0
            Rover.mode = 'stop'
            Rover = stop_mode(Rover)
    
            
    return Rover




# Mode for when the rover is stuck
def stuck_mode(Rover):
    Rover.throttle = Rover.throttle_set * 2
    
    # Count time stuck
    time_stuck = time.time() - Rover.time_start

    '''
    If rover stuck in low speed in forward mode,
    try to use max throttle first
    '''
    if 5 <= time_stuck < 10:
        Rover.throttle = Rover.max_throttle
    # Try to turn the rover
    elif 10 <= time_stuck < 11:
        Rover.throttle = 0
        Rover.brake = 0
        Rover.steer = -15
    elif 11 <= time_stuck < 12:
        Rover.steer = 0
        Rover.throttle = Rover.max_throttle
    elif 12 <= time_stuck < 13:
        Rover.throttle = 0
        Rover.brake = 0
        Rover.steer = -15
    elif 14 <= time_stuck < 16:
        Rover.steer = 0
        Rover.throttle = Rover.max_throttle
    elif 16 <= time_stuck < 17:
        Rover.throttle = -1
        Rover.brake = 0
        Rover.steer = 10
    elif time_stuck > 17:
        # Reset stuck timer
        Rover.time_start = time.time()
        time_stuck = 0
        Rover.throttle = 0
        
    return Rover


'''
Stop mode is complementary to Stuck Mode    
Stop and turn if there is no path forward
'''
def stop_mode(Rover):
    if Rover.mode == 'stop':
        # If still moving, keep braking
        if Rover.vel > 0.2:
            Rover.throttle = 0
            Rover.brake = Rover.brake_set
            Rover.steer = 0
        # If not moving (velocity under 0.2), do something else
        elif Rover.vel <= 0.2:
            if len(Rover.nav_angles) < Rover.go_forward:
                Rover.throttle = 0
                Rover.brake = 0
                Rover.steer = -15
            if len(Rover.nav_angles) >= Rover.go_forward:
                Rover.throttle = Rover.throttle_set
                Rover.brake = 0
                Rover.steer = np.clip(np.mean(Rover.nav_angles * 180 / np.pi), -15, 15)
            Rover.mode = 'forward'
        
    return Rover
    

def decision_step(Rover):

    # Record starting point for returning purposes
    if Rover.count == 0:
        Rover.start_pos = Rover.pos
        
    '''
    TO-DO:
        1) Apply functionality not to revisit paths.
        2) Apply edge detection, so as to crawl walls.
    '''

    
    # If any rocks are visible, move toward them
    if len(Rover.rock_angles) > 0:
        Rover.throttle = 0.1
        Rover.steer = np.clip(np.mean(Rover.rock_angles * 180/np.pi), -15, 15)

        
    # If near a rock, pick it up
    if Rover.near_sample and Rover.count > \
    Rover.near_sample_count + Rover.timeout_after_pickup:
        Rover.near_sample_count = Rover.count

        Rover.throttle = 0
        Rover.brake = Rover.brake_set
        Rover.send_pickup = True
        Rover.samples_recovered += 1


    # Check if we have vision data to make decisions with
    if Rover.nav_angles is not None:
        if Rover.samples_recovered < 6:
            # Check for Rover.mode status
            if Rover.mode == 'forward':
    
                # Check the extent of navigable terrain
                if len(Rover.nav_angles) >= Rover.stop_forward:
                    # If mode is forward, navigable terrain looks good
                    # and velocity is below max, then throttle
                    if Rover.vel < Rover.max_vel:
                        # If stuck
                        if Rover.vel < 0.1 and not Rover.picking_up:
                            Rover = stuck_mode(Rover)
                        else:
                            # Set throttle value to throttle setting
                            Rover.throttle = Rover.throttle_set
                    else: # Else coast
                        Rover.throttle = 0
                    Rover.brake = 0
                    # Set steering to average angle clipped to the range +/- 15
                    Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
    
                # If there's a lack of navigable terrain pixels then go to 'stop' mode
                elif len(Rover.nav_angles) < Rover.stop_forward:
                        # Set mode to "stop" and hit the brakes!
                        Rover.throttle = 0
                        # Set brake to stored brake value
                        Rover.brake = Rover.brake_set
                        Rover.steer = 0
                        Rover.mode = 'stop'
    
    
            # If we're already in "stop" mode then make different decisions
            elif Rover.mode == 'stop':
                # If we're in stop mode but still moving keep braking
                if Rover.vel > 0.2:
                    Rover.throttle = 0
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                # If we're not moving (vel < 0.2) then do something else
                elif Rover.vel <= 0.2:
                    # Now we're stopped and we have vision data to see if there's a path forward
                    if len(Rover.nav_angles) < Rover.go_forward:
                        Rover.throttle = 0
                        # Release the brake to allow turning
                        Rover.brake = 0
                        # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                        Rover.steer = -15 # Could be more clever here about which way to turn
                    # If we're stopped but see sufficient navigable terrain in front then go!
                    if len(Rover.nav_angles) >= Rover.go_forward:
                        # Set throttle back to stored value
                        Rover.throttle = Rover.throttle_set
                        # Release the brake
                        Rover.brake = 0
                        # Set steer to mean angle
                        Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
                        Rover.mode = 'forward'

      
        # All samples have been found
        else:
            # Calculate angle and distance from current point to starting point.
            starting_point = np.array(Rover.start_pos)
            rover_point = np.array(Rover.pos)
            Rover.home_dist = np.sqrt(np.sum((rover_point - starting_point) ** 2))
            
            if Rover.home_dist > 2:
                Rover = return_home(Rover)
            
            # Near home
            else:
                Rover.throttle = 0
                Rover.steer = 0
                if Rover.vel > 0.2:
                    Rover.brake = 1
                else:
                    Rover.brake = Rover.brake_set
                
                print("All samples successfully collected.")
            

    #Just to make the rover do something
    #even if no modifications have been made to the code
    
    else:
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0

    return Rover
