import glob
import os
import sys
import numpy as np
import math

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import argparse
import logging
import random

def main():
    
    # --------------
    # User-defined variables
    # --------------
    display_spectator_position = True
    display_elapsed_time = True
    selected_map = 'Town02'
    geofence = [98.118385, 169.524979, 170.196945, 226.819290] # Lower x-bound,
                                 # upper x-bound, lower y-bound, upper y-bound
    
    # --------------
    # Parse command line inputs
    # --------------
    
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '-n', '--number-of-vehicles',
        metavar='N',
        default=50,
        type=int,
        help='number of vehicles (default: 50)')
    argparser.add_argument(
        '--safe',
        action='store_true',
        help='avoid spawning vehicles prone to accidents')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-f', '--recorder_filename',
        metavar='F',
        default="test1.log",
        help='recorder filename (test1.log)')
    argparser.add_argument(
        '-tm_p', '--tm_port',
        metavar='P',
        default=8000,
        type=int,
        help='port to communicate with TM (default: 8000)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '--filterv',
        metavar='PATTERN',
        default='vehicle.*',
        help='vehicles filter (default: "vehicle.*")')
    args = argparser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    print("Recording simulation playback on file: %s" % client.start_recorder("test1.log"))

    # Create output .txt file for all agents and include a header
    for i in range(args.number_of_vehicles):
        myrecording = open(r"D:\carla\CARLAUE4\Saved\myrecording%d.txt" % (i),"w+")
        myrecording.write('Simulation Frame | Timestamp | Position | Velocity ' +
                          '| Heading | Angular Velocity | Acceleration | ' +
                              'Stopped at Red Light + Light ID \n')
        myrecording.close()
    try:
        client.load_world(selected_map) # Load a new map (Town02 in this case)
        client.reload_world()
        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(2.0)
        world = client.get_world()
        Map = world.get_map()
        
        # Set time of day to noon
        weather = carla.WeatherParameters(sun_altitude_angle=90)
        world.set_weather(weather)
        
        synchronous_master = False

        if args.sync:
            settings = world.get_settings()
            traffic_manager.set_synchronous_mode(True)
            if not settings.synchronous_mode:
                synchronous_master = True
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
                world.apply_settings(settings)
            else:
                synchronous_master = False
              
        # --------------
        # Place spectator view above our chosen intersection
        # --------------
        
        spectator = world.get_spectator()
        spectator.set_transform(carla.Transform(carla.Location(x=135.503845, \
                                                    y=197.694901, z=52.118252),
        carla.Rotation(pitch=-88.995552, yaw=-90.038940, roll=0.008958)))

        # --------------
        # Import Blueprints
        # --------------

        blueprints = world.get_blueprint_library().filter(args.filterv)

        if args.safe:
            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
            blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
            blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('t2')]

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if args.number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif args.number_of_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, args.number_of_vehicles, number_of_spawn_points)
            args.number_of_vehicles = number_of_spawn_points

        # --------------
        # Spawn vehicles
        # --------------

        actor_list = []
        for n, transform in enumerate(spawn_points):
            if n >= args.number_of_vehicles:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')
            actor = world.spawn_actor(blueprint,transform)
            actor.set_autopilot(True)

            actor_list.append(actor)

        print('Spawned %d vehicles, press Ctrl+C to exit.' % (args.number_of_vehicles))

        # wait for a tick to ensure client receives the last transform of the vehicles we have just created
        if not args.sync or not synchronous_master:
            framestamp = world.wait_for_tick()
            frame_zero = framestamp.frame
            time_zero = framestamp.elapsed_seconds
        else:
            world.tick()
        
        # --------------
        # Game loop. Prevents the script from finishing.
        # --------------
        headings = np.zeros(args.number_of_vehicles) # Initialize headings of vehicles
        while True:
                 
            if args.sync and synchronous_master:
                world.tick()
            else:
                framestamp = world.wait_for_tick()
                current_frame = framestamp.frame - frame_zero
                time_elapsed = framestamp.elapsed_seconds - time_zero
                if not (np.mod(current_frame,10)): # Display info every 10 frames
                    if display_spectator_position:
                        print('Spectator position: ')
                        print(str(world.get_spectator().get_transform())+'\n')
                    if display_elapsed_time:
                        print('Elapsed time: ')
                        print(str(time_elapsed)+'\n')
            
            if not (np.mod(current_frame,10)):
                for i in range(args.number_of_vehicles):
                    myrecording = open(r"D:\carla\CarlaUE4\Saved\myrecording%d.txt" % (i),"a")
                    
                    # Check if vehicle stopped by traffic light
                    red_light = 0
                    if actor_list[i].is_at_traffic_light():
                        traffic_light = actor_list[i].get_traffic_light()
                        red_light = 1
                    
                    # Apply geofence
                    location = actor_list[i].get_location()
                    if location.x > geofence[0] and location.x < geofence[1] and \
                        location.y > geofence[2] and location.y < geofence[3]:
                    
                        # Convert x-y-z to Lat,Lon,Alt (unused at the moment)
                        coords = Map.transform_to_geolocation(location)
                        
                        # Compute heading
                        speed = math.sqrt(actor_list[i].get_velocity().x**2 + \
                                          actor_list[i].get_velocity().y**2)
                        
                        # Compute new heading is vehicle is moving
                        if speed > 1:
                            heading = round(np.mod(math.atan2(actor_list[i].get_velocity().x, \
                                            actor_list[i].get_velocity().y)*180/math.pi, 360),1)
                            headings[i] = heading
                        else:
                            # If speed is low (i.e. vehicle is stopped, use 
                            # last recorded heading).
                            heading = headings[i]
                       
                        # Convert data values to strings before outputting
                        my_frame = str(current_frame) + ' | '
                        my_timestamp = str(time_elapsed) + ' | '
                        my_location = str(location.x) + ',' + str(location.y) + \
                            ',' + str(location.z) + ' | '
                        my_velocity = str(actor_list[i].get_velocity().x) + \
                            ',' + str(actor_list[i].get_velocity().y) + ',' + \
                                str(actor_list[i].get_velocity().z) + ' | '
                        my_heading = str(heading) + ' | '
                        my_angular_velocity = str(actor_list[i].get_angular_velocity().x) + \
                            ',' + str(actor_list[i].get_angular_velocity().y) + ',' + \
                                str(actor_list[i].get_angular_velocity().z) + ' | '
                        my_acceleration = str(actor_list[i].get_acceleration().x) + \
                            ',' + str(actor_list[i].get_acceleration().y) + ',' + \
                                str(actor_list[i].get_acceleration().z) + ' | '
                        if red_light:
                            my_light = str(red_light) + ',' + str(traffic_light.id) + '\n'
                        else:
                            my_light = str(red_light) + ',0\n'
                        
                        # Write to output .txt file
                        myrecording.write(my_frame)
                        myrecording.write(my_timestamp)
                        myrecording.write(my_location)
                        myrecording.write(my_velocity)
                        myrecording.write(my_heading)
                        myrecording.write(my_angular_velocity)
                        myrecording.write(my_acceleration)
                        myrecording.write(my_light)
                        myrecording.close()

    finally:
        # --------------
        # Destroy actors
        # --------------
        
        client.stop_recorder()
        print('--------------------------------------------')
        print('Playback file saved to ~\CarlaUE4\Saved')
        print('Output .txt files saved to ~\CarlaUE4\Saved')
        na = 0
        actors = world.get_actors()
        for a in actors.filter('vehicle.*'):
            a.destroy()
            na = na+1
        print('Destroyed %d vehicles' % na)
        
        
if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\nDone running. Good bye!')
        print('--------------------------------------------\n')
