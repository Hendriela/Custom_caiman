# -*- coding: utf-8 -*-
"""
Created on Tue May 07 2019

@author: Adrian
"""

## hardcoded important parameters
PORT = 49552
CONNECTION_TIMEOUT = 120  # s

GAIN = 0.2     # make position translation smaller or larger
NR_RUNS = 1000    # let LabVIEW stop the program...
DEBUG = False     # True or False, if DEBUG more messages are displayed

BUILD_FOLDER = 'version_03_vr'
SAVE_LOGFILE = True
LOGFILE_PATH = 'test_logfile_vrCamera.txt'

# Initialize
import controller_true_vr as c
import time
from mlagents.envs import UnityEnvironment
import os


# this controller sends and receives messages
# string commands are stored in FIFO cache and can be popped with fifo_get_job()
# velocity commands are stores in a variable that can be accessed any time
controller = c.ServerController( port = PORT, debug=DEBUG, connection_wait_time=CONNECTION_TIMEOUT)
controller.start()

# Object to keep track of the current position and reward state in the corridor
state = c.LocationState( connection = controller.server_thread )

#%% Start VR environment with black screen

env_name = os.path.abspath(os.path.join(
    os.path.abspath(os.curdir), BUILD_FOLDER, 'mouse_corridor'))

env = UnityEnvironment(file_name=env_name)

# Set the default brain to work with
default_brain = env.brain_names[0]
brain = env.brains[default_brain]

# Reset the environment to show view from black box (this combination seems to work...)
env_info = env.reset(train_mode=False)[default_brain]
env_info = env.step([0, 0])[default_brain]
env_info = env.reset(train_mode=False)[default_brain]


#%% Wait for connections

for i in range(CONNECTION_TIMEOUT+1):
    ## wait for command to start to be sent from LabVIEW
    command = controller.fifo_get_job()

    if command is None:
        time.sleep(1)  # wait
    elif command == 'start_vr':
        print('Starting game...')
        break
    else:
        raise Exception('Invalid command "{:s}" received'.format(command))

    if i == CONNECTION_TIMEOUT:
        env.close()
        raise Exception('Waiting for connection timed out!')


#%% Run VR

completed_runs = 0

while True:

    if state.in_iti == True:
        # check for trial start command
        command = controller.fifo_get_job()
        time.sleep( 0.01)

        if command == 'start_trial':
            env_info = env.reset(train_mode=False)[default_brain]
            env_info = env.step([0, 1])[default_brain]  # move mouse out of black box (1)
            state.start_trial()

        elif command == 'stop_vr':
            print('Stopping main loop.')
            break

        elif command is not None:
            print('Received wrong command: {:s}'.format(command))

    else:  # mouse in corridor
        # print('Mouse in corridor')
        vel = controller.get_velocity()
        # print(vel)
        if vel == None:
            vel = 0  # avoid sending none to VR in any case
            print('Warning: None velocity had to be transformed to 0.')
        # step takes two arguments: First is velocity, second is reset to start position
        env_info = env.step([vel * GAIN, 0])[default_brain]

        # get position, zone and information about corridor end from return value
        position = env_info.vector_observations[0][0]
        # velocity = env_info.vector_observations[0][1]   # for testing
        zone = int( env_info.vector_observations[0][2] )
        corridor_end = env_info.local_done[0]

        if corridor_end:
            # at the end reset manually as well to put in black box for sure...
            env_info = env.reset(train_mode=False)[default_brain]

        # update the location state with these variables
        # this function does quite a lot, including sending messages about changed
        # reward states to LabVIEW as well as setting state.in_iti (inter trial interval)
        # to true at the end of the corridor
        state.update_location_state( position, zone, corridor_end )

        # send new VR position to LabVIEW
        controller.server_thread.send_message(1, position)

        if corridor_end:
            completed_runs += 1


        # handle commands from LabVIEW
        new_job = controller.fifo_get_job()
        if new_job is not None:

            if new_job == 'stop_vr':
                print('Shutting down Unity env.step(...) loop.')
                break
            else:
                print('Warning: The following command is being ignored:')
                print(new_job)

        if completed_runs == NR_RUNS:
            print('Maximum number of runs reached. Stopping game!')

            break

#%% Shutdown

controller.stop()
env.close()

if SAVE_LOGFILE == True:
    state.write_log_file(LOGFILE_PATH, overwrite=True)
