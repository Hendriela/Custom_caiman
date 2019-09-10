# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 12:19:40 2019

@author: Adrian
"""

import socket
import threading
from datetime import datetime
import os

# seperate message of type COMMAND:VALUE;
def convert_command(command_string):
    """Convert command string like COMMAND:VALUE into command_nr and value

    Use example:
    convert_command('0:3.2341')    returns (0, 3.2341)
    convert_command('3:start_vr')  returns (3, 'start_vr')

    Adrian 2019-03-11
    """
    # check if two commands are in the string
    try:
        if command_string.split('\r\n')[1] != '':
            print('Warning: More than one command detected! '
                  'Second command "{}" is currently ignored'.format(command_string.split('\r\n')[1]))

    except:
        # more useful exception with received message
        raise Exception('Invalid command: {}'.format( command_string ))

    command = command_string.split('\r\n')[0]
    try:
        number = int( command.split(':')[0] )
        value = command.split(':')[1]
    except Exception as error:
        print('Error during the processing of the following command "%s"' % command)
        print('Throwing error:')
        raise error

    if number == 0:
        # in this case it is the velocity as float number
        value = float( value )

    return number, value



class TcpServerThread(threading.Thread):
    """Class that runs a server that updates a controller with velocity values

    Adrian 2018-02-21
    """

    connection = None
    connection_wait_time = None  # seconds, set in init
    message_wait_time = 120  # seconds
    controller = None   # object of ServerController class

    def __init__(self, controller, host='127.0.0.1', port=49591, debug=False, connection_wait_time=60 ):
        super().__init__()

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.controller = controller
        self.stopEvent = threading.Event()
        self.host = host
        self.port = port
        self.connection_wait_time = connection_wait_time

        self.debug = debug

        # the connection will be established when the thread is started

    def wait_for_connection(self):

        print('Waiting for connections...')
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, True)
        self.socket.bind( (self.host, self.port) )
        self.socket.settimeout(self.connection_wait_time)
        self.socket.listen()

        try:
            self.connection, addr = self.socket.accept()
        except socket.timeout as error:
            print('No client connected to the server within the wait time.\n'
                   'Throwing error after closing the connection:')
            self.close_connection()
            raise error

        self.controller.receiving_input = True

        # set timeout for receiving messages, in case the client fails
        self.connection.settimeout(self.message_wait_time)

        print('Connected to ', addr)

    def close_connection(self):
        # from socket import SHUT_RDWR
        # self.socket.shutdown(SHUT_RDWR)
        self.socket.close()
        self.socket = None
        self.connection = None
        self.controller.receiving_input = False
        print('Connection closed.')

    def stop_server(self):
        self.stopEvent.set()


    def run(self):
        """Main worker that is called to start the server and receive data"""

        print('Server thread started...')
        global current_speed

        self.wait_for_connection()

        while not self.stopEvent.is_set():
            if self.debug: print( datetime.now() )

            try:
                message = self.connection.recv(12*4)  # this blocks until at least 1 message is received

            except socket.timeout as error:
                print('No message was received within the message_wait_time.\n'
                   'Closing the connection and shutting down VR.')
                self.controller.add_job( 'stop_vr' )
                self.stop_server()  # this closes the connection

            except ConnectionResetError as error:
                print('The client has ended the connection. Shutting down VR...')
                self.controller.add_job( 'stop_vr' )
                self.stop_server()  # this closes the connection

            # check if empty message was received => connection terminated
            if not message:
                break

            command = message.decode("utf-8")  # transform to string
            if self.debug: print(command)
            # data was received
            # if self.debug: print('Received message {:s}'.format(command) )
            i = 0
            for command in command.split('\r\n')[:-1]:  # without empty string
                if self.debug: print('Nr % s' % i)
                i = i + 1
                command_nr, value = convert_command(command + '\r\n') # for backwards compatibility

                if command_nr == 3 and value == 'stop_vr':
                    print('Thread received stopping command, triggering stop event.')
                    self.stop_server()


                if command_nr == 0:
                    # update controller with current velocity value (value is a float in this case)
                    self.controller.set_velocity( value )

                elif command_nr == 3:
                    # transform commands back to old format (not used currently):
                    # if value == 'stop_vr_': value = 'stop_vr'
                    # elif value == 'st_trial': value = 'start_trial'

                    # add the status command to the job queue of the controller
                    self.controller.add_job( value )

                else:
                    print('The received command is currently not supported: {:s}'.format(command))


        self.close_connection()

        print('Server thread stopped.')

    def send_message(self, command_nr, value):
        if command_nr == 3:
            message = '{:d}:{:s}'.format(command_nr, value) + '\r\n'
        elif command_nr == 1:
            message = '{:d}:{:.3f}'.format(command_nr, value) + '\r\n'
        else:
            raise Exception('Command number {} not supported yet'.format(command_nr))

        if self.connection is not None:
            if self.debug: print('Sending message "{}"'.format(message) )
            self.connection.sendall( message.encode( encoding='utf-8' ))

        else:
            print('The message "{}" could not be sent because connection closed.'.format(message))



class ServerController():

    current_vel = None
    job_queue = list()
    receiving_input = False
    server_thread = None

    def __init__(self, port = 49591, debug=False, connection_wait_time=60):

        self.server_thread = TcpServerThread( self, port=port, debug=debug,
                                            connection_wait_time=connection_wait_time)

    def start(self):
        self.server_thread.start()

    def stop(self):
        self.server_thread.stop_server()


    def set_velocity( self, velocity):
        self.current_vel = velocity

    def get_velocity( self):
        global current_vel

        if self.current_vel == None:
            print('Error: The server is not connected to the client! Returning None')
        else:
            return self.current_vel

    def add_job(self, new_status):
        self.job_queue.append( new_status )

    def fifo_get_job(self):
        """Get job from queue in 'first in, first out' way. Returns None if queue is empty"""

        if len(self.job_queue) == 0:
            # no job to be done
            return None
        else:
            # get the job that was added first and delete it from list
            return self.job_queue.pop(0)



class LocationState():
    """Class to keep track of the current position and reward state in the corridor

    TODO: documentation

    Adrian 2019-03-12
    """
    position = None     # if None, mouse is in iti box, else float along corridor

    # here only the VR knows the reward positons!!
    # reward_locations = list()    # format [ [30,40], [70,80] ] for two reward zones
    last_reward_zone_nr = -1
    in_reward_zone = False
    in_iti = True    # iti: inter-trial interval

    debug = None
    max_position = None
    last_reward_zone_entry = None     # variable to save position at reward beginning
    connection = None

    # logging of experimental parameters
    logging = None    # true or false set in __init__
    time_log = list()
    pos_log = list()
    zone_log = list()

    def __init__(self, connection=None, debug=True, logging=True):
        """ Initialized the object
        """
        self.connection = connection

        self.debug = debug
        self.logging = logging

    def update_location_state(self, position, reward_zone, corridor_end):
        """ Update the state with the return values of the env.step() command

        Parameters:
        --------
        TODO


        """

        # check if this is a valid call of the function
        if self.in_iti == True:
            raise Exception('This function can not be called in the inter-trial interval. Call start_trial() before.')

        # save given position
        self.position = position

        # check if inter trial interval was entered
        if corridor_end:
            self.entering_iti()

        #%% update the reward state of the mouse
        elif self.in_reward_zone == False:
            # mouse is in the normal corridor (no reward zone)

            if reward_zone > self.last_reward_zone_nr:
                # mouse entered a new reward zone it has not been in before
                self.entering_reward_zone( reward_zone )
                self.last_reward_zone_entry = position
            else:
                pass  # mouse still in normal corridor, jump to logging

        else:  # mouse is in reward zone
            # did the mouse leave the current reward zone?
            if reward_zone == -1: # -1 means normal corridor
                # yes => check if position is larger than upon entering the reward
                # zone to avoid jitter at beginning of reward zone to result in leaving
                # of reward zone
                if position > (self.last_reward_zone_entry + 1 ):
                    self.leaving_reward_zone()
            else:
                pass  # mouse still in current reward zone

        #%% logging of some parameters
        if self.logging == True:
            self.time_log.append( datetime.now() )
            self.pos_log.append( self.position )
            self.zone_log.append( reward_zone )



    def entering_reward_zone(self, current_zone):
        """Executed when mouse enters the reward zone"""

        if self.debug: print('Mouse entered reward zone {:d}'.format(current_zone))

        self.in_reward_zone = True
        self.last_reward_zone_nr = current_zone

        if self.connection is not None:
            # send this state to LabVIEW
            self.connection.send_message(3,'enter_reward')

    def leaving_reward_zone(self):
        """Executed when mouse left the reward zone"""

        if self.debug: print('Mouse left reward zone {:d}'.format(self.last_reward_zone_nr))

        self.in_reward_zone = False

        if self.connection is not None:
            # send this state to LabVIEW
            self.connection.send_message(3,'leave_reward')

    def entering_iti(self):

        if self.debug: print('Mouse entered inter-trial-interval zone')
        self.in_iti = True
        self.last_reward_zone_nr = -1
        self.in_reward_zone = False   # just to make sure
        self.position = None   # this defines ITI position

        if self.connection is not None:
            # send this state to LabVIEW
            self.connection.send_message(3,'end_corridor')

    def start_trial(self):
        if self.debug: print('Starting trial')
        self.in_iti = False
        self.position = 0


    def write_log_file(self, file_name, overwrite=False):

        if os.path.isfile( file_name ):
            # file already exists
            if overwrite == False:
                raise Exception('File {} already exists! Aborting...'.format(file_name))
            else:
                print('Warning: Overwriting file {}.'.format(file_name))

        with open(file_name, 'w') as f:
           f.write('Timestamp\tPosition\tZone\n')  # header

           for t, pos, rew in zip(self.time_log,
                                  self.pos_log,
                                  self.zone_log):
               if pos is not None:
                   f.write( '{}\t{:.3f}\t{}'.format(t, pos, rew) + '\n')
