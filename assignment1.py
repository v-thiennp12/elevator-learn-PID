from time import perf_counter
import numpy as np
from sim.elevator import sim_run

# Simulator Options
options = {}
options['FIG_SIZE']             = [8, 8] # [Width, Height]
options['PID_DEBUG']            = True

# Physics Options
options['GRAVITY']              = True
options['FRICTION']             = True
options['ELEVATOR_MASS']        = 1000
options['COUNTERWEIGHT_MASS']   = 1000
options['PEOPLE_MASS']          = 150

# Controller Options
options['CONTROLLER']           = True
options['START_LOC']            = 27.0 #3.0
options['SET_POINT']            = 3.0  #27.0
options['OUTPUT_GAIN']          = 2000

t_end                           = 20 #s 30.2

class Controller:
    def __init__(self, reference):
        self.r           = reference
        self.dt          = 0.05

        self.prev_time   = 0.0
        self.prev_error  = None
        self.int_err     = 0.0
        self.windup      = 5.0
        self.output      = 0.0
        self.output_max  = 2.5
        self.output_data = [[0.0, 0.0, 0.0, 0.0]]

    def run(self, x, t):
        k_P          = 2.0 #1.5
        k_D          = 3.0 #2.5
        k_I          = 0.75 #0.3

        P_out        = 0.0
        D_out        = 0.0
        I_out        = 0.0
        # self.output  = 0.0 #important, need to reset self.output at each new step

        # Controller run time.
        
        if t - self.prev_time < 0.05:
            return self.output
        else:
        #if (self.prev_time != None):
            # print(self.prev_time)
            # print(t)
            dt           = t - self.prev_time
           
            #P control
            error        = (self.r - x)
            P_out        = k_P*error

            #D control
            if (self.prev_error != None):
                error_dot = (error - self.prev_error)/dt
                D_out     = k_D*error_dot

            self.prev_error   = error
            
            #I control
            self.int_err += error * dt
            #windup prevention
            if (self.int_err > self.windup):
                self.int_err = self.windup
            elif (self.int_err < -1*self.windup):
                self.int_err = -1*self.windup

            I_out         = k_I*self.int_err

            #total PID controller output
            
            self.output     = P_out + D_out + I_out

            #saturation of actuator (before output gain, it's not realistic)
            if (self.output > self.output_max):
                self.output = self.output_max
            elif (self.output < -1*self.output_max):
                self.output = -1*self.output_max

            self.prev_time  = t

            # debug PID
            self.output_data = np.concatenate((self.output_data, np.array([[t, P_out, I_out, D_out]])), axis=0)
                        
            return self.output            

sim_run(options, Controller, t_end)


        # self.output  = 0.0 #important, need to reset self.output at each new step if use += opertor
        # self.output  += P_out
        # self.output  += D_out
        # self.output  += I_out