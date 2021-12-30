import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import time
from scipy.integrate import ode
import uuid

def sim_run(options, PidController, t_end):
    start       = time.clock()
    # Simulator Options
    FIG_SIZE    = options['FIG_SIZE'] # [Width, Height]
    PID_DEBUG   = options['PID_DEBUG']

    # Physics Options
    GRAVITY     = options['GRAVITY']
    FRICTION    = options['FRICTION']
    E_MASS      = options['ELEVATOR_MASS']
    CW_MASS     = options['COUNTERWEIGHT_MASS']
    P_MASS      = options['PEOPLE_MASS']

    # Controller Options
    CONTROLLER  = options['CONTROLLER']
    START_LOC   = options['START_LOC']
    SET_POINT   = options['SET_POINT']
    OUTPUT_GAIN = options['OUTPUT_GAIN']

    Pid = PidController(SET_POINT)

    # ODE Solver
    def elevator_physics(t, state):
        # State vector.
        x           = state[0]
        x_dot       = state[1]

        # Acceleration of gravity.
        g           = -9.8
        x_dot_dot   = 0

        if CONTROLLER:
            x_dot_dot += Pid.run(x,t) * OUTPUT_GAIN / (E_MASS + CW_MASS + P_MASS)
        if GRAVITY:
            x_dot_dot += g*(E_MASS + P_MASS - CW_MASS) / (E_MASS + P_MASS + CW_MASS)
        if FRICTION:
            x_dot_dot -= x_dot * 0.2

        #print(t, x_dot, x_dot_dot)
        # Output state derivatives.
        return [x_dot, x_dot_dot]

    # ODE Info.
    solver = ode(elevator_physics)
    solver.set_integrator('dopri5')

    # Set initial values.
    t0      = 0.0
    #t_end   = 30.2
    #dt      = 0.05
    dt      = Pid.dt
    t       = np.arange(t0, t_end, dt)

    # Solution array and initial states.
    sol             = np.zeros((int(t_end/dt), 3))
    state_initial   = [START_LOC, 0.0]
    solver.set_initial_value(state_initial, t0)
    sol[0]          = [state_initial[0], state_initial[1], 0.0]
    prev_vel        = state_initial[1]

    # Repeatedly call the `integrate` method to advance the
    # solution to time t[k], and save the solution in sol[k].
    k = 1
    while solver.successful() and solver.t < t_end-dt:
        solver.integrate(t[k])
        sol[k] = [solver.y[0], solver.y[1], (solver.y[1]-prev_vel)/dt]
        #print(sol[k])
        k += 1
        prev_vel = solver.y[1]

    state = sol


    ###################
    # SIMULATOR DISPLAY


    def update_plot(num):
        #print(state[num])

        # Time bar.
        time_bar.set_data([7.8, 7.8], [0, num*dt])

        # Elevator.
        el_l.set_data([3, 3],[state[num,0], state[num,0]+3])
        el_r.set_data([6, 6],[state[num,0], state[num,0]+3])
        el_t.set_data([3, 6],[state[num,0]+3, state[num,0]+3])
        el_b.set_data([3, 6],[state[num,0], state[num,0]])

        # Timer.
        time_text.set_text(str(round(num*dt+0.04,1)) + 's')

        # Strip Chart.
        pos.set_data(t[0:num], state[0:num,0])
        vel.set_data(t[0:num], state[0:num,1])
        acc.set_data(t[0:num], state[0:num,2])

        # Status
        if abs(state[num,1]) < 0.01 and abs(SET_POINT-state[num,0]) < 0.03:
            pos_status.set_text('PASS')
        if abs(state[num,1]) > 18 and len(vel_status.get_text()) < 1:
            vel_status.set_text('FAIL')
        if abs(state[num,2]) > 5 and len(acc_status.get_text()) < 1:
            acc_status.set_text('FAIL')

        # Debug time line.
        if PID_DEBUG:            
            err_status.set_text(str(np.round(SET_POINT - state[num,0], 3)))
            err_status.set_position((num*dt + 0.5, SET_POINT - state[num,0] + 1))

            err_line.set_data([num*dt, num*dt], [-1000, 1000])
            p_line.set_data([num*dt, num*dt], [-1000, 1000])
            i_line.set_data([num*dt, num*dt], [-1000, 1000])
            d_line.set_data([num*dt, num*dt], [-1000, 1000])
            return time_bar, el_l, el_r, el_t, el_b, time_text, \
                   pos, vel, acc, acc_status, vel_status, pos_status, err_status, \
                   err_line, p_line, i_line, d_line
        else:
            return time_bar, el_l, el_r, el_t, el_b, time_text, \
                   pos, vel, acc, acc_status, vel_status, pos_status


    # Total Figure
    fig = plt.figure(figsize=(FIG_SIZE[0], FIG_SIZE[1]))
    gs = gridspec.GridSpec(14,8)

    # Elevator plot settings.
    ax = fig.add_subplot(gs[:, :2])
    plt.xlim(0, 8)
    
    if (SET_POINT > START_LOC):
        min_Y = START_LOC - 1
        max_Y = SET_POINT + 3
    else:
        min_Y = SET_POINT - 1
        max_Y = START_LOC + 3

    plt.ylim(min_Y, max_Y)
    #plt.ylim(0, SET_POINT + 1)

    plt.xticks([])
    plt.yticks(np.arange(0,SET_POINT*2+1,3))
    plt.title('Elevator')

    # Time display.
    time_text = ax.text(6, 0.5, '', fontsize=15)

    # Floor Labels.
    floors          = ['  _start', '  _roof', '  SET_POINT']
    floor_height    = [START_LOC, START_LOC + 3, SET_POINT]
    floor_x         = [0.25, 0.25, 0.25]
    for i in range(len(floors)):
        ax.text(floor_x[i], floor_height[i] + 0.5, floors[i])
        ax.plot([0, 1.5], [floor_height[i], floor_height[i]], 'b-')

    # Plot info.
    time_bar, = ax.plot([], [], 'r-')
    el_l, el_r = ax.plot([], [], 'k-', [], [], 'k-')
    el_t, el_b = ax.plot([], [], 'k-', [], [], 'k-')

    # Strip chart settings.
    strip_width = 4
    if PID_DEBUG:
        strip_width = 5

    # Position
    ax = fig.add_subplot(gs[0:4, strip_width:])
    pos, = ax.plot([], [], '-b')
    pos_status = ax.text(1.0, SET_POINT, '', fontsize=20, color='g')
    plt.title('Position')
    plt.xticks([0,t_end])
    plt.xlim(0, t_end)
    if SET_POINT > START_LOC:
        plt.ylim(START_LOC - 10, SET_POINT+10)
    else:
        plt.ylim(SET_POINT - 10, START_LOC+10)

    # Velocity
    ax = fig.add_subplot(gs[5:9, strip_width:])
    vel, = ax.plot([], [], '-b')
    vel_status = ax.text(1.0, -18.0, '', fontsize=20, color='r')
    plt.title('Velocity')
    plt.xticks([0,t_end])
    plt.xlim(0, t_end)
    plt.ylim(-20, 20)

    # Acceleration
    ax = fig.add_subplot(gs[10:14, strip_width:])
    acc, = ax.plot([], [], '-b')
    acc_status = ax.text(1.0, -4.0, '', fontsize=20, color='r')
    plt.title('Acceleration')
    plt.xlabel('Time (s)')
    plt.xlim(0, t_end)
    plt.ylim(-5, 5)

    if PID_DEBUG:
        debug_width = 2
        data = Pid.output_data

        #static error
        #state
        ax = fig.add_subplot(gs[0:2, debug_width:strip_width])
        err_status = ax.text(1.0, SET_POINT - START_LOC, '', fontsize=10, color='g')
        err_plot,  = ax.plot(t[:], SET_POINT - state[:,0], '-g')
        err_line,  = ax.plot([], [], '-r')        
        plt.title('Static Error' + ' - Steady Err ' + str(np.round(np.abs(SET_POINT - state[-1,0]), 3)))
        plt.xticks([0,t_end])
        plt.xlim(0, t_end)
        Y_min = np.min(SET_POINT - state[:,0]) - 1
        Y_max = np.max(SET_POINT - state[:,0]) + 10
        #ax.text(1.0, SET_POINT - START_LOC, 'Min abs(Err) * ' + str(np.min(np.abs(SET_POINT - state[:,0]))), fontsize=10, color='g')
        plt.ylim(Y_min, Y_max)

        #print(len(data[:,0]))
        # P
        ax = fig.add_subplot(gs[3:6, debug_width:strip_width])
        p_plot, = ax.plot(data[:,0], data[:,1], '-k')
        p_line, = ax.plot([], [], '-r')
        plt.title('P Output Acceleration')
        plt.xticks([0,t_end])
        plt.xlim(0, t_end)
        #plt.ylim(-10, 10)

        # I
        ax = fig.add_subplot(gs[7:10, debug_width:strip_width])
        i_plot, = ax.plot(data[:,0], data[:,2], '-k')
        i_line, = ax.plot([], [], '-r')
        plt.title('I Output Acceleration')
        plt.xticks([0,t_end])
        plt.xlim(0, t_end)
        #plt.ylim(-10, 10)

        # D
        ax = fig.add_subplot(gs[11:14, debug_width:strip_width])
        d_plot, = ax.plot(data[:,0], data[:,3], '-k')
        d_line, = ax.plot([], [], '-r')
        plt.title('D Output Acceleration')
        plt.xlabel('Time (s)')
        plt.xlim(0, t_end)

    print("Compute Time: ", round(time.clock() - start, 3), "seconds.")

    # Animation.
    elevator_ani = animation.FuncAnimation(fig, update_plot, frames=range(0,int(t_end*1/dt)), interval=100, repeat = True, blit=True)
    plt.show()

    temp_filename = uuid.uuid1().hex+'.html'
    elevator_ani.save(temp_filename)

        #elevator_ani = animation.FuncAnimation(fig, update_plot, frames=range(0,int(t_end*20)), interval=100, repeat = True, blit=True)
        # elevator_ani = animation.FuncAnimation(fig, update_plot, frames=generator, interval=50, repeat = False, blit=True)
        # temp_filename = uuid.uuid1().hex+'.gif'
        # writergif = animation.PillowWriter(fps=20)
        # elevator_ani.save(temp_filename, writergif)