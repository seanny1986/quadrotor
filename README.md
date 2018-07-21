
This repository contains code for the following:

1.  A 6DOF quadrotor flight dynamics simulation. The files for this are contained in the "simulation" folder. The main
    simulation uses quaternion rotations with an explicit RK4 integration scheme. In the legacy subfolder, you can also
    find code for a simulation using rotation matrices with a semi-implicit Euler integrator, and a second simulation
    using quaternion rotations and leap-frog integration. These are mainly kept for learning purposes. In general, I've
    aimed for clarity with simulation code -- the code using the semi-implicit Euler scheme is, IMO, a fairly clean and
    straightforward implementation that helps to get the idea across. You can adjust simulation parameters by editing the 
    config.py file. Future updates will include adding stochastic wind disturbance to the vehicle, and --potentially -- 
    multiple vehicles in the same environment.

2.  Environment wrappers for the 6DOF quadrotor flight dynamics simulation, located in the "environments" folder. Our goal 
    is to apply learning algorithms to  solving tasks that human do poorly, or to learn behaviors that can potentially 
    minimize human oversight (i.e. greater autonomy). These wrappers include tasks such as: climbing to altitude and hovering, 
    flying to a randomly generated waypoint, perching on a wall, rapidly descending from altitude without entering a vortex 
    ring state, flying straight and level, and navigating through a 3D box-world to get to a given goal. Future environments 
    are intended to include 3D gathering tasks, seek/avoid, one-on-one pursuit/evasion.

3.  Basic flight controllers, located in the "controllers" folder. At this stage, only a PID hover and waypoint controller
    is included. This was implemented to provide a clear example for other students, and for sanity checking the simulation.
    Would be good to have other controllers implemented, but that's currently a low priority for me. Trajectory planning and
    following would also be very nice to have, along with an optimization routine for the existing PD controller.

4.  DRL policies located in the "policies" folder. This folder contains the actor-critic architectures we wish to apply to
    the tasks set out in the "environments" folder. Current policies include the Cross-Entropy Method, Deep Deterministic
    Policy Gradient, Generalized Advantage Estimation, Proximal Policy Optimization, Q-PROP, TRPO, and my own monster that 
    I'm calling Forward Model Importance Sampling. These algorithms have all been validated and shown to learn on other tasks 
    in OpenAI Gym. Policies located in the "ind" folder use a diagonal covariance matrix when selecting actions -- i.e.
    actions are not correlated with one another. This is not strictly correct, since actions definitely will be correlated
    (consider a climb-to-altitude task for example, where if one motor is producing high thrust, all motors will be).
    Policies located in the "cf" folder output a lower triangular Cholesky factor matrix, where the diagonal is always
    positive. These policies are (in theory) able to capture and learn the covariance over actions for a given task.

5.  DRL trainers located in the "trainers" folder. These trainers implement the actual learning algorithms typically found
    in papers, as opposed to policies, which only contain the network architecture and helper functions. It makes sense
    to keep these algorithms separate from the network architecture since we might, for example, do hyperparameter search
    by spawning multiples of the same network with different hyperparameters to determine the best settings. You can edit
    the experiment settings using the config.py file located in this folder. If you add a new algorithm to this repository,
    you should add a trainer to this folder, and then add the training parameters to the config file in a dictionary.

6.  A main experiment script (example). This script fires up multiple instances of the simulator, along with the desired
    policy search algorithms. 