"""
        Config file for the aircraft. We define the following parameters:
        
        mass = the mass of the vehicle in kg
        prop_radius = the radius of the propellers in meters (this is cosmetic only, no momentum theory)
        l = the length between the centre of mass and the centre of the prop disk (i.e. arm length)
        Jxx = the mass moment of inertia about the x-axis (roll)
        Jyy = the mass moment of inertia about the y-axis (pitch)
        Jzz = the mass moment of inertia about the z-axis (yaw)
        kt = motor thrust coefficient
        kq = motor torque coefficient
        kd1 = linear drag coefficient
        kd2 = angular drag coefficient
        dt = solver time step
    """

params = {"mass":0.65,
        "prop_radius": 0.1,
        "l": 0.23,
        "Jxx": 7.5e-3,
        "Jyy": 7.5e-3,
        "Jzz": 1.3-2,
        "kt": 3.13e-5,
        "kq": 7.5e-7,
        "kd1": 9e-3,
        "kd2": 9e-4,
        "dt": 0.05}