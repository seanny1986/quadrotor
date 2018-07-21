def make(name):
    if name == "attitude_control":
        import environments.attitude_control as attitude_control
        return attitude_control.Environment()
    if name == "flying_skills":
        import environments.flying_skills as flying_skills
        return flying_skills.Environment()
    if name == "maneuvers":
        import environments.maneuvers as maneuvers
        return maneuvers.Environment()
    if name == "box_world":
        import environments.box_world as box_world
        return box_world.Environment()
    if name == "climb_hover":
        import environments.climb_hover as climb_hover
        return climb_hover.Environment()
    if name == "perching":
        import environments.perching as perching
        return perching.Environment()
    if name == "rapid_descent":
        import environments.rapid_descent as rapid_descent
        return rapid_descent.Environment()
    if name == "straight_and_level":
        import environments.straight_and_level as straight_and_level
        return straight_and_level.Environment()
    if name == "model_training":
        import environments.model_training as model_training
        return model_training.Environment()
    if name == "recovery":
        import environments.recovery as recovery
        return recovery.Environment()
    