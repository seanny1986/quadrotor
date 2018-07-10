def make(name):
    if name == "flying_skills":
        import environments.flying_skills as flying_skills
        return flying_skills.Environment()
    if name == "maneuvers":
        import environments.maneuvers as maneuvers
        return maneuvers.Environment()
    if name == "box_world":
        import environments.box_world as box_world
        return box_world.Environment()
    if name == "hover":
        import environments.hover as hover
        return hover.Environment()
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