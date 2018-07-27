def make(name):
    if name == "hover":
        import environments.hover as hover
        return hover.Environment()
    if name == "flying_skills":
        import environments.flying_skills as flying_skills
        return flying_skills.Environment()
    if name == "climb_hover":
        import environments.climb_hover as climb_hover
        return climb_hover.Environment()
    if name == "model_training":
        import environments.model_training as model_training
        return model_training.Environment()
    