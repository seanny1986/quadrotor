import local_flight
import maneuvers
import box_world
import hover
import perching
import rapid_descent

def make(name):
    if name == "local_flight":
        return local_flight.Environment()
    if name == "maneuvers":
        return maneuvers.Environment()
    if name == "box_world":
        return box_world.Environment()
    if name == "hover":
        return hover.Environment()
    if name == "perching":
        return perching.Environment()
    if name == "rapid_descent":
        return rapid_descent.Environment()