import local_flight
import maneuvers
import box_world
import hover

def make(name):
    if name == "local_flightflight":
        return local_flight.Environment()
    if name == "maneuvers":
        return maneuvers.Environment()
    if name == "box_world":
        return box_world.Environment()
    if name == "hover":
        return hover.Environment()