from numba.experimental import jitclass
from numba import double
from .map import Map
import numpy as np

# The compilation types for the members of the Camera class
spec = [
    ("map", Map.class_type.instance_type),
    ("x", double),
    ("y", double),
    ("yaw", double),
    ("pitch", double),
    ("fov", double)
]


@jitclass(spec)
class Camera(object):
    """
    This class represents a camera that is controlled by the user and navigates the map
    """
    def __init__(self, world: Map) -> None:
        """
        Constructor

        :param world: the map that this camera is navigating
        """
        self.map: Map = world  # The map this camera is navigating
        self.x: float = world.start_x  # The x position of the camera in map coordinates
        self.y: float = world.start_y  # The y position of the camera in map coordinates
        self.yaw: float = 0  # The rotation within the xy plane in radians
        self.pitch: float = 0  # The rotation with respect to the xy plane in radians
        self.fov: float = np.pi / 3  # The horizontal field of view
