from numba.experimental import jitclass
from numba import float32
import numpy as np
import pygame as pg


@jitclass([("pos_x", float32), ("pos_y", float32), ("yaw", float32), ("pitch", float32), ("fov", float32)])
class Camera(object):
    """
    This class represents a camera that is controlled by the user and navigates the map
    """
    def __init__(self) -> None:
        """
        Constructor
        """
        self.pos_x: float = 0  # The x position of the camera in map coordinates
        self.pos_y: float = 0  # The y position of the camera in map coordinates
        self.yaw: float = 0  # The rotation within the xy plane in radians
        self.pitch: float = 0  # The rotation with respect to the xy plane in radians
        self.fov: float = np.pi / 3  # The horizontal field of view
