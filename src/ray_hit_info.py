from numba.experimental import jitclass
from numba import double, int64
from enum import IntEnum


class HitSide(IntEnum):
    """
    An enumeration describing which side of a wall was hit
    """
    LEFT = 0
    TOP = 1
    RIGHT = 2
    BOTTOM = 3


# The compilation types for the RayHit
spec = [
    ("distance", double),
    ("hit_x", double),
    ("hit_y", double),
    ("map_x", int64),
    ("map_y", int64),
    ("hit_side", int64)
]


@jitclass(spec)
class RayHitInfo(object):
    """
    Utility class that holds information about a ray cast
    """
    def __init__(self) -> None:
        """
        Constructor
        """
        self.distance: float = 0  # The distance traveled by the ray
        self.hit_x: float = 0  # The x coordinate of the ray intersection
        self.hit_y: float = 0  # The y coordinate of the ray intersection
        self.map_x: int = 0  # The x coordinate of the hit cell
        self.map_y: int = 0  # The y coordinate of the hit cell
        self.hit_side: HitSide = HitSide.LEFT  # The side of the cell that was hit

    def is_vertical_hit(self) -> bool:
        """
        Check if the side hit by the ray was a vertical side

        :return: `True` if the left or right side of a cell were hit, `False` otherwise
        """
        return self.hit_side == HitSide.LEFT or self.hit_side == HitSide.RIGHT

    def is_horizontal_hit(self) -> bool:
        """
        Check if the side hit by the ray was a vertical side

        :return: `True` if the left or right side of a cell were hit, `False` otherwise
        """
        return self.hit_side == HitSide.TOP or self.hit_side == HitSide.BOTTOM
