from numba.experimental import jitclass
from numba import int64, double
from .ray_hit_info import RayHitInfo, HitSide
import numpy as np

# The compilation types for the Map class
spec = [
    ("size", int64),
    ("cells", int64[:, :]),
    ("colors", double[:, :, :]),
    ("start_x", double),
    ("start_y", double)
]


@jitclass(spec)
class Map(object):
    """
    This class represents the (square) map that the player will walk around
    """
    def __init__(self, size: int) -> None:
        """
        Constructor

        Creates a square map that is randomly generated. Each cell has a 1/3 chance being a walled cell.

        :param size: the width and height of the map to generate
        """
        self.size: int = size
        self.cells: np.array = np.random.choice(np.array([0, 0, 1]), (size, size))  # The cells that make up the map
        self.colors: np.array = np.random.uniform(0, 1, (size, size, 3))  # The corresponding colors for each cell
        self.start_x: float = np.random.randint(1, size - 1) + 0.5  # The starting x position for the player
        self.start_y: float = np.random.randint(1, size - 1) + 0.5  # The starting y position for the player

        # Ensure the border consists of walled cells and the player is not spawned in a walled cell
        self.cells[0, :], self.cells[size - 1, :], self.cells[:, 0], self.cells[:, size - 1] = (1, 1, 1, 1)
        self.cells[int(self.start_x)][int(self.start_y)] = 0

    def ray_cast(self, start_x: float, start_y: float, angle: float) -> RayHitInfo:
        """
        Perform a ray cast in the map and determine information about the intersection

        :param start_x: the x coordinate of the ray's starting point
        :param start_y: the y position of the ray's starting point
        :param angle: the angle to shoot the ray
        :return:
        """
        # Initialize the hit info with the cell we're starting in
        info = RayHitInfo()
        info.map_x = int(start_x)
        info.map_y = int(start_y)

        # Calculate the (normalized) direction of the ray and the step between cells
        ray_dir_x, ray_dir_y = np.cos(angle), np.sin(angle)
        step_x = -1 if ray_dir_x < 0 else 1
        step_y = -1 if ray_dir_y < 0 else 1

        # Calculate the distance the ray travels between x and y grid boundaries
        delta_dist_x = np.inf if np.equal(ray_dir_x, 0) else np.abs(1 / ray_dir_x)
        delta_dist_y = np.inf if np.equal(ray_dir_y, 0) else np.abs(1 / ray_dir_y)

        # Calculate the distance the ray travels until the first grid boundary intersection
        side_dist_x = (start_x - info.map_x if ray_dir_x < 0 else info.map_x + 1 - start_x) * delta_dist_x
        side_dist_y = (start_y - info.map_y if ray_dir_y < 0 else info.map_y + 1 - start_y) * delta_dist_y

        # Perform DDA until we reach a cell that is not empty
        while 0 <= info.map_x < self.size and 0 <= info.map_y < self.size and self.cells[info.map_x][info.map_y] == 0:
            if side_dist_x < side_dist_y:
                side_dist_x += delta_dist_x
                info.map_x += step_x
                info.hit_side = HitSide.LEFT if step_x == -1 else HitSide.RIGHT
            else:
                side_dist_y += delta_dist_y
                info.map_y += step_y
                info.hit_side = HitSide.TOP if step_x == -1 else HitSide.BOTTOM

        # Calculate the distance traveled by the ray and the x and y locations of the hit
        info.distance = side_dist_y - delta_dist_y if info.is_horizontal_side_hit() else side_dist_x - delta_dist_x
        info.hit_x = start_x + info.distance * ray_dir_x
        info.hit_y = start_y + info.distance * ray_dir_y

        return info
