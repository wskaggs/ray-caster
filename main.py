from src import Camera, Map
from numba import njit
import pygame as pg
import numpy as np


@njit
def refresh(buffer: np.array, camera: Camera, sky: np.array) -> np.array:
    """
    Draw the next frame

    :param buffer: the screen buffer to draw to
    :param camera: the camera we are viewing through
    :param sky: the normalized pixels of the sky texture
    :return: the new screen buffer
    """
    # Clear the buffer with black
    buffer.fill(0)

    # Extract out the dimensions of the buffer and textures
    buffer_width, buffer_height, _ = buffer.shape
    half_buffer_height = buffer_height // 2
    sky_width, sky_height, _ = sky.shape

    # Calculate the vertical shear based on the pitch of the camera
    y_shear = -int(half_buffer_height * camera.pitch)

    # Draw each column of the buffer individually
    yaw_angle_delta = camera.fov / buffer_width

    for i in range(buffer_width):
        # Calculate the yaw angle for this column
        yaw_prime = camera.yaw - camera.fov / 2 + i * yaw_angle_delta

        # Render the sky first to ensure occlusion with other scene objects
        sky_x = int(np.interp(yaw_prime % (2 * np.pi), (0, 2 * np.pi), (0, sky_width - 1)))
        buffer[i][:] = sky[sky_x][buffer_height - y_shear: 2 * buffer_height - y_shear]

        # Perform the ray cast
        hit_info = camera.map.ray_cast(camera.x, camera.y, yaw_prime)

        # Determine the height of the corresponding coordinates on screen
        wall_draw_height = int(half_buffer_height / hit_info.distance)
        wall_start = (buffer_height - wall_draw_height) // 2 + y_shear
        wall_end = half_buffer_height + wall_draw_height + y_shear

        # The above positions may be off-screen, we need to make sure we don't index out-of-bounds
        wall_draw_start = max(0, wall_start)
        wall_draw_end = min(wall_end, buffer_height - 1)

        # Draw the wall
        buffer[i][wall_draw_start: wall_draw_end] = np.array([1, 0, 0])

        # Draw the floor
        if wall_draw_end != buffer_height - 1:
            buffer[i][wall_draw_end: buffer_height] = np.array([0, 0, 0])

    return buffer


def main() -> None:
    """
    Entry point into the application
    """
    # Create the window, create the clock, and create the buffer to draw to
    window_width, window_height = 1600, 900
    window = pg.display.set_mode((window_width, window_height))
    clock = pg.time.Clock()
    buffer = np.zeros((window_height // 4, window_width // 4, 3))

    # Lock the cursor on screen
    pg.mouse.set_visible(False)
    pg.event.set_grab(True)

    # Load textures
    sky_texture = pg.transform.smoothscale(pg.image.load("images/sky.png"), (window_width, 2 * window_height))

    # Convert the textures into their normalized pixel values
    sky = pg.surfarray.array3d(sky_texture) / 255

    # Run the game loop
    window_should_close = False
    world = Map(25)
    camera = Camera(world)

    while not window_should_close:
        # Handle events
        for event in pg.event.get():
            if event.type == pg.QUIT or event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                window_should_close = True

        # Update the yaw and pitch of the camera based on the mouse movement
        if pg.mouse.get_focused():
            mouse_delta_x, mouse_delta_y = pg.mouse.get_rel()
            camera.yaw += mouse_delta_x / 500
            camera.pitch = np.clip(camera.pitch + mouse_delta_y / 500, -np.pi / 4, np.pi / 4)

        # Determine the velocity (without rotation) based on the movement controls
        pressed_keys = pg.key.get_pressed()
        vel_x = 1 if pressed_keys[pg.K_w] else -1 if pressed_keys[pg.K_s] else 0
        vel_y = 1 if pressed_keys[pg.K_d] else -1 if pressed_keys[pg.K_a] else 0

        # Normalize the velocity if required
        vel_mag_squared = vel_x ** 2 + vel_y ** 2

        if not np.equal(vel_mag_squared, 0):
            vel_mag = np.sqrt(vel_mag_squared)
            vel_x /= vel_mag
            vel_y /= vel_mag

        # Rotate the velocity based on the camera yaw
        yaw = camera.yaw
        vel_x, vel_y = vel_x * np.cos(yaw) - vel_y * np.sin(yaw), vel_x * np.sin(yaw) + vel_y * np.cos(yaw)

        # Determine the new position based on the velocity and time
        delta_time = clock.tick() / 1000
        new_x = camera.x + vel_x * delta_time
        new_y = camera.y + vel_y * delta_time

        # Update the camera position while preventing any clipping with walls
        if world.cells[int(new_x)][int(camera.y)] == 0:
            camera.x = new_x
        if world.cells[int(camera.x)][int(new_y)] == 0:
            camera.y = new_y

        # Refresh the buffer and create a surface from it
        buffer = refresh(buffer, camera, sky)
        surface = pg.transform.smoothscale(pg.surfarray.make_surface(buffer * 255), (window_width, window_height))

        # Update the screen
        window.blit(surface, (0, 0))
        pg.display.flip()


if __name__ == "__main__":
    pg.init()
    main()
    pg.quit()
