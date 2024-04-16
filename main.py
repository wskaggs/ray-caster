from src import Camera
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
    sky_width, sky_height, _ = sky.shape

    # Calculate the vertical shear based on the pitch of the camera
    y_shear = -int(buffer_height / 2 * camera.pitch)

    # Draw each column of the buffer individually
    yaw_angle_delta = camera.fov / buffer_width

    for i in range(buffer_width):
        # Calculate the yaw angle for this column
        yaw_prime = camera.yaw - camera.fov / 2 + i * yaw_angle_delta

        # Render the sky first to ensure occlusion with other scene objects
        sky_x = int(np.interp(yaw_prime % (2 * np.pi), (0, 2 * np.pi), (0, sky_width - 1)))
        buffer[i][:] = sky[sky_x][buffer_height - y_shear: 2 * buffer_height - y_shear]

    return buffer


def main() -> None:
    """
    Entry point into the application
    """
    # Create the window and create the buffer to draw to
    window_width, window_height = 1600, 900
    window = pg.display.set_mode((window_width, window_height))
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
    camera = Camera()

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
