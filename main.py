import pygame as pg
import numpy as np


def main() -> None:
    """
    Entry point into the application
    """
    # Create the window and set up the timer
    window = pg.display.set_mode((1600, 900))
    clock = pg.time.Clock()

    # Run the game loop
    window_should_close = False

    while not window_should_close:
        # Handle events
        for event in pg.event.get():
            if event.type == pg.QUIT or event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                window_should_close = True

        # Update the scene
        delta_time = clock.tick() / 1000

        # Refresh the screen
        window.fill((0, 0, 0))
        pg.display.flip()


if __name__ == "__main__":
    pg.init()
    main()
    pg.quit()
