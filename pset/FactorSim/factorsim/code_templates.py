PREPEND_CODE = """
import os
#os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame
import sys
import random
import numpy as np
import math


"""

APPEND_CODE = """
class Game():
    def __init__(self):
        self.state_manager = StateManager()
        self.state_manager.screen = pygame.display.set_mode((self.state_manager.SCREEN_WIDTH,
                                                             self.state_manager.SCREEN_HEIGHT))

    def run(self, event):
        state_manager = self.state_manager
        if event.type == pygame.QUIT:
            return False
{input_logic_code}
        # call all the logics
{logic_code}
        # Fill the screen with white
        state_manager.screen.fill((255, 255, 255))
{render_code}
        return not state_manager.game_over

if __name__ == "__main__":
    game = Game()
    pygame.init()
    global event
    running = True
    while running:
        event = pygame.event.poll()
        running = game.run(event)
        pygame.display.flip()
    pygame.quit()
"""
