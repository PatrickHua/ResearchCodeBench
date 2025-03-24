import pygame
import sys
import random


class BaseGame:
    def __init__(self, width, height, grid_size, fps, colors):
        self.width = width
        self.height = height

        self.fps = fps
        self.colors = colors

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))

        self.clock = pygame.time.Clock()
        self.game_over = False
        self.win_message = False

    def init_assets(self):
        pass

    def reset_game(self):
        pass

    def handle_events(self):
        pass

    def update_game_state(self):
        pass

    def run(self):
        self.init_assets()
        while True:
            self.handle_events()
            self.update_game_state()
