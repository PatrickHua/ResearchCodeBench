import importlib
import unittest
import pygame
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
target_path = os.path.join(script_dir, "../../../")
sys.path.append(target_path)
from unittest_utils import JsonTestRunner

os.environ["SDL_VIDEODRIVER"] = "dummy"


if len(sys.argv) > 1:
    # Pop the last argument from sys.argv, which is the path to the game implementation
    implementation_path = sys.argv.pop()
    spec = importlib.util.spec_from_file_location("", implementation_path)
    game_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(game_module)

    Game = game_module.Game
else:
    from mdp import Game


class TestPixelcopter(unittest.TestCase):
    def setUp(self):
        pygame.init()

    def tearDown(self):
        pygame.quit()

    def test_gravity(self):
        self.test_number = 2
        game = Game()
        initial_y = game.state_manager.copter_position["y"]
        game.run(pygame.event.Event(pygame.NOEVENT))  # Update the game state
        game.run(pygame.event.Event(pygame.NOEVENT))  # Update the game state
        game.run(pygame.event.Event(pygame.NOEVENT))  # Update the game state
        self.assertGreater(game.state_manager.copter_position["y"], initial_y)

    def test_pixelcopter_jump(self):
        self.test_number = 3

        game = Game()
        game.run(pygame.event.Event(pygame.NOEVENT))  # Update the game state
        y_without_jump = game.state_manager.copter_position["y"]

        game = Game()
        game.run(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))  # Simulate jump event
        y_with_jump = game.state_manager.copter_position["y"]
        self.assertTrue(y_with_jump < y_without_jump)

    def test_obstacle_spawn(self):
        self.test_number = 4
        game = Game()
        game.state_manager.obstacle_positions = []  # Clear obstacles
        game.run(pygame.event.Event(pygame.NOEVENT))  # Run game to spawn obstacles
        self.assertTrue(len(game.state_manager.obstacle_positions) > 0)

    def test_collision_with_obstacle(self):
        self.test_number = 5

        game = Game()
        game.state_manager.copter_position["y"] = 0  # Place copter at the top
        game.run(event=pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))  # Trigger jump to collide
        self.assertTrue(game.state_manager.game_over)

        game = Game()
        game.state_manager.obstacle_positions = [
            {"x": 100, "gap_y": 200, "counted": False}
        ]  # Place obstacle
        game.state_manager.copter_position["y"] = 200 - 10  # Place copter at the top
        game.run(pygame.event.Event(pygame.NOEVENT))  # Trigger collision
        self.assertTrue(game.state_manager.game_over)


if __name__ == "__main__":
    unittest.main(testRunner=JsonTestRunner(game_name='pixelcopter', verbosity=2))
