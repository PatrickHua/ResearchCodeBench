import json
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
    print(implementation_path)
    spec = importlib.util.spec_from_file_location("", implementation_path)
    game_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(game_module)

    Game = game_module.Game
else:
    from mdp import Game


class TestSnake(unittest.TestCase):
    def setUp(self):
        pygame.init()

    def tearDown(self):
        pygame.quit()

    def test_snake_direction_change(self):
        self.test_number = 2

        for initial_direction in ["RIGHT", "LEFT", "UP", "DOWN"]:
            for new_direction in ["RIGHT", "LEFT", "UP", "DOWN"]:
                if initial_direction == new_direction:
                    continue

                game = Game()
                state_manager = game.state_manager
                state_manager.snake_direction = initial_direction
                event = pygame.event.Event(
                    pygame.KEYDOWN, {"key": getattr(pygame, f"K_{new_direction}")}
                )
                game.run(event)

                if initial_direction in ["RIGHT", "LEFT"]:
                    if new_direction in ["UP", "DOWN"]:
                        self.assertEqual(state_manager.snake_direction, new_direction)
                    else:
                        self.assertEqual(
                            state_manager.snake_direction, initial_direction
                        )
                if initial_direction in ["UP", "DOWN"]:
                    if new_direction in ["LEFT", "RIGHT"]:
                        self.assertEqual(state_manager.snake_direction, new_direction)
                    else:
                        self.assertEqual(
                            state_manager.snake_direction, initial_direction
                        )

    def test_snake_growth(self):
        self.test_number = 6

        game = Game()
        state_manager = game.state_manager

        initial_length = len(state_manager.snake_segments)
        # place it 1 unit away from the food
        state_manager.snake_segments = [{"x": 20, "y": 20}]
        state_manager.food_position = {"x": 20, "y": 20}
        game.run(pygame.event.Event(pygame.NOEVENT))
        game.run(pygame.event.Event(pygame.NOEVENT))
        self.assertTrue(len(state_manager.snake_segments) > initial_length)


    def test_snake_collision(self):
        self.test_number = 5

        game = Game()
        state_manager = game.state_manager

        state_manager.snake_segments = [{"x": 0, "y": state_manager.SCREEN_HEIGHT // 2}]
        state_manager.snake_direction = "LEFT"
        game.run(pygame.event.Event(pygame.NOEVENT))
        self.assertTrue(state_manager.game_over)

        state_manager.snake_segments = [
            {"x": state_manager.SCREEN_WIDTH, "y": state_manager.SCREEN_HEIGHT // 2}
        ]
        state_manager.snake_direction = "RIGHT"
        game.run(pygame.event.Event(pygame.NOEVENT))
        self.assertTrue(state_manager.game_over)

        state_manager.snake_segments = [{"x": state_manager.SCREEN_WIDTH // 2, "y": -1}]
        state_manager.snake_direction = "UP"
        game.run(pygame.event.Event(pygame.NOEVENT))
        self.assertTrue(state_manager.game_over)

        state_manager.snake_segments = [
            {"x": state_manager.SCREEN_WIDTH // 2, "y": state_manager.SCREEN_HEIGHT + 1}
        ]
        state_manager.snake_direction = "DOWN"
        game.run(pygame.event.Event(pygame.NOEVENT))
        self.assertTrue(state_manager.game_over)


if __name__ == "__main__":
    unittest.main(testRunner=JsonTestRunner(game_name="snake", verbosity=2))