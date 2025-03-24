import json
import importlib
import unittest
import pygame
import sys
import os
from unittest.mock import MagicMock

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


class TestBreakoutGame(unittest.TestCase):
    def setUp(self):
        pygame.init()

    def tearDown(self) -> None:
        pygame.quit()

    def test_paddle_move(self):
        self.test_number = 1

        game = Game()

        game.state_manager.paddle["x"] = 300
        game.run(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_LEFT}))
        self.assertEqual(game.state_manager.paddle["x"], 260)

        game.run(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RIGHT}))
        self.assertEqual(game.state_manager.paddle["x"], 300)

    def test_ball_update(self):
        self.test_number = 3

        game = Game()

        initial_y = game.state_manager.ball["y"]
        initial_dy = game.state_manager.ball["velocity"]["y"]

        game.run(pygame.event.Event(pygame.NOEVENT))

        self.assertEqual(game.state_manager.ball["y"], initial_y + initial_dy)

    def test_brick_init(self):
        self.test_number = 4

        game = Game()

        brick = game.state_manager.bricks[0]
        self.assertEqual(brick["x"], 0)
        self.assertEqual(brick["y"], 0)

    def test_ball_init(self):
        self.test_number = 2

        game = Game()

        ball = game.state_manager.ball
        self.assertEqual(ball["x"], 300)
        self.assertEqual(ball["y"], 400)

    def test_update_collide(self):
        self.test_number = 5

        game = Game()

        game.state_manager.ball["velocity"]["y"] = 1
        game.state_manager.ball["x"] = 0
        game.state_manager.ball["y"] = 0
        game.state_manager.bricks[0]["x"] = 0
        game.state_manager.bricks[0]["y"] = 0
        game.run(pygame.event.Event(pygame.NOEVENT))
        self.assertEqual(game.state_manager.ball["velocity"]["y"], -1)

    def test_game_over(self):
        self.test_number = 8

        game = Game()

        game.state_manager.ball["y"] = 600
        self.assertTrue(game.run(pygame.event.Event(pygame.NOEVENT)))
        game.state_manager.ball["y"] = 800
        self.assertFalse(game.run(pygame.event.Event(pygame.NOEVENT)))


if __name__ == "__main__":
    unittest.main(testRunner=JsonTestRunner(game_name="breakout", verbosity=2))