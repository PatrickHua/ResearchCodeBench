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


class TestPong(unittest.TestCase):
    def setUp(self):
        pygame.init()

    def tearDown(self):
        pygame.quit()

    def test_paddle_cpu_movement(self):
        self.test_number = 4

        game = Game()
        initial_pos = game.state_manager.cpu_paddle["y"]
        game.state_manager.ball["y"] = initial_pos + 50
        game.run(event=pygame.event.Event(pygame.NOEVENT))  # Run game to update score
        self.assertTrue(game.state_manager.cpu_paddle["y"] > initial_pos)

    def test_ball_collision_with_paddle(self):
        self.test_number = 6

        game = Game()
        game.state_manager.ball["x"] = game.state_manager.player_paddle["x"] #+ game.state_manager.paddle_width
        game.state_manager.ball["y"] = game.state_manager.player_paddle["y"] #+ game.state_manager.paddle_height // 2
        game.state_manager.ball["speed_x"] += 1
        original_speed = (game.state_manager.ball["speed_x"], game.state_manager.ball["speed_y"])

        game.run(event=pygame.event.Event(pygame.NOEVENT)) 
        game.run(event=pygame.event.Event(pygame.NOEVENT)) 
        game.run(event=pygame.event.Event(pygame.NOEVENT)) 
        new_speed = (game.state_manager.ball["speed_x"], game.state_manager.ball["speed_y"])
        self.assertNotEqual(original_speed, new_speed)

        game = Game()
        game.state_manager.ball["x"] = game.state_manager.cpu_paddle["x"] #+ game.state_manager.paddle_width
        game.state_manager.ball["y"] = game.state_manager.cpu_paddle["y"] #+ game.state_manager.paddle_height // 2
        game.state_manager.ball["speed_x"] += 1
        original_speed = (game.state_manager.ball["speed_x"], game.state_manager.ball["speed_y"])

        game.run(event=pygame.event.Event(pygame.NOEVENT)) 
        game.run(event=pygame.event.Event(pygame.NOEVENT)) 
        game.run(event=pygame.event.Event(pygame.NOEVENT)) 
        new_speed = (game.state_manager.ball["speed_x"], game.state_manager.ball["speed_y"])
        self.assertNotEqual(original_speed, new_speed)

    def test_score_increment(self):
        self.test_number = 7

        game = Game()
        ball = game.state_manager.ball
        ball["x"] = -10  # Move ball out of bounds on the left side
        game.state_manager.score = 0
        game.run(event=pygame.event.Event(pygame.NOEVENT))  # Run game to update score
        # self.assertTrue(game.state_manager.cpu_score > 0)  # CPU score should increment

        game = Game()
        ball = game.state_manager.ball
        ball["x"] = 1e5 # Move ball out of bounds on the left side
        game.state_manager.cpu_score = 0
        game.run(event=pygame.event.Event(pygame.NOEVENT))  # Run game to update score
        self.assertTrue(game.state_manager.score > 0)  # CPU score should increment

    def test_game_over_condition(self):
        self.test_number = 10

        game = Game()
        game.state_manager.score = 5
        game.run(event=pygame.event.Event(pygame.NOEVENT))  # Run game to update score
        self.assertTrue(game.state_manager.game_over)


if __name__ == "__main__":
    unittest.main(testRunner=JsonTestRunner(game_name="pong", verbosity=2))