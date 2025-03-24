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


class TestWaterworld(unittest.TestCase):
    def setUp(self):
        pygame.init()

    def tearDown(self):
        pygame.quit()

    def test_movement(self):
        self.test_number = 1

        game = Game()
        # Test moving right
        initial_position_x = game.state_manager.player_position_x
        initial_position_y = game.state_manager.player_position_y
        event = pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RIGHT})
        game.run(event)
        new_position_x = game.state_manager.player_position_x
        new_position_y = game.state_manager.player_position_y
        self.assertTrue(new_position_x > initial_position_x)
        # self.assertEqual(new_position_y, initial_position_y)

        game = Game()
        # Test moving left
        initial_position_x = game.state_manager.player_position_x
        initial_position_y = game.state_manager.player_position_y
        event = pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_LEFT})
        game.run(event)
        new_position_x = game.state_manager.player_position_x
        new_position_y = game.state_manager.player_position_y
        self.assertTrue(new_position_x < initial_position_x) 
        # self.assertEqual(new_position_y, initial_position_y) 

        game = Game()
        # Test moving up
        initial_position_x = game.state_manager.player_position_x
        initial_position_y = game.state_manager.player_position_y
        event = pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_UP})
        game.run(event)
        new_position_x = game.state_manager.player_position_x
        new_position_y = game.state_manager.player_position_y
        self.assertTrue(new_position_y < initial_position_y)
        # self.assertEqual(new_position_x, initial_position_x)

        game = Game()
        # Test moving down
        initial_position_x = game.state_manager.player_position_x
        initial_position_y = game.state_manager.player_position_y
        event = pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN})
        game.run(event)
        new_position_x = game.state_manager.player_position_x
        new_position_y = game.state_manager.player_position_y
        self.assertTrue(new_position_y > initial_position_y)
        # self.assertEqual(new_position_x, initial_position_x)


    def test_collision(self):
        self.test_number = 7
        game = Game()
        initial_score = game.state_manager.score
        circle = {
            "x": game.state_manager.player_position_x,
            "y": game.state_manager.player_position_y,
        }
        game.state_manager.green_circles.append(circle)
        before_green_circles = game.state_manager.green_circles.copy()
        # Update circle logic to detect collision
        game.run(pygame.event.Event(pygame.NOEVENT))
        self.assertNotEqual(before_green_circles, game.state_manager.green_circles)

        initial_score = game.state_manager.score
        circle = {
            "x": game.state_manager.player_position_x,
            "y": game.state_manager.player_position_y,
        }
        game.state_manager.red_circles.append(circle)
        before_red_circles = game.state_manager.red_circles.copy()
        # Update circle logic to detect collision
        game.run(pygame.event.Event(pygame.NOEVENT))
        self.assertNotEqual(before_red_circles, game.state_manager.red_circles)

    def test_score_update(self):
        self.test_number = 4
        game = Game()
        initial_score = game.state_manager.score
        circle = {
            "x": game.state_manager.player_position_x,
            "y": game.state_manager.player_position_y,
        }
        game.state_manager.green_circles.append(circle)
        # Run circle logic to collect the circle
        game.run(pygame.event.Event(pygame.NOEVENT))
        new_score = game.state_manager.score
        self.assertTrue(
            new_score > initial_score
        )  # Check if score is incremented for collecting green circle

        game = Game()
        initial_score = game.state_manager.score
        circle = {
            "x": game.state_manager.player_position_x,
            "y": game.state_manager.player_position_y,
        }
        game.state_manager.red_circles.append(circle)
        # Run circle logic to collect the circle
        game.run(pygame.event.Event(pygame.NOEVENT))
        new_score = game.state_manager.score
        self.assertTrue(
            new_score < initial_score
        )  # Check if score is incremented for collecting green circle


if __name__ == "__main__":
    unittest.main(testRunner=JsonTestRunner(game_name="waterworld", verbosity=2))