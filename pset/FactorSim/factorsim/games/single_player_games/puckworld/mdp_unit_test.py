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


class TestPuckworld(unittest.TestCase):
    def setUp(self):
        pygame.init()

    def tearDown(self):
        pygame.quit()

    def test_agent_movement(self):
        self.test_number = 4

        game = Game()
        initial_position = game.state_manager.agent_position.copy()
        # Simulate agent movement
        game.run(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP))
        new_position = game.state_manager.agent_position.copy()
        self.assertNotEqual(initial_position, new_position)

    def test_reward_system(self):
        self.test_number = 5

        game = Game()
        initial_score = game.state_manager.score
        game.state_manager.agent_position = game.state_manager.green_dot_position.copy()
        game.run(pygame.event.Event(pygame.NOEVENT))
        new_score = game.state_manager.score
        green_reward = new_score - initial_score

        game = Game()
        initial_score = game.state_manager.score
        game.state_manager.agent_position = game.state_manager.red_puck_position.copy()
        game.run(pygame.event.Event(pygame.NOEVENT))
        new_score = game.state_manager.score
        red_reward = new_score - initial_score

        self.assertTrue(green_reward > red_reward)

    def test_red_puck_movement(self):
        self.test_number = 3

        game = Game()
        initial_position = game.state_manager.red_puck_position.copy()
        # Simulate red puck movement
        game.run(pygame.event.Event(pygame.NOEVENT))
        new_position = game.state_manager.red_puck_position.copy()
        self.assertNotEqual(initial_position, new_position)

    def test_green_dot_relocation(self):
        self.test_number = 8

        game = Game()
        initial_position = game.state_manager.green_dot_position.copy()
        # Simulate green dot relocation
        game.state_manager.agent_position = game.state_manager.green_dot_position.copy()
        game.run(pygame.event.Event(pygame.NOEVENT))
        new_position = game.state_manager.green_dot_position.copy()
        self.assertNotEqual(initial_position, new_position)


if __name__ == "__main__":
    unittest.main(testRunner=JsonTestRunner(game_name="puckworld", verbosity=2))
