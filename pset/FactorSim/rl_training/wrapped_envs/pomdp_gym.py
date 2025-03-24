import random
import gymnasium as gym
import numpy as np
import pygame


class PygameEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, config=None):
        super(PygameEnv, self).__init__()
        self.config = config
        self.env_name = config["name"]
        self.normalize = True if "normalize" not in config.keys() else config["normalize"]
        self.max_episode_steps = config["max_episode_steps"]
        self.env_list = config["env_list"]

        self.env_id = None if "env_id" not in config.keys() else config["env_id"]
        self.game_class = None if "game_class" not in config.keys() else config["game_class"]

        if self.env_name == "puckworld" or self.env_name == "waterworld":
            self.action_space = gym.spaces.Discrete(n=5)
        elif self.env_name == "flappy_bird" or self.env_name == "pixelcopter":
            self.action_space = gym.spaces.Discrete(n=2)
        elif self.env_name == "catcher" or self.env_name == "pong":
            self.action_space = gym.spaces.Discrete(n=3)
        elif self.env_name == "snake":
            self.action_space = gym.spaces.Discrete(n=5)
        
        pygame.init()
        self.my_reset()
        # Define action and observation space, They must be gym.spaces objects
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(self.get_state().shape[-1],), dtype=np.float64)

    def my_reset(self):
        # Initialize your game class
        pygame.quit()
        pygame.init()
        if self.game_class is None:
            self.env_id, self.game_class = random.choice(self.env_list)
            
        self.game = self.game_class()
        self.state_manager = self.game.state_manager
        self.previous_score = 0
        self.current_step = 0

    def get_state(self):
        state_manager = self.state_manager
        if self.env_name == "flappy_bird":
            if len(state_manager.next_pipe_position) == 0:
                state_manager.next_pipe_position = [{"x": 0, "bottom_pipe_length": 0}]
            
            pipes = []
            for p in state_manager.next_pipe_position:
                if p["x"] > state_manager.bird_position_x:
                    pipes.append((p, p.x + p.width/2 - self.player.pos_x ))

            pipes.sort(key=lambda p: p[1])
            if len(pipes) == 0:
                pipes = [{"x": 0, "bottom_pipe_length": 0}]

            observation = np.array(
                [
                    (state_manager.bird_position_y + state_manager.bird_height/2) / state_manager.SCREEN_HEIGHT,
                    state_manager.bird_velocity_y / state_manager.SCREEN_HEIGHT,
                    float(state_manager.next_pipe_position[0]["x"] - state_manager.bird_position_x)/state_manager.SCREEN_WIDTH,
                    float(state_manager.SCREEN_HEIGHT - state_manager.next_pipe_position[0]["bottom_pipe_length"])/state_manager.SCREEN_HEIGHT,
                ]
            )

        elif self.env_name == "catcher":
            observation = np.array(
                [
                    float(state_manager.catcher_position_x) / state_manager.SCREEN_WIDTH,
                    float(state_manager.ball_x) / state_manager.SCREEN_WIDTH,
                    float(state_manager.ball_y) / state_manager.SCREEN_HEIGHT,
                ]
            )

        elif self.env_name == "puckworld":
            observation = np.array(
                [
                    float(state_manager.agent_position["x"]    ) / state_manager.SCREEN_WIDTH,
                    float(state_manager.agent_position["y"]    ) / state_manager.SCREEN_HEIGHT,
                    float(state_manager.green_dot_position["x"]) / state_manager.SCREEN_WIDTH,
                    float(state_manager.green_dot_position["y"]) / state_manager.SCREEN_HEIGHT,
                    float(state_manager.red_puck_position["x"])  / state_manager.SCREEN_WIDTH,
                    float(state_manager.red_puck_position["y"])  / state_manager.SCREEN_HEIGHT,
                ]
            )

        elif self.env_name == "pong":
            observation = np.array(
                [
                    float(state_manager.player_paddle_y + state_manager.player_paddle_height/2)/state_manager.SCREEN_HEIGHT,
                    float(state_manager.ball["y"])/state_manager.SCREEN_HEIGHT,
                ]
            )
        elif self.env_name == "pixelcopter":
            min_dist = 999
            min_block = None
            for b in state_manager.cavern_obstacle_positions:  # Groups do not return in order
                dist_to = b["x"] - state_manager.copter_position_x
                if dist_to >= 0 and dist_to < min_dist:
                    min_block = b
                    min_dist = dist_to

            if min_block is None:
                min_block = {"ceiling_obstacle_length": 0, "floor_obstacle_length": 0}

            observation = np.array(
                [
                    float(state_manager.copter_position_y) / state_manager.SCREEN_HEIGHT,
                    float(state_manager.copter_velocity_y),
                    abs(state_manager.copter_position_y - min_block["ceiling_obstacle_length"]) / state_manager.SCREEN_HEIGHT,
                    abs(state_manager.SCREEN_HEIGHT - min_block["floor_obstacle_length"] - state_manager.copter_position_y) / state_manager.SCREEN_HEIGHT,
                ]
            )

        elif self.env_name == "snake":
            observation = np.array(
                [
                    float(state_manager.snake_head_x)/state_manager.SCREEN_WIDTH,
                    float(state_manager.snake_head_y)/state_manager.SCREEN_HEIGHT,
                    float(state_manager.food_position_x)/state_manager.SCREEN_WIDTH,
                    float(state_manager.food_position_y)/state_manager.SCREEN_HEIGHT,
                ]
            )
        
        elif self.env_name == "waterworld":
            good_creep_pos = bad_creep_pos = [0.]*6
            assert len(state_manager.green_circles) <= 3
            assert len(state_manager.red_circles) <= 3
            for idx, pos in enumerate(state_manager.green_circles):
                good_creep_pos[idx*2] = float(pos["x"])/state_manager.SCREEN_WIDTH
                good_creep_pos[idx*2+1] = float(pos["y"])/state_manager.SCREEN_HEIGHT
            for idx, pos in enumerate(state_manager.red_circles):
                bad_creep_pos[idx*2] = float(pos["x"])/state_manager.SCREEN_WIDTH
                bad_creep_pos[idx*2+1] = float(pos["y"])/state_manager.SCREEN_HEIGHT
            observation = np.array([
                    float(state_manager.player_position_x)/state_manager.SCREEN_WIDTH,
                    float(state_manager.player_position_y)/state_manager.SCREEN_HEIGHT,
                ] + good_creep_pos + bad_creep_pos
            )
            assert observation.shape[0] == 2 + len(good_creep_pos) + len(bad_creep_pos)

        else:
            assert False
            
        return observation

    def perform_action(self, action):
        if self.env_name == "puckworld" or self.env_name == "waterworld" or self.env_name == "snake":
            # actions = ["up", "left", "right", "down", do nothing]
            if action == 0:
                return pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_UP})
            elif action == 1:  
                return pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_LEFT})
            elif action == 2:
                return pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RIGHT})
            elif action == 3:
                return pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN})
            else:  # Do nothing
                return pygame.event.Event(pygame.NOEVENT)
        
        if self.env_name == "pong":
            # actions = ["up", "down", do nothing]
            if action == 0:
                return pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_UP})
            elif action == 1:
                return pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_DOWN})
            else:  # Do nothing
                return pygame.event.Event(pygame.NOEVENT)
        
        if self.env_name == "catcher":
            if action == 0:  # Left
                return pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_LEFT})
            elif action == 1:  # Right
                return pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RIGHT})
            else:  # Do nothing
                return pygame.event.Event(pygame.NOEVENT)
        
        if self.env_name == "flappy_bird" or self.env_name == "pixelcopter":
            # Implement action handling here
            if action == 0:
                return pygame.event.Event(pygame.MOUSEBUTTONDOWN)
            else:
                return pygame.event.Event(pygame.NOEVENT)


    def step(self, action):
        self.current_step += 1
        info = {"has_exception": False}

        #old_lives = self.game.state_manager.lives
        event = self.perform_action(action)
        try:
            running = self.game.run(event)
        except Exception as e:
            print(e)
            info["has_exception"] = True
            running = False

        observation = self.get_state()
        reward = self.game.state_manager.score - self.previous_score
        self.previous_score = self.game.state_manager.score
        if running:
            if self.env_name == "flappy_bird":
                reward += 0.01
            elif self.env_name == "pong":
                reward += 0.01
            elif self.env_name == "puckworld":
                reward = float(reward) / 100

        terminated = self.current_step > self.max_episode_steps or not running
        truncated = False

        pygame.display.flip()
        return observation, reward, terminated, truncated, info


    def reset(self, *, seed=None, options=None):
        self.my_reset()
        observation = self.get_state()
        info = {}
        return observation, info


if __name__ == "__main__":
    env = PygameEnv()
    observation, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = env.action_space.sample()
        observation, reward, done, truncated, info = env.step(action)
        print(observation)
        total_reward += reward
    print(f"Total reward: {total_reward}")
    pygame.quit()
    env.close()
