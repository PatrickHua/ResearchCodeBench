import random
import gymnasium as gym
import numpy as np
import pygame


class PygameOOPEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, config=None):
        super(PygameOOPEnv, self).__init__()
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
            
        # print(self.config["idx"], self.env_id)
        self.game = self.game_class()
        self.previous_score = 0
        self.current_step = 0

    def get_state(self):
        SCREEN_WIDTH, SCREEN_HEIGHT= self.game.screen.get_size()

        if self.env_name == "catcher":
            catcher = self.game.catcher
            ball = self.game.ball
            observation = np.array(
                [
                    float(catcher.rect.centerx) / SCREEN_WIDTH,
                    float(ball.rect.centerx) / SCREEN_WIDTH,
                    float(ball.rect.centery) / SCREEN_HEIGHT,
                ]
            )

        elif self.env_name == "flappy_bird":
            bird = self.game.bird
            pipes = self.game.pipes
            # select the pipe with the smallest rect.centerx value
            closest_pipes = []
            smallest_x = 1e6
            for pipe in pipes:
                if pipe.rect.centerx < smallest_x:
                    smallest_x = pipe.rect.centerx
                    closest_pipes = [pipe]
                elif pipe.rect.centerx == smallest_x:
                    closest_pipes.append(pipe)
            
            if len(pipes) >= 2:
                top_pipe = closest_pipes[0]# if closest_pipes[0].rect.centery < closest_pipes[1].rect.centery else closest_pipes[1]
                bottom_pipe = closest_pipes[0]# if closest_pipes[0].rect.centery > closest_pipes[1].rect.centery else closest_pipes[1]
            elif len(pipes) == 1:
                if closest_pipes[0].rect.topleft[1] == 0:
                    top_pipe = closest_pipes[0]
                    bottom_pipe = None
                else:
                    bottom_pipe = closest_pipes[0]
                    top_pipe = None
            else: 
                top_pipe = bottom_pipe = None

            observation = np.array(
                [
                    bird.rect.centery / SCREEN_HEIGHT,
                    bird.vel / SCREEN_HEIGHT,
                    0 if len(pipes) < 1 else float(smallest_x - bird.rect.centerx)/SCREEN_WIDTH,
                    0 if bottom_pipe is None else float(bottom_pipe.rect.topleft[1]) / SCREEN_HEIGHT,
                ]
            )


        elif self.env_name == "puckworld":
            agent = self.game.agent
            green_dot = self.game.green_dot
            red_puck = self.game.red_puck
            observation = np.array(
                [
                    float(agent.rect.centerx) / SCREEN_WIDTH,
                    float(agent.rect.centery) / SCREEN_HEIGHT,
                    float(green_dot.rect.centerx) / SCREEN_WIDTH,
                    float(green_dot.rect.centery) / SCREEN_HEIGHT,
                    float(red_puck.rect.centerx) / SCREEN_WIDTH,
                    float(red_puck.rect.centery) / SCREEN_HEIGHT,
                ]
            )

        elif self.env_name == "pong":
            paddle_human = self.game.paddle_human
            ball = self.game.ball
            observation = np.array(
                [
                    float(paddle_human.rect.centery) / SCREEN_HEIGHT,
                    float(ball.rect.centery) / SCREEN_HEIGHT,
                ]
            )
        elif self.env_name == "pixelcopter":
            copter = self.game.player
            obstacles = self.game.obstacles

            closest_top_y = copter.rect.centery
            closest_bottom_y = SCREEN_HEIGHT - copter.rect.centery

            smallest_x = 1e6
            for obstacle in obstacles:
                if obstacle.rect_top.centerx >= copter.rect.centerx:
                    dist = obstacle.rect_top.centerx - copter.rect.centerx
                    if dist < smallest_x:
                        smallest_x = dist

                        closest_top_y = abs(copter.rect.centery - obstacle.rect_top.rect.bottomleft[1])
                        closest_bottom_y = abs(obstacle.rect_bottom.topleft[1] - copter.rect.centery)
            
            observation = np.array(
                [
                    float(copter.rect.centery) / SCREEN_HEIGHT,
                    float(copter.momentum) / SCREEN_HEIGHT,
                    closest_top_y / SCREEN_HEIGHT,
                    closest_bottom_y / SCREEN_HEIGHT,
                ]
            )

        elif self.env_name == "snake":
            snake = self.game.snake
            food = self.game.food

            observation = np.array(
                [
                    float(snake.head[0])/SCREEN_WIDTH,
                    float(snake.head[1])/SCREEN_HEIGHT,
                    float(food.rect.topleft[0])/SCREEN_WIDTH,
                    float(food.rect.topleft[1])/SCREEN_HEIGHT,
                ]
            )
        
        elif self.env_name == "waterworld":
            agent = self.game.agent
            green_circles = self.game.green_circles
            red_circles = self.game.red_circles
            good_creep_pos = bad_creep_pos = [0.]*6
            #assert len(green_circles) <= 3
            #assert len(red_circles) <= 3
            for idx, spirte in enumerate(red_circles):
                good_creep_pos[idx*2] = float(spirte.rect.centerx)   / SCREEN_WIDTH
                good_creep_pos[idx*2+1] = float(spirte.rect.centery) / SCREEN_HEIGHT
                if idx > 5:
                    break
            for idx, spirte in enumerate(green_circles):
                bad_creep_pos[idx*2] = float(spirte.rect.centerx)    /SCREEN_WIDTH
                bad_creep_pos[idx*2+1] = float(spirte.rect.centery)  /SCREEN_HEIGHT
                if idx > 5:
                    break

            observation = np.array([
                    float(agent.rect.centerx)/SCREEN_WIDTH,
                    float(agent.rect.centery)/SCREEN_HEIGHT,
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

        if hasattr(self.game, "state_manager"):
            reward = self.game.state_manager.score - self.previous_score
            self.previous_score = self.game.state_manager.score
        else:
            reward = self.game.score - self.previous_score
            self.previous_score = self.game.score

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
