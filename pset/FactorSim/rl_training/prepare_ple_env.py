import sys
import gymnasium as gym
import numpy as np
from ple import PLE


class PLEPygameEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, config):
        super(PLEPygameEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces object
        self.env_name = config["name"]
        self.max_episode_steps = 10000
        if self.env_name == "flappy_bird":
            from ple.games.flappybird import FlappyBird
            self.game = FlappyBird()
        elif self.env_name == "catcher":
            from ple.games.catcher import Catcher
            self.game = Catcher()
        elif self.env_name == "pixelcopter":
            from ple.games.pixelcopter import Pixelcopter
            self.game = Pixelcopter()
        elif self.env_name == "puckworld":
            from ple.games.puckworld import PuckWorld
            self.game = PuckWorld()
        elif self.env_name == "snake":
            from ple.games.snake import Snake
            self.game = Snake()
        elif self.env_name == "waterworld":
            from ple.games.waterworld import WaterWorld
            self.game = WaterWorld()
            self.max_episode_steps = 100
        elif self.env_name == "pong":
            from ple.games.pong import Pong
            self.game = Pong(MAX_SCORE=1)
        elif self.env_name == "monster_kong":
            from ple.games.monsterkong import MonsterKong
            self.game = MonsterKong()

        else:
            assert False

        self.game.rng = np.random.RandomState(np.random.randint(0, 50))
        self.p = PLE(self.game, display_screen=True)
        self.p.init()
        self.all_possible_actions = self.p.getActionSet()
        self.action_space = gym.spaces.Discrete(n=len(self.all_possible_actions))  
        if self.env_name == "monster_kong":
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(0,), dtype=np.float64)
        else:
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=self.get_state().shape, dtype=np.float64)
    

    def get_state(self):
        if self.env_name == "monster_kong":
            return None
        observation_dict = self.p.getGameState()
        if self.env_name == "flappy_bird":
            observation = np.array([
                float(observation_dict["player_y"])/self.game.height, 
                float(observation_dict["player_vel"]), #/self.game.height,
                float(observation_dict["next_pipe_dist_to_player"])/self.game.width,
                #float(observation_dict["next_pipe_top_y"])/self.game.height,
                float(observation_dict["next_pipe_bottom_y"])/self.game.height
            ])
        
        elif self.env_name == "catcher":
            observation = np.array([
                float(observation_dict["player_x"])/self.game.width,
                float(observation_dict["fruit_x"])/self.game.width,
                float(observation_dict["fruit_y"])/self.game.height,
            ])
        
        elif self.env_name == "puckworld":
            observation = np.array([
                float(observation_dict["player_x"])/self.game.width,
                float(observation_dict["player_y"])/self.game.height,
                #float(observation_dict["player_velocity_x"])/self.game.width,
                #float(observation_dict["player_velocity_y"])/self.game.height,
                float(observation_dict["good_creep_x"])/self.game.width,
                float(observation_dict["good_creep_y"])/self.game.height,
                float(observation_dict["bad_creep_x"])/self.game.width,
                float(observation_dict["bad_creep_y"])/self.game.height,
            ])

        elif self.env_name == "pong":
            observation = np.array([
                # float(observation_dict["player_x"])/self.game.width,
                float(observation_dict["player_y"])/self.game.height,
                # float(observation_dict["player_velocity"])*100/self.game.height,
                #float(observation_dict["cpu_y"])/self.game.height,
                #(float(observation_dict["ball_x"]) - float(observation_dict["player_x"]))/self.game.width,
                float(observation_dict["ball_y"])/self.game.height,
            ])

        elif self.env_name == "pixelcopter":
            observation = np.array([
                float(observation_dict["player_y"])/self.game.height,
                float(observation_dict["player_vel"]), #/self.game.height,
                float(observation_dict["player_dist_to_ceil"])/self.game.height,
                float(observation_dict["player_dist_to_floor"])/self.game.height,
            ])
            
        elif self.env_name == "waterworld":
            good_creep_pos = bad_creep_pos = [0.]*6
            _good_creep_pos = observation_dict["creep_pos"]["GOOD"]
            _bad_creep_pos = observation_dict["creep_pos"]["BAD"]
            for idx, pos in enumerate(_good_creep_pos):
                good_creep_pos[idx*2] = float(pos[0])/self.game.width
                good_creep_pos[idx*2+1] = float(pos[1])/self.game.height
            for idx, pos in enumerate(_bad_creep_pos):
                bad_creep_pos[idx*2] = float(pos[0])/self.game.width
                bad_creep_pos[idx*2+1] = float(pos[1])/self.game.height
            observation = np.array([
                    float(observation_dict["player_x"])/self.game.width,
                    float(observation_dict["player_y"])/self.game.height,
                ] + good_creep_pos + bad_creep_pos
            )
            assert observation.shape[0] == 2 + len(good_creep_pos) + len(bad_creep_pos)

        elif self.env_name == "snake":
            observation = np.array([
                float(observation_dict["snake_head_x"])/self.game.width,
                float(observation_dict["snake_head_y"])/self.game.height,
                float(observation_dict["food_x"])/self.game.height,
                float(observation_dict["food_y"])/self.game.height,
            ])
        elif self.env_name == "":
            assert False

        return observation

    def step(self, action):
        self.current_step += 1
        reward = self.p.act(self.all_possible_actions[action])
        observation = self.get_state()
        terminated = self.p.game_over() or self.current_step > self.max_episode_steps
        #if not terminated and self.env_name == "pong":
        #    reward += 0.01
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        self.current_step = 0
        self.p.reset_game()
        info = {}  # Additional info for debugging
        return self.get_state(), info


def ple_env_creator(config):
    env = PLEPygameEnv(config=config)
    return env

if __name__ == "__main__":
    from ray.rllib.utils import check_env
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    #for env_name in ["snake", "pong", "waterworld", "snake"]:
    NUM_TRIALS = 10
    game_name = sys.argv[1]

    for env_name in [game_name]:
        env_config = {"name": env_name}
        env = ple_env_creator(env_config) 
        # PLEPygameEnv(config=env_config) # return an env instance
        check_env(env)
        print(f"{env_name} passed")
        print(env.p.getActionSet())

        episode_rewards = []
        episode_lengths = []
        for _ in range(NUM_TRIALS):
            i = 0
            total_reward = 0
            done = False
            observation, _ = env.reset()
            while not done:
                action = env.action_space.sample()
                ### pong
                #if observation[0] < observation[3]:
                #    action = 1
                #else:
                #    action = 0

                ### puckworld
                #if observation[0] < observation[6]:
                #    action = 2
                #else: 
                #    action = 1
                #elif observation[1] < observation[5]:
                #    action = 4

                ### catcher
                #if observation[0] < observation[1]:
                #    action = 2
                #else:
                #    action = 1
                observation, reward, done, truncated, info = env.step(action)
                total_reward += reward
                i += 1
            episode_lengths.append(i)
            episode_rewards.append(total_reward)

        print(episode_lengths)
        print(episode_rewards)
        print('mean episode_rewards over {NUM_TRIALS}', np.mean(episode_rewards))
        env.close()
