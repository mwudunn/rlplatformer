import gym
import logging
import numpy

from gym import error, spaces, utils
from gym.utils import seeding
from gym_platformer.envs.platformer_game import *

logger = logging.getLogger(__name__)

class PlatformerEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    ACTIONS = [Action.LEFT, Action.RIGHT, Action.JUMP]

    def __init__(self, platformer_file="test_map.npy", plaformer_size=None, enable_render=True, alloted_time=50000):
        self.viewer = None
        self.enable_render = enable_render

        if platformer_file:
            #TODO: Add platform naming abilities
            self.platformer_view = Platformer2D(file_path=platformer_file,
                                                enable_render=enable_render,
                                                alloted_time=alloted_time)
        else:
            raise AttributeError("Failed to find platformer file")
        
        self.game_map = self.platformer_view.game_map
        self.window_size = self.platformer_view.window_size
        
        #TODO: Define action space
        self.action_space = spaces.Discrete(3)

        #TODO: Define observation - window size is a tuple defining grid
        #obs_shape = len(self.window_size) + 3
        num_coins = len(self.platformer_view.game_map.coins)
        obs_shape = len(self.window_size) + num_coins
        low = np.zeros(obs_shape, dtype=int)
        #high =  np.array([self.window_size[0], self.window_size[1], float("inf"), float("inf"), alloted_time])
        high = [self.window_size[0], self.window_size[1]]
        high.extend([1 for i in range(num_coins)])
        high =  np.array(high)
        self.observation_space = spaces.Box(low, high, dtype=np.int64)

        # Initial conditions
        self.state = self.get_state()
        self.goal_location = self.platformer_view.get_goal_location()

        # Simulation related variables
        self.seed()
        self.reset()

        # Initialize the relevant attributes
        # self.configure()

    def step(self, action):
        """

        Parameters
        ----------
        action :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        prev_ob = self.get_state()

        _ = self.take_action(action)

        
        #TODO: Define what an observation is (window) 
        ob = self.get_state()
        
        #TODO: Get reward from current state:
        reward = self.get_reward(ob, prev_ob)
        

        # Define game over variables
        done = self.done()

        return ob, reward, done, {}

    def reset(self):
        #TODO reset game in platformer_game
        self.platformer_view.reset()
        return self.get_state()

    def render(self, mode='human', close=False):
        self.platformer_view.render(mode, close)
    
    def get_state(self):
        self.state = self.platformer_view.get_state()
        return self.state

    def take_action(self, action): 
        action = int(action)
        self.platformer_view.perform_action(self.ACTIONS[action])
        self.platformer_view.update_environment()

    
    def manhattanDistance(self, pt1, pt2):
        return abs(int(pt1[0]) - int(pt2[0])) + abs(int(pt1[1]) - int(pt2[1]))
    
    def get_distance_from_goal(self):
        # print(self.platformer_view.get_player().get_location(), self.goal_location)
        return self.manhattanDistance(self.get_state(), self.goal_location)


    def get_reward(self, ob, prev_ob):
        """ Reward is given for minimizing the distance to the goal. """
        prev_coins = prev_ob[2:]
        coins = ob[2:]
        # Encode:
        #   - At goal (max reward for reaching the goal)
        #   - Can see goal/seen goal: get better rewards
        #   - Velocity (higher abs(velocity)  - better rewards)
        #   - Clock time (> clock time - worse rewards)
        #   - Distance from goal (closer to goal - better rewards)
        #   - Depth?
        #   - Stuck (velocity = 0), encourage jumping/changing direction )
        
        # Can see goal:
        if self.game_map.check_goal_collision(self.platformer_view.get_player()):
            remaining_time = self.platformer_view.get_remaining_time()
            print(remaining_time)
            return 10.0 * (self.platformer_view.get_remaining_time() / self.platformer_view.alloted_time) + 10
        if sum(prev_coins) != sum(coins):
            return 1.0
        return 0.0
        #return 1.0


    def done(self):
        return self.platformer_view.game_over
