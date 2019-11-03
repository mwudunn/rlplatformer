import gym
import logging
import numpy

from gym import error, spaces, utils
from gym.utils import seeding
from gym_platformer.envs.platformer_game import *

logger = logging.getLogger(__name__)

class PlatformerEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    ACTIONS = [Action.NONE, Action.LEFT, Action.RIGHT, Action.JUMP]

    def __init__(self, platformer_file="test_map.npy", plaformer_size=None, enable_render=False):
        self.viewer = None
        self.enable_render = enable_render

        if platformer_file:
            #TODO: Add platform naming abilities
            self.platformer_view = Platformer2D(file_path=platformer_file,
                                                render=enable_render)
        else:
            raise AttributeError("Failed to find platformer file")
        
        self.window_size = self.platformer_view.window_size
        
        #TODO: Define action space
        self.action_space = spaces.Discrete(3)

        #TODO: Define observation - window size is a tuple defining grid
        low = np.zeros(len(self.window_size), dtype=int)
        high =  np.array(self.window_size, dtype=int) - np.ones(len(self.window_size), dtype=int)
        self.observation_space = spaces.Box(low, high, dtype=np.int64)

        # Initial conditions
        self.player = (self.platformer_view.player)
        self.state = (None, self.player.get_location()) 
        self.goal = self.platformer_view.goal
        self.dt = 1

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
        _ = self.take_action(action)
        
        #TODO: Define what an observation is (window) 
        ob = self.get_state()
        
        #TODO: Get reward from current state:
        reward = self.get_reward()
        
        # Define game over variables
        done = self.done()

        return ob, reward, done, {}

    def reset(self):
        #TODO reset game in platformer_game
        self.platformer_view.reset()
        return self.get_state()

    def render(self, mode='human', close=False):
        pass
    
    def get_state(self):
        self.state = (self.platformer_view.window, self.player.get_location())
        return self.state

    def take_action(self, action): 
        if isinstance(action, int):
            self.player.perform_action(self.ACTIONS[action], self.dt)
        else:
            self.player.perform_action(action, self.dt)
        self.platformer_view.update_environment()
    
    def manhattanDistance(pt1, pt2):
        return abs(p1[0] - pt2[0]) + abs(pt1[1] - pt2[1])
    
    def get_distance_from_goal(self):
        return manhattanDistance(self.player.get_location(), self.goal)

    def get_reward(self):
        """ Reward is given for minimizing the distance to the goal. """
        state = self.get_state()
        # Encode:
        #   - At goal (max reward for reaching the goal)
        #   - Can see goal/seen goal: get better rewards
        #   - Velocity (higher abs(velocity)  - better rewards)
        #   - Clock time (> clock time - worse rewards)
        #   - Distance from goal (closer to goal - better rewards)
        #   - Depth?
        #   - Stuck (velocity = 0), encourage jumping/changing direction )
        
        # Can see goal:
        goalVisible = False
        if 2 in state:
            goalVisible = True
        
        if state[1] == self.goal:
            return 10.0
        elif abs(self.player.velocity[0]):
            return 1.0/(self.get_distance_from_goal())
        else:
            return -1.0/(self.get_distance_from_goal())


    def done(self):
        return self.platformer_view.game_over
