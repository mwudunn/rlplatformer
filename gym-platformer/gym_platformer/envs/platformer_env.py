import gym
import logging
import numpy

from gym import error, spaces, utils
from gym.utils import seeding
from gym_platformer.envs.plaformer_game import Plaformer2D

logger = logging.getLogger(__name__)

class PlatformerEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    ACTIONS = ["Left", "Right", "Jump"]

    def __init__(self, platformer_file=None, plaformer_size=None, enable_render=True):
        self.viewer = None
        self.enable_render = enable_render

        if platformer_file:
            #TODO: Add platform naming abilities
            self.platformer_view = Platformer2d(file_path=platformer_file,
                                                size=platformer_size,
                                                render=enable_render)
        else:
            raise AttributeError("Failed to find platformer file")

        
        #TODO: Define action space

        #TODO: Define observation

        # Initial conditions
        self.start_point = self.platformer_view.start_point
        self.goal = self.platformer_view.goal
        self.player = self.platformer_view.player

        # Simulation related variables
        self.seed()
        self.reset()

        # Initialize the relevant attributes
        self.configure()

    def _step(self, action):
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
        _ = self._take_action(action)
        
        #TODO: Define what an observation is (window) 
        ob = self._get_state()
        
        #TODO: Get reward from current state:
        reward = self._get_reward()
        
        # Define game over variables
        done = self._done()

        return ob, reward, done, {}

    def _reset(self):
        #TODO reset game in platformer_game
        pass

    def _render(self, mode='human', close=False):
        pass
    
    def _get_state(self):
        pass

    def _take_action(self, action): 

        if isinstance(action, int):
            self.player.perform_action(self.ACTIONS[action])
        else:
            self.player.perform_action(action)

    def _get_reward(self):
        """ Reward is given for minimizing the distance to the goal. """
        current_state = self.env.getState()

    def _done(self):
        return self.platformer_view.game_over
