from gym.envs.registration import register

register(
        id='platformer-v0',
        entry_point='gym_plaformer.envs:PlatformerEnv',

        )
