import os
import time

from QLearning.infrastructure.rl_trainer import RL_Trainer
from QLearning.agents.dqn_agent import DQNAgent
from QLearning.infrastructure.dqn_utils import get_env_kwargs
from QLearning.scripts.exploration import ExemplarExploration
from QLearning.scripts.density_model import Exemplar


class Q_Trainer(object):

    def __init__(self, params):
        self.params = params

        train_args = {
            'num_agent_train_steps_per_iter': params['num_agent_train_steps_per_iter'],
            'num_critic_updates_per_agent_update': params['num_critic_updates_per_agent_update'],
            'train_batch_size': params['batch_size'],
            'double_q': params['double_q'],
            'learning_rate': params['learning_rate'],
        }

        env_args = get_env_kwargs(params['env_name'])

        self.agent_params = {**train_args, **env_args, **params}

        self.params['agent_class'] = DQNAgent
        self.params['agent_params'] = self.agent_params
        self.params['train_batch_size'] = params['batch_size']
        self.params['env_wrappers'] = self.agent_params['env_wrappers']

        self.rl_trainer = RL_Trainer(self.params)

    def run_training_loop(self):
        self.rl_trainer.run_training_loop(
            self.agent_params['num_timesteps'],
            collect_policy = self.rl_trainer.agent.actor,
            eval_policy = self.rl_trainer.agent.actor,
            )

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name',  default='Platformer-v0',
                        choices=('PongNoFrameskip-v4',
                                 'LunarLander-v2',
                                 'Platformer-v0')
                        )

    parser.add_argument('--ep_len', type=int, default=200)
    parser.add_argument('--exp_name', type=str, default='todo')

    parser.add_argument('--eval_batch_size', type=int, default=1000)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1)
    parser.add_argument('--num_critic_updates_per_agent_update', type=int, default=1)
    parser.add_argument('--double_q', action='store_true')

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--use_gpu', '-gpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)
    parser.add_argument('--scalar_log_freq', type=int, default=int(1e4))
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001)
    parser.add_argument('--save_params', action='store_true')
    parser.add_argument('--render', type=bool, default=False)

    parser.add_argument('--bonus_coeff', '-bc', type=float, default=1)
    parser.add_argument('--density_model', type=str, default='ex2')
    parser.add_argument('--kl_weight', '-kl', type=float, default=1e-2)
    parser.add_argument('--density_lr', '-dlr', type=float, default=5e-3)
    parser.add_argument('--density_train_iters', '-dti', type=int, default=1000)
    parser.add_argument('--density_batch_size', '-db', type=int, default=64)
    parser.add_argument('--density_hiddim', '-dh', type=int, default=32)
    parser.add_argument('--replay_size', '-rs', type=int, default=int(1e6))
    parser.add_argument('--sigma', '-sig', type=float, default=0.2)

    args = parser.parse_args()

    # convert to dictionary
    params = vars(args)
    params['video_log_freq'] = 10000 # This param is not used for DQN
    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    logdir_prefix = 'dqn_'
    if args.double_q:
        logdir_prefix += 'double_q_'

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = logdir_prefix + args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")

    trainer = Q_Trainer(params)
    trainer.run_training_loop()


if __name__ == "__main__":
    import QLearning.agents.dqn_agent
    print(QLearning.agents.dqn_agent.__file__)

    main()
