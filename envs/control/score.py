import gym

from envs.base import EnvBase
from utils.functions.gym import control_task_name

class Env(EnvBase):

    def __init__(self, args, rank, size):

        for key in args.additional_arguments:
            if key not in ['task', 'trials']:
                raise RuntimeError("Control Score does not support `args.additional_arguments['" + key + "']`.")
        
        tasks = ['acrobot', 'cart_pole', 'mountain_car', 'mountain_car_continuous', 'pendulum', 'bipedal_walker',
                 'bipedal_walker_hardcore', 'lunar_lander', 'lunar_lander_continuous', 'ant', 'half_cheetah',
                 'hopper', 'humanoid', 'humanoid_standup', 'inverted_double_pendulum', 'inverted_pendulum',
                 'reacher', 'swimmer', 'walker_2d']
                 
        if 'task' not in args.additional_arguments:
            raise RuntimeError("Control Score requires `args.additional_arguments['task']`.")
        elif args.additional_arguments['task'] not in tasks:
            raise RuntimeError("Control Score does not support `args.additional_arguments['task']`: " + \
                               args.additional_arguments['task'] + ". Please select a task from ", tasks, ".")

        if 'trials' not in args.additional_arguments:
            args.additional_arguments['trials'] = 1
        elif not isinstance(args.additional_arguments['trials'], int) or args.additional_arguments['trials'] < 1:
            raise RuntimeError("Control Score requires `args.additional_arguments['trials']` >= 1.")

        super().__init__(args, rank, size)

        task = control_task_name(args.additional_arguments['task'])

        self.emulator = gym.make(task)

    def run(self, gen_nb):

        [bot] = self.bots
        bot_fitness = 0

        for i in range(self.args.additional_arguments['trials']):

            self.emulator.seed(gen_nb * self.args.additional_arguments['trials'] + i)
            obs = self.emulator.reset()
            done = False

            while not done:

                obs, rew, done, _ = self.emulator.step( bot(obs) )
                
                bot_fitness += rew

            bot.reset()

        bot_fitness /= self.args.additional_arguments['trials']

        return [bot_fitness]