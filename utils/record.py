import argparse
import glob
import numpy as np
import os
import pickle
import random
import sys
import torch
import warnings

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/..')

warnings.filterwarnings("ignore", category=UserWarning)
sys.setrecursionlimit(2**31-1)

parser = argparse.ArgumentParser()

parser.add_argument('--state_path', '-s', type=str, required=True,
                    help="Path to the saved state. \
                          <=> data/states/<env_path>/<additional_arguments>/<bots_path>/<population_size>/<generation>/")

parser.add_argument('--nb_obs', '-o', type=int, default=2**31-1,
                    help="Number of observations to record the agent on.")

args = parser.parse_args()

MAX_INT = 2**31-1

"""
Process arguments
"""

if not os.path.isfile(args.state_path + 'scores.npy'):
    raise RuntimeError('This state has not yet been evaluated. Please run `utils/evaluate.py`.')

if args.state_path[-1] == '/':
    args.state_path = args.state_path[:-1]

split_path = args.state_path.split('/')

env_path = split_path[-5]
additional_arguments = split_path[-4]
bots_path = split_path[-3]
pop_size = int(split_path[-2])
gen_nb = int(split_path[-1])

split_additional_arguments = additional_arguments.split('~')

task = split_additional_arguments[0].split('.')[1]
trials = split_additional_arguments[1].split('.')[1]

"""
Initialize environment
"""

import gym
from gym import wrappers
from utils.functions.gym import control_task_name

emulator = gym.make( control_task_name(task) )

emulator = wrappers.RecordVideo(emulator, args.state_path) # MuJoCo will not display the video in real-time

"""
Import bots
"""

if 'static.rnn' in bots_path:
    from bots.static.rnn.control import Bot
else: # 'dynamic.rnn' in bots_path:
    from bots.dynamic.rnn.control import Bot

"""
Find elite
"""

pkl_files = [os.path.basename(x) for x in glob.glob(args.state_path + '/*.pkl')]

state_files = []

for pkl_file in pkl_files:

    if pkl_file[:-4].isdigit():

        state_files.append(pkl_file)

if len(state_files) == 0:
    raise RuntimeError("Directory '" + args.state_path + "/' empty.")

try:

    with open(args.state_path + '/0.pkl', 'rb') as f:
        state = pickle.load(f)

except IOError:

    print("File '" + args.state_path + "/0.pkl' doesn't exist / is corrupted.")

if len(args.state_path) == 3:

    full_seed_list, _, _ = state

else: # len(state) == 4:

    _, _, latest_fitnesses_and_bot_sizes, bots = state

    for i in range( 1, len(state_files) ):

        try:

            with open(args.state_path + '/' + str(i) + '.pkl', 'rb') as f:
                bots += pickle.load(f)[0]

        except IOError:

            print("File '" + args.state_path + '/' + str(i) + ".pkl' doesn't exist / is corrupted.")

    fitnesses_sorting_indices = latest_fitnesses_and_bot_sizes[:, :, 0].argsort(axis=0)
    fitnesses_rankings = fitnesses_sorting_indices.argsort(axis=0)
    selected = np.greater_equal(fitnesses_rankings, pop_size//2)
    selected_indices = np.where(selected[:,0] == True)[0]

scores = np.load(args.state_path + '/scores.npy')

i = scores.mean(axis=1).argmax()

if len(state) == 3:

    bot = Bot(0)
    bot.build(full_seed_list[i][0])

else: # len(state) == 4:

    bot = bots[selected_indices[i]][0]

bot.setup_to_run()
bot.reset()

np.random.seed(MAX_INT)
torch.manual_seed(MAX_INT)
random.seed(MAX_INT)

emulator.seed(MAX_INT)
obs = emulator.reset()
done = False

score=0

if 'dynamic.rnn' in bots_path:
    with open(args.state_path + '/net.pkl', 'wb') as f:
        pickle.dump(str(bot.nets[0].nodes['layered']), f)
    print('Architecture saved at ' + args.state_path + '/net.pkl')

if hasattr(bot, 'n'):
    print("Number of states experienced: ", bot.n)
    
for k in range(args.nb_obs):

    obs, rew, done, _ = emulator.step( bot(obs) )
    score += rew

    if done:
        print('Score:', score)
        break