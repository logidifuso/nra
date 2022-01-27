"""
Main script of the library, executed by as many processes as are provided through `mpiexec`.

After parsing the user's arguments, all processes initialize an environment object (env) and other experiment specific
variables in order to loop over the main evolutionary algorithm operations (variation, evaluation & selection)
and the various MPI-powered communication events that ought to take place during the process.
"""
import argparse
import copy
import numpy as np
import pickle
import sys
import time
import warnings
from mpi4py import MPI

from utils.functions.misc import initialize_environment

np.set_printoptions(suppress=True)
warnings.filterwarnings('ignore')
sys.setrecursionlimit(2**31-1)

parser = argparse.ArgumentParser()

parser.add_argument('--env_path', '-e', type=str, required=True,
                    help="Path to the env class file.")

parser.add_argument('--bots_path', '-b', type=str, required=True,
                    help="Path to the bot class file.")

parser.add_argument('--population_size', '-p', type=int, required=True,
                    help="Number of bots per population. Must be a multiple of the number of MPI processes \
                          and must remain constant across successive experiments.")

parser.add_argument('--nb_elapsed_generations', '-l', type=int, default=0,
                    help="Number of elapsed generations.")

parser.add_argument('--nb_generations', '-g', type=int, required=True,
                    help="Number of generations to run.")

parser.add_argument('--elitism', '-t', type=float, default=0,
                    help="Proportion (if float in [0, 0.5]) or number (if int in [0, 0.5*pop_size]) of the \
                          best performing bots which will not be mutated each generation.")

parser.add_argument('--save_frequency', '-f', type=int, default=0,
                    help="Frequency (int in [0, nb_generations]) at which to save the experiment's state.")

parser.add_argument('--communication', '-c', choices=['ps', 'ps_p2p', 'big_ps_p2p'], default='ps_p2p',
                    help="ps : A primary process scatters/gathers data to/from secondary processes. \
                          ps_p2p : ps + peer-to-peer data exchange between all processes. \
                          big_ps_p2p : ps_p2p - initial/final bot scatter/gather (when combined size of bots > 2GB). \
                                       (the number of MPI processes must remain constant for successive experiments) \
                          All protocols must remain constant across successive experiments.")

parser.add_argument('--enable_gpu_use', '-u', type=int, default=0,
                    help="Makes use of GPUs if they are available.")

parser.add_argument('--additional_arguments', '-a', type=str, default='{}',
                    help="JSON string or path to a JSON file of additional arguments.")

args = parser.parse_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

env = initialize_environment(args, rank, size)
old_nb_gen = args.nb_elapsed_generations
new_nb_gen = args.nb_generations
pop_size = args.population_size
batch_size = pop_size // size

ps_comm = args.communication == 'ps'
ps_p2p_comm = args.communication == 'ps_p2p'
big_ps_p2p_comm = args.communication == 'big_ps_p2p'
p2p_comm = args.communication == 'ps_p2p' or args.communication == 'big_ps_p2p'

full_seed_list = None
fitnesses = None

full_seed_list_batch = np.empty((batch_size, 1, 1), dtype=np.uint32)
fitnesses_batch = np.empty((batch_size, 1), dtype=np.float32)

if p2p_comm:

    bots = None
    pairing_and_seeds = None
    fitnesses_and_bot_sizes = None

    bots_batch = []

    # [MPI buffer size, pair position, sending, seed]
    pairing_and_seeds_batch = np.empty((batch_size, 1, 4), dtype=np.uint32)

    # [fitness, pickled bot size]
    fitnesses_and_bot_sizes_batch = np.empty((batch_size, 1, 2), dtype=np.float32) 

if rank == 0:

    fitnesses = np.empty((pop_size, 1), dtype=np.float32) 

    pairing_and_seeds = np.empty((pop_size, 1, 4), dtype=np.uint32) 

    fitnesses_and_bot_sizes = np.empty((pop_size, 1, 2), dtype=np.float32)

    if old_nb_gen > 0:

        state = env.io.load_state()

        if ps_comm:

            full_seed_list, full_fitness_list, latest_fitnesses = state

            fitnesses_sorting_indices = latest_fitnesses.argsort(axis=0)

        else: # p2p_comm:

            if ps_p2p_comm:
                full_seed_list, full_fitness_list, latest_fitnesses_and_bot_sizes, bots = state
            else: # big_ps_p2p_comm:
                full_seed_list, full_fitness_list, latest_fitnesses_and_bot_sizes, bots_batch = state

            fitnesses_sorting_indices = latest_fitnesses_and_bot_sizes[:, :, 0].argsort(axis=0)

        fitnesses_rankings = fitnesses_sorting_indices.argsort(axis=0)

    else: # old_nb_gen == 0:

        full_seed_list = np.empty((pop_size, 1, 0), dtype=np.uint32)
        full_fitness_list = np.empty((pop_size, 1, 0), dtype=np.float32)

if old_nb_gen > 0:

    if ps_p2p_comm:

        if rank == 0:

            for i in range(pop_size):
                fitnesses_and_bot_sizes[i, 0, 1] = len( pickle.dumps(bots[i][0]) )

            bots = [ bots[i * batch_size: (i+1) * batch_size] for i in range(size) ]

        bots_batch = comm.scatter(bots, root=0)

    elif big_ps_p2p_comm:
    
        if rank != 0:
            [bots_batch] = env.io.load_state()

        for i in range(batch_size):
            fitnesses_and_bot_sizes_batch[i, 0, 1] = len( pickle.dumps(bots_batch[i][0]) )

        comm.Gather(fitnesses_and_bot_sizes_batch, fitnesses_and_bot_sizes, root=0)

for gen_nb in range(old_nb_gen, old_nb_gen + new_nb_gen):

    np.random.seed(gen_nb)

    if rank == 0:

        start = time.time()
        
        new_seeds = env.io.generate_new_seeds(gen_nb)
        
        full_seed_list = np.concatenate((full_seed_list, new_seeds), 2)

        if gen_nb != 0:
            full_seed_list[:, 0] = full_seed_list[:, 0][ fitnesses_rankings[:, 0] ]

    if ps_comm or gen_nb == 0:

        full_seed_list_batch = np.empty((batch_size, 1, gen_nb + 1), dtype=np.uint32)

        comm.Scatter(full_seed_list, full_seed_list_batch, root=0)

    else: # p2p_comm and gen > 0:

        if rank == 0:

            pairing_and_seeds[:, :, 0] = np.max(fitnesses_and_bot_sizes[:, :, 1]) # MPI buffer size

            pair_ranking = (fitnesses_rankings[:, 0] + pop_size // 2) % pop_size

            pairing_and_seeds[:, 0, 1] = fitnesses_sorting_indices[:,0][pair_ranking] # pair position

            pairing_and_seeds[:, :, 2] = np.greater_equal(fitnesses_rankings, pop_size // 2) # sending

            pairing_and_seeds[:, :, 3] = full_seed_list[:, :, -1] # seed

        comm.Scatter(pairing_and_seeds, pairing_and_seeds_batch, root=0)

        req = []

        for i in range(batch_size):

            pair = int(pairing_and_seeds_batch[i, 0, 1] // batch_size)

            if pairing_and_seeds_batch[i, 0, 2] == 1: # sending

                tag = int(pop_size * 0 + batch_size * rank + i)

                req.append( comm.isend(bots_batch[i][0], dest=pair, tag=tag) )

            else: # pairing_and_seeds_batch[i, 0, 2] == 0: # receiving

                tag = int(pop_size * 0 + pairing_and_seeds_batch[i, 0, 1])

                req.append( comm.irecv(pairing_and_seeds_batch[i, 0, 0], source=pair, tag=tag) )

        received_bots = MPI.Request.waitall(req)

        for i, bot in enumerate(received_bots):
            if bot is not None:
                bots_batch[i // 1][i % 1] = bot

    for i in range(batch_size):

        if ps_comm or gen_nb == 0:

            env.build_bots(full_seed_list_batch[i]) # Variations from scratch

        else:

            env.bots = bots_batch[i]
            env.extend_bots(pairing_and_seeds_batch[i, :, 3]) # Variation

        fitnesses_batch[i] = env.evaluate_bots(gen_nb) # Evaluation

        if p2p_comm:

            if gen_nb == 0:
                bots_batch.append( copy.deepcopy(env.bots) )

            fitnesses_and_bot_sizes_batch[i, :, 0] = fitnesses_batch[i]

            fitnesses_and_bot_sizes_batch[i, 0, 1] = len(pickle.dumps(bots_batch[i][0]))
    
    if ps_comm:

        comm.Gather(fitnesses_batch, fitnesses, root=0)

    else: # p2p_comm:

        comm.Gather(fitnesses_and_bot_sizes_batch, fitnesses_and_bot_sizes, root=0)

    if rank == 0:

        if p2p_comm:
            fitnesses = fitnesses_and_bot_sizes[:, :, 0]

        fitnesses_sorting_indices = fitnesses.argsort(axis=0)
        fitnesses_rankings = fitnesses_sorting_indices.argsort(axis=0)

        full_seed_list[:, 0] = full_seed_list[:, 0][fitnesses_sorting_indices[:, 0]]

        full_seed_list[:pop_size // 2] = full_seed_list[pop_size // 2:] # Selection

        print(gen_nb + 1, ':', int( time.time() - start ), '\n', np.mean(fitnesses, 0), '\n', np.max(fitnesses, 0) )

        full_fitness_list = np.concatenate((full_fitness_list, fitnesses[:, :, None]), 2)

    if gen_nb + 1 in env.io.save_points:
        
        if ps_comm:

            if rank == 0:
                env.io.save_state([full_seed_list, full_fitness_list, fitnesses], gen_nb + 1)

        if ps_p2p_comm:
            
            batched_bots = comm.gather(bots_batch, root=0)

            if rank == 0:
                
                bots = []

                for bot_batch in batched_bots:
                    bots = bots + bot_batch

                env.io.save_state([full_seed_list, full_fitness_list, fitnesses_and_bot_sizes, bots], gen_nb + 1)

        if big_ps_p2p_comm:

            if rank == 0:
                env.io.save_state([full_seed_list, full_fitness_list, fitnesses_and_bot_sizes, bots_batch], gen_nb + 1)
            else: # rank != 0:
                env.io.save_state([bots_batch], gen_nb + 1)