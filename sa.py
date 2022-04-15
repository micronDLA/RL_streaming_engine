#------------------------------------------------------------------------------+
#
#   Nathan A. Rooy
#   Simple Simulated Annealing
#   from: https://github.com/nathanrooy/simulated-annealing
#   2019 - DEC
#
#------------------------------------------------------------------------------+

#--- IMPORT DEPENDENCIES ------------------------------------------------------+

from random import randint
from random import random
from math import exp
from math import log
from train_alt import get_nodes_rand
from collections import deque
from tqdm import tqdm
import time
import numpy as np

#--- MAIN ---------------------------------------------------------------------+

class minimize():
    '''Simple Simulated Annealing
    '''

    def __init__(self, args, env, graphdef, device, writer, cooling_schedule='linear', step_max=1000, t_min=0, t_max=100, bounds=[], alpha=None, damping=1):

        # checks
        assert cooling_schedule in ['linear','exponential','logarithmic', 'quadratic'], 'cooling_schedule must be either "linear", "exponential", "logarithmic", or "quadratic"'

        # initialize starting conditions
        self.reward_buf = deque(maxlen=100)
        self.reward_buf.append(0)
        self.t = t_max
        self.t_max = t_max
        self.t_min = t_min
        self.step_max = step_max
        self.hist = []
        self.cooling_schedule = cooling_schedule
        self.args, self.env, self.graphdef, self.device, self.writer = args, env, graphdef, device, writer

        self.bounds = bounds[:]
        self.damping = damping
        # current_state: vector of all nodes placed
        # current_energy: mean(reward buf)
        reward, nodes_place = get_nodes_rand([], self.args, self.env, self.graphdef, self.device, self.reward_buf)
        self.current_energy, self.current_state = reward, nodes_place

        self.best_state = self.current_state
        self.best_energy = self.current_energy
        print(f'INIT graph ready time yet: {self.best_energy}, {self.best_state}')

        # initialize cooling schedule
        if self.cooling_schedule == 'linear':
            if alpha != None:
                self.update_t = self.cooling_linear_m
                self.cooling_schedule = 'linear multiplicative cooling'
                self.alpha = alpha

            if alpha == None:
                self.update_t = self.cooling_linear_a
                self.cooling_schedule = 'linear additive cooling'

        if self.cooling_schedule == 'quadratic':
            if alpha != None:
                self.update_t = self.cooling_quadratic_m
                self.cooling_schedule = 'quadratic multiplicative cooling'
                self.alpha = alpha

            if alpha == None:
                self.update_t = self.cooling_quadratic_a
                self.cooling_schedule = 'quadratic additive cooling'

        if self.cooling_schedule == 'exponential':
            if alpha == None: self.alpha =  0.8
            else: self.alpha = alpha
            self.update_t = self.cooling_exponential

        if self.cooling_schedule == 'logarithmic':
            if alpha == None: self.alpha =  0.8
            else: self.alpha = alpha
            self.update_t = self.cooling_logarithmic


        # begin optimizing
        self.step, self.accept = 1, 0
        i_episode = 0
        pbar = tqdm(total=self.step_max)
        while self.step < self.step_max and self.t >= self.t_min and self.t > 0:

            env.reset()
            # get neighbor
            reward, proposed_neighbor = self.get_neighbor()

            # check energy level of neighbor
            E_n = reward
            dE = E_n - self.current_energy

            # determine if we should accept the current neighbor
            if random() < self.safe_exp(-dE / self.t):
                self.current_energy = E_n
                self.current_state = proposed_neighbor[:]
                self.accept += 1

            # check if the current neighbor is best solution so far
            if E_n < self.best_energy:
                self.best_energy = E_n
                self.best_state = proposed_neighbor[:]
                print(f'Best graph ready time yet: {self.best_energy}, {self.best_state}')

                self.writer.add_scalar('SA Best readytime/episode', self.best_energy, i_episode)
                self.writer.flush()


            # persist some info for later
            self.hist.append([
                self.step,
                self.t,
                self.current_energy,
                self.best_energy])

            # update some stuff
            self.t = self.update_t(self.step)
            if reward < 100:
                self.step += 1
                pbar.update(1)
            if i_episode % self.args.log_interval == 0:
                self.writer.add_scalar('SA Mean reward/episode', np.mean(self.reward_buf), i_episode)
                self.writer.flush()
            i_episode += 1

        # generate some final stats
        self.acceptance_rate = self.accept / self.step


    def get_neighbor(self):
        '''
        get neighbor by select a node in sequence to drop from current
        then random place the remaining nodes
        '''
        x = randint(0, len(self.current_state))
        cur_s = self.current_state[:x]
        reward, next_s = get_nodes_rand(cur_s, self.args, self.env, self.graphdef, self.device, self.reward_buf)
        return reward, next_s


    def results(self):
        print('+------------------------ RESULTS -------------------------+\n')
        print(f'cooling sched.: {self.cooling_schedule}')
        if self.damping != 1: print(f'       damping: {self.damping}\n')
        else: print('\n')

        print(f'  initial temp: {self.t_max}')
        print(f'    final temp: {self.t:0.6f}')
        print(f'     max steps: {self.step_max}')
        print(f'    final step: {self.step}\n')

        print(f'  final energy: {self.best_energy:0.6f}\n')
        print('+-------------------------- END ---------------------------+')

    # linear multiplicative cooling
    def cooling_linear_m(self, step):
        return self.t_max /  (1 + self.alpha * step)

    # linear additive cooling
    def cooling_linear_a(self, step):
        return self.t_min + (self.t_max - self.t_min) * ((self.step_max - step)/self.step_max)

    # quadratic multiplicative cooling
    def cooling_quadratic_m(self, step):
        return self.t_min / (1 + self.alpha * step**2)

    # quadratic additive cooling
    def cooling_quadratic_a(self, step):
        return self.t_min + (self.t_max - self.t_min) * ((self.step_max - step)/self.step_max)**2

    # exponential multiplicative cooling
    def cooling_exponential_m(self, step):
        return self.t_max * self.alpha**step

    # logarithmical multiplicative cooling
    def cooling_logarithmic_m(self, step):
        return self.t_max / (self.alpha * log(step + 1))


    def safe_exp(self, x):
        try: return exp(x)
        except: return 0

#--- END ----------------------------------------------------------------------+
