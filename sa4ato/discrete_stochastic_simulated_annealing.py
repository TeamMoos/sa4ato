"""
    Source:
    Alrefaei, Mahmoud H und Andradóttir, Sigrún:
    A simulated annealing algorithm with constant temperature for discrete stochastic optimization.
    In:Management science,Band 45(5):S. 748–764, 1999.
"""
#    SA4ATO - Simulated Annealing for Assemble to Order
#    Copyright (C) 2022  Timo Kühne, Jonathan Laib
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as published
#    by the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

import time
from dataclasses import dataclass
from enum import Enum
from math import exp
from random import random, randint

import numpy as np


def safe_exp(x):
    try:
        return exp(x)
    except OverflowError:
        return 0


@dataclass
class State:
    value: np.ndarray

    def __eq__(self, other):
        return np.array_equal(self.value, other.value)


class OptSolution(Enum):
    COUNT = 'v1'
    MEAN = 'v2'


class NeighbourhoodFunc(Enum):
    RANDOM_NEIGHBOURHOOD = 'n1'
    DIRECT_NEIGHBOURHOOD = 'n2'


class Minimize:

    def __init__(self, func, x0, opt_solution, neighbourhood_func, t, bounds, step_max=1000, l=1, **kwargs):
        """

        :param func: function to be minimized
        :param x0: starting value
        :param opt_solution:
        :param neighbourhood_func: neighbourhood function which specifies how neighbour is found
        :param t: temperature
        :param bounds: bounds for solution values
        :param step_max: number of iterations
        :param l: number of values to calculate the mean
        """
        # initialize starting conditions
        self.t = t
        self.L = l
        self.step_max = step_max
        self.opt_solution = opt_solution
        self.neighbourhood_func = neighbourhood_func
        self.hist = []

        self.cost_func = func
        self.x0 = State(x0)  # x0
        self.bounds = bounds
        self.current_state = self.x0

        self.best_state = self.current_state
        self.best_energy = np.inf

        # initialize way of finding optimal solution
        if self.opt_solution == OptSolution.COUNT:
            self.l = np.floor(np.log(10 + 1))
            self.list_states = [self.x0]
            self.list_count = [1]  # counts the visits of a specific state
            self.max_count = 1
        if self.opt_solution == OptSolution.MEAN:
            self.list_states = [self.x0]
            self.list_count = [1]  # counts the number of observations made for a specific state
            self.list_mean = [self.cost_func(self.x0.value, **kwargs)]

        # initialize neighbourhood scheme
        if self.neighbourhood_func == NeighbourhoodFunc.RANDOM_NEIGHBOURHOOD:
            self.get_neighbor = self.move_random
        if self.neighbourhood_func == NeighbourhoodFunc.DIRECT_NEIGHBOURHOOD:
            self.get_neighbor = self.move_discrete

        # begin optimizing
        self.step, self.accept = 1, 0
        self.start_time = time.time()
        while self.step <= self.step_max:
            # STEP 1: new proposed_neighbour, equal for v1 and v2
            proposed_neighbor = self.get_neighbor()

            # STEP 2: observing l values for current_state and proposed neighbour, in addition v2 updates the
            # mean/count table
            sum_current_state, sum_proposed_neighbour = 0, 0
            for i in range(self.L):
                sum_current_state += self.cost_func(self.current_state.value, **kwargs)
                sum_proposed_neighbour += self.cost_func(proposed_neighbor.value, **kwargs)

            mean_current_state = sum_current_state / self.L
            mean_proposed_neighbour = sum_proposed_neighbour / self.L

            # update means if opt_solution == OptSolution.MEAN
            if self.opt_solution == OptSolution.MEAN:
                # update values for current_state
                index_current_state = self.list_states.index(self.current_state)
                self.list_mean[index_current_state] = (self.list_mean[index_current_state] *
                                                       self.list_count[index_current_state] +
                                                       mean_current_state * l) / \
                                                      (self.list_count[index_current_state] + l)
                self.list_count[index_current_state] += l

                # update values for proposed_neighbour
                if proposed_neighbor not in self.list_states:
                    self.list_states += [proposed_neighbor]
                    self.list_mean += [mean_proposed_neighbour]
                    self.list_count += [l]
                else:
                    index_proposed_neighbour = self.list_states.index(proposed_neighbor)
                    self.list_mean[index_proposed_neighbour] = (self.list_mean[index_proposed_neighbour] *
                                                                self.list_count[index_proposed_neighbour] +
                                                                mean_proposed_neighbour * l) / \
                                                               (self.list_count[index_proposed_neighbour] + l)
                    self.list_count[index_proposed_neighbour] += l

            # STEP 3: determine if we should accept the current neighbor, equal for v1 and v2
            if random() <= safe_exp(-(mean_proposed_neighbour - mean_current_state) / self.t):
                self.current_state = proposed_neighbor
                self.accept += 1

            # STEP 4: var 1 updates count list
            # update lists if opt_solution == OptSolution.COUNT
            if self.opt_solution == OptSolution.COUNT:
                if self.current_state not in self.list_states:
                    self.list_states += [self.current_state]
                    self.list_count += [1]
                else:
                    self.list_count[self.list_states.index(self.current_state)] += 1

                # check if the current neighbor is the best solution so far
                if self.neighbourhood_func == NeighbourhoodFunc.DIRECT_NEIGHBOURHOOD:
                    norm_current_state = self.normalizer(self.current_state)
                    norm_best_state = self.normalizer(self.best_state)
                else:
                    norm_current_state = 1
                    norm_best_state = 1

                # normalize the number of visits for the current state as proposed in paper
                index_current_state = self.list_states.index(self.current_state)
                if self.list_count[index_current_state] / norm_current_state > self.max_count / norm_best_state:
                    sum_current_state = 0
                    for i in range(self.L):
                        sum_current_state += self.cost_func(self.current_state.value, **kwargs)
                    self.best_energy = sum_current_state / l
                    self.best_state = self.current_state
                    self.max_count = self.list_count[self.list_states.index(self.current_state)]

            # update best solution if opt_solution == OptSolution.MEAN
            elif self.opt_solution == OptSolution.MEAN:
                index_best_mean = np.argmin(self.list_mean)
                self.best_energy = self.list_mean[index_best_mean]
                self.best_state = self.list_states[index_best_mean]

            # RANDOM STUFF
            # persist some info for later
            self.hist.append([
                self.step,
                mean_proposed_neighbour,
                self.best_energy,
                self.best_state.value])

            # update step and also update l if opt_solution == OptSolution.COUNT
            self.step += 1
            if self.opt_solution == OptSolution.COUNT:
                self.l = np.floor(np.log(10 + self.step))

        # generate some final stats
        self.acceptance_rate = self.accept / self.step

        # recalculating the mean for the best state with 100 observations for both variants v1 and v2
        num_final_observations = 100
        sum_best_energy = 0
        if self.opt_solution == OptSolution.COUNT:
            for i in range(num_final_observations):
                sum_best_energy += self.cost_func(self.best_state.value, **kwargs)
            self.best_energy = sum_best_energy / num_final_observations

        elif self.opt_solution == OptSolution.MEAN:
            index_best_state = self.list_states.index(self.best_state)
            if self.list_count[index_best_state] < num_final_observations:
                for i in range(num_final_observations - self.list_count[index_best_state]):
                    sum_best_energy += self.cost_func(self.best_state.value, **kwargs)
                self.list_mean[index_best_state] = (self.list_mean[index_best_state] * self.list_count[
                    index_best_state] + sum_best_energy) / num_final_observations
                self.list_count[index_best_state] = num_final_observations
                self.best_energy = self.list_mean[index_best_state]

        self.end_time = time.time()

    def move_discrete(self):
        """
        calculates a random neighbour next to the current state

        :return: neighbour in form of vector
        """
        neighbour = np.array(self.current_state.value)

        while State(neighbour) == self.current_state:
            for i in range(np.size(neighbour)):
                if neighbour[i] in range(self.bounds[0] + 1, self.bounds[1]):  # item not equal upper/lower bound
                    neighbour[i] += randint(-1, 1)
                elif neighbour[i] == self.bounds[0]:
                    neighbour[i] += randint(0, 1)
                elif neighbour[i] == self.bounds[1]:
                    neighbour[i] += randint(-1, 0)

        # print(neighbour)
        return State(neighbour)

    def move_random(self):
        """
        calculates a random neighbour

        :return: neighbor in form of vector
        """
        neighbour = self.current_state
        while neighbour == self.current_state:
            neighbour = State(np.random.randint(low=self.bounds[0], high=self.bounds[1] + 1, size=len(self.x0.value)))

        return neighbour

    def normalizer(self, state):
        """


        :param state: state to be normalized
        :return: normalizer value
        """
        norm = 1
        for i in range(np.size(state.value)):
            if state.value[i] == self.bounds[0] or state.value[i] == self.bounds[1]:
                norm *= 2
            else:
                norm *= 3
        return norm - 1

    def results(self):
        """
        print the results

        """
        print('+------------------------ RESULTS -------------------------+\n')
        print(f'      opt. solution: {self.opt_solution}')
        print(f'neighbourhood func.: {self.neighbourhood_func}\n')

        print(f'        final state: {self.best_state}')
        # minus in front of the best energy because our algorithm is coded for minimization
        # but our simulation wants to be maximized
        print(f'       final energy: {-1 * self.best_energy}')
        print(f'    acceptance rate: {self.acceptance_rate}')
        print(f'        runtime [s]: {self.end_time - self.start_time}')
        print('+-------------------------- END ---------------------------+')
