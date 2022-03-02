"""
    Source:
    Horng, Shih-Cheng und Lin, Shieh-Shing:
    Ordinal optimization based metaheuristic algorithm for optimal inventory policy of assemble-to-order systems.
    In:Applied Mathematical Modelling, Band 42:S. 43–57, 2017.
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


from enum import Enum

import numpy as np


class ATOSystem(Enum):
    LARGE = 'large'
    MODERATELY = 'moderately'


# lambda_j: arrival time, 10 key items, 2 non-key items
products_large_sized = np.array([[1.0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0],
                                 [1.2, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0],
                                 [1.4, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1],
                                 [1.6, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1],
                                 [1.8, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0],
                                 [2.0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1],
                                 [2.2, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0],
                                 [2.4, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1]])

# p_j: Profit, q_j: holding cost, mu_j: avg production time, sigma_j: std production time, C_j: capacity
items_large_sized = np.array([[1, 2, 0.11, 0.03, 30],
                              [2, 2, 0.09, 0.02, 30],
                              [3, 2, 0.18, 0.02, 30],
                              [4, 2, 0.12, 0.03, 30],
                              [5, 2, 0.15, 0.02, 30],
                              [6, 2, 0.13, 0.01, 30],
                              [7, 2, 0.11, 0.03, 30],
                              [8, 2, 0.22, 0.02, 30],
                              [9, 2, 0.17, 0.01, 30],
                              [10, 2, 0.13, 0.02, 30],
                              [11, 2, 0.21, 0.01, 30],
                              [12, 2, 0.15, 0.02, 30]])

items_moderately_sized = np.array([[1, 2, .15, .0225, 20],
                                   [2, 2, .40, .06, 20],
                                   [3, 2, .25, .0375, 20],
                                   [4, 2, .15, .0225, 20],
                                   [5, 2, .25, .0375, 20],
                                   [6, 2, .08, .012, 20],
                                   [7, 2, .13, .0195, 20],
                                   [8, 2, .40, .06, 20]])

products_moderately_sized = np.array([[3.6, 1, 0, 0, 1, 0, 1, 1, 0],
                                      [3, 1, 0, 0, 0, 1, 1, 1, 0],
                                      [2.4, 0, 1, 0, 1, 0, 1, 0, 0],
                                      [1.8, 0, 0, 1, 1, 0, 1, 0, 1],
                                      [1.2, 0, 0, 1, 0, 1, 1, 1, 0]])


def ato(inventory_level: np.ndarray, ato_system: ATOSystem, t_warm_up: int = 20, t_max: int = 70) -> float:
    """
    Assemble to order model function

    :param inventory_level: the starting inventory level as well as the inventory level goal
    :param t_warm_up: periods without tracked statistics
    :param t_max: periods
    :param ato_system: decide which size of ato system you want to use
    :return: negative mean profit over t_max-t_warm_up periods
    """

    # initialize size of ATO System
    if ato_system == ATOSystem.LARGE:
        products = products_large_sized
        items = items_large_sized
        n_key_items = 10
    else:
        products = products_moderately_sized
        items = items_moderately_sized
        n_key_items = 6

    # initialize number of products, items, key and non key
    n_products = len(products)
    n_items = len(items)
    n_nkey_items = n_items - n_key_items

    # checks whether the inventory vector is of the right size
    assert len(inventory_level) == n_items, 'inventory level have to have the same size as number of items'

    inventory = np.array(inventory_level)  # set inventory to initial inventory level

    profit = 0  # total profit
    fulfilled_orders = 0  # number of fulfilled orders

    t_replenishment_order = np.zeros((n_items, 1))  # time of replenishment order for the jth item
    t_previous_order = 0  # time of previous order
    k = 0  # k = index of Product of next order to fulfill

    # Randomly generate the values of first arrival times t i
    arrival_times = np.random.exponential(1 / products[:, 0])
    # arrival_times = np.random.gamma(2, products[:, 0])

    while min(arrival_times) <= t_max:

        min_t = t_max  # Keep track of minimum order time found so far
        for i in range(n_products):
            if arrival_times[i] <= min_t:
                min_t = arrival_times[i]  # Time of next order to fulfill
                k = i  # k = index of Product of next order to fulfill

        # generate time of next order for the kth product
        # arrival_times[k] = arrival_times[k] + np.random.gamma(2, products[k, 0])
        arrival_times[k] = arrival_times[k] + np.random.exponential(1 / products[k, 0])

        # Update inventory levels up to time of next order (order to be fulfilled in this iteration)
        t_replenishment_order.sort(axis=1)
        max_entries = 0
        for i in range(n_items):
            if np.count_nonzero(t_replenishment_order[i, :]) >= max_entries:
                max_entries = np.count_nonzero(t_replenishment_order[i, :])

        size_t_replenishment_order = np.size(t_replenishment_order, axis=1)
        if max_entries > 0:
            t_replenishment_order = np.delete(t_replenishment_order, range(0, size_t_replenishment_order - max_entries),
                                              axis=1)

        size_t_replenishment_order = np.size(t_replenishment_order, axis=1)  # columns/size of t_replenishment_order
        for j in range(n_items):
            for replenishment in range(size_t_replenishment_order):
                # if jth item is ready to use (rj <= min_t), add inventory ready to be used
                if t_replenishment_order[j, replenishment] != 0 and t_replenishment_order[j, replenishment] <= min_t:
                    inventory[j] += 1  # increase inventory xj by one
                    t_replenishment_order[
                        j, replenishment] = 0  # removes replenishment order from t_replenishment_order

        # if all key items are available
        key_items_available = 0
        for i in range(n_key_items):
            if products[k, i + 1] <= inventory[i]:
                key_items_available = key_items_available + 1

        if key_items_available == n_key_items:
            if min_t >= t_warm_up:
                profit += np.dot(products[k, range(1, n_key_items + 1)],
                                 items[range(0, n_key_items), 0])  # update profit

            fulfilled_orders = fulfilled_orders + 1

            # key items
            for i in range(n_key_items):
                if products[k, i + 1] != 0:
                    amount = products[k, i + 1]
                    inventory[i] = inventory[i] - amount  # decrease inventory
                    for a in range(int(amount)):
                        # place replenishment orders for the amount of key items used
                        new_replenishment_order = np.zeros((n_items, 1))
                        new_replenishment_order[i] = np.maximum(min_t,np.max(
                            t_replenishment_order[i, :])) + np.maximum(0,np.random.normal(
                            items[i, 2],items[i, 3]))  # second maximum does the truncation at zero
                        t_replenishment_order = np.hstack((t_replenishment_order, new_replenishment_order))

            # non-key items
            for i in range(n_nkey_items):
                if (products[k, i + n_key_items + 1] <= inventory[i + n_key_items]) and (
                        products[k, i + n_key_items + 1] != 0):
                    if min_t >= t_warm_up:
                        profit += items[n_key_items + i, 0] * products[k, i + n_key_items + 1]  # update profit
                    amount = products[k, i + n_key_items + 1]
                    inventory[i + n_key_items] = inventory[i + n_key_items] - amount  # decrease inventory
                    for a in range(int(amount)):
                        # place replenishment orders for the amount of key items used
                        new_replenishment_order = np.zeros((n_items, 1))
                        new_replenishment_order[i + n_key_items] = np.maximum(min_t, np.max(
                            t_replenishment_order[n_key_items + i, :])) + np.maximum(0, np.random.normal(
                            items[n_key_items + i, 2],
                            items[n_key_items + i, 3]))  # second maximum does the truncation at zero
                        t_replenishment_order = np.hstack((t_replenishment_order, new_replenishment_order))

        # reduce holding costs from profit
        if min_t >= t_warm_up:
            profit = profit - np.dot(inventory, items[:, 1]) * (min_t - t_previous_order)

        t_previous_order = min_t  # Update the time of previous to be current order time

    # minus in front of profit, because our Simulated Annealing is code for minimization
    return -(profit / (t_max - t_warm_up))
