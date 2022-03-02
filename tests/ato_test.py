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

import numpy as np

from sa4ato.assemble_to_order import ATOSystem
from sa4ato.assemble_to_order import ato


def test_ato_profit():
    """
    tests the assemble-to-order simulation with given base inventory
    """
    # np.array([2, 6, 5, 2, 3, 4, 3, 9, 6, 5, 6, 4])
    # np.array([3, 4, 3, 7, 5, 6, 4, 3])
    profit = -ato(np.array([5, 5, 5, 5, 5, 5, 5, 5]), ato_system=ATOSystem.MODERATELY)
    print('Profit: ', profit)


def test_ato_profit_random_base_inventory():
    """
    tests the assemble-to-order simulation with random initialized base inventory
    """
    profit = -ato(np.random.randint(0, 31, 12), t_warm_up=20, t_max=70)
    print('Profit: ', profit)
