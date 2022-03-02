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


import random

import numpy as np

from sa4ato import discrete_stochastic_simulated_annealing as sa
from sa4ato.assemble_to_order import ATOSystem
from sa4ato.assemble_to_order import ato


def test_sa_on_ato():
    """

    """
    np.random.seed(0)
    random.seed(0)
    x0 = np.random.randint(0, 21, 8)
    opt = sa.Minimize(ato, x0, opt_solution=sa.OptSolution.MEAN,
                      neighbourhood_func=sa.NeighbourhoodFunc.DIRECT_NEIGHBOURHOOD, t=1, bounds=[0, 20],
                      step_max=10, ato_system=ATOSystem.MODERATELY)

    opt.results()
