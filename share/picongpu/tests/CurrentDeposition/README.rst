CurrentDeposition: Testing its functionality in PIConGPU
========================================================

An electron with a velocity of 0.999c moves one time-step in a defined direction. The Pusher is free, so there is no interaction between the particle and the fields.
The test compares the results of the current density field simulated by PIConGPU j(PIConGPU), and the one calculated by a Python reference implementation j(Python).
Both results are given in the units used by PIConGPU for an adequate evaluation with the help of PIConGPU's numerical uncertainty.
Therefore abs(j(PIConGPU) - j(Python)) is calculated and compared against epsilon = 1e-5, which is slightly higher than PIConGPU numerical uncertainty.
