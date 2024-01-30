PusherScaling: Testing the functionality of the Pushers in PIConGPU
===================================================================

An electron with a velocity of 0.5c moves perpendicular to a homogeneous magnetic field (in the z-direction).
The electron should, therefore, move in a perfect circle since there is no acceleration in the direction of movement.
According to B. Ripperda et al. (A Comprehensive Comparison of Relativistic Particle Integrators; https://doi.org/10.3847/1538-4365/aab114),
there is a numerical phase lag d(phi) proportional to the timestep square dt² (Boris Pusher).
The test simulates the described setup for 10, 20, 40, 80, and 160 steps per turn. The phase lag d(phi) is computed for each scenario.
Because we double the timestep each time (starting with 160 steps per turn), the phase lag should grow with a factor of 4.
The test checks this factor by calculating the quotient of the phase lags between two timestep values. It then calculates the exponent x
of d(phi) = 2^x (which should be 2) and the standard deviation of x (which should be zero). The exponent should not differ by more than
epsilon = 0.1, and the standard deviation should not exceed delta = 0.05 from the ideal values.

The thresholds epsilon and delta were determined as follows:
---------------------------------------------------------------
epsilon = 0.1 (for the exponent of d(phi) = 2^x, which should be 2)
delta = 0.05 (for the standard deviation of x)

They double the deviation found in recent PIConGPU runs (2023-09-07).
So the results of PIConGPU's simulations should not exceed these thresholds.
