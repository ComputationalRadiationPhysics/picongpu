Pusher: Testing its functionality in PIConGPU
=============================================

An electron with a velocity of 0.5c moves perpendicular to a homogeneous magnetic field (in the z-direction).
The electron should, therefore, move in a perfect circle since there is no acceleration in the direction of movement.
The test compares the radii of a particle simulated by PIConGPU. 
The radius is calculated by the position of the particle and its momentum (r =  p / (q * B) ).
If the relative change between two periods is greater than epsilon_position = 5e-5 or epsilon_momentum = 1e-5 respectively, the test fails.
The calculation of the radius with the position of the particle has a higher error than the calculation with the momentum.
Therefore the test compares the change of the radius against two different epsilons.
Also, the absolute phase change during one turn is regarded. 
It should be smaller than delta = 0.16 rad.

the acceptance thresholds for position-, momentum- and phase-error were determined as follows:

epsilon: The greatest change in the radius measured between 2 turns (2000 Turns simulated) is approximately 4e-6 for the calculation with the momentum.
         For the calculation with the position of the particle, this value is approximately 2.5e-5.
         Those values doubled yield the two epsilons as an approximation 
         of the maximal uncertainty PIConGPU should have. Therefore, the test uses this value to check the radius change.

delta: The measured phase difference between 2 revolutions is approximately 0.08rad. This doubled is the maximal error to be accepted for the phase difference between 2 turns.

PIConGPU run September 2023; 50 steps per turn.
