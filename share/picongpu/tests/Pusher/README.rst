Pusher: Testing its functionality in PIConGPU
=============================================

An electron with a velocity of 0.5c moves perpendicular to a homogeneous magnetic field (in the z-direction).
The electron should, therefore, move in a perfect circle since there is no acceleration in the direction of movement.
The test compares the radii of a particle simulated by PIConGPU. If the relative change between two periods is greater than 
epsilon = 1e-5, the test fails. Also, the absolute phase change during one turn is regarded. It should be smaller than delta = 0.25 rad.

The thresholds epsilon and delta were determined as follows:

epsilon: The greatest change in the radius measured between 2 turns (2000 Turns simulated) is approximately 4e-6. This doubled yields epsilon as an approximation 
-------- of the maximal uncertainty PIConGPU should have. Therefore, the test uses this value to check the radius change.

delta: The measured phase difference between 2 revolutions is approximately 0.125rad. This doubled is the maximal error to be accepted for the phase difference between 2 turns. 
------

PIConGPU run September 2023; 50 steps per turn.
