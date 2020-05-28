==============
Atomic Physics
==============

Overview
--------

    Definitions:

    I will not distinguish between atoms and ions, both will be refered to as
    ions irrespective of their actual charge, since both are handeled principally
    equally in PIConGPU.
    In addition I am going to denote an internal state of ions as atomic state,
    in contrast to the "external" charge state.

This module in PIConGPU covers excitations/deexcitations and ionization of bound
electrons in plasmas.

A bound electron in this context means an electron bound to a specific ion.
Each bound electron occupies a quantum states inside the ions potential well.
The the occupied electron states define a electron hull configuration of the ion,
denoted to as the atomic state of this ion.

An ions atomic state changes due to
 - excitations
 - deexcitations
 - ionization 
 - capture
of electrons, caused by interactions with photons, electrons or randomly.
A combined rate of change between given atomic states is used to describe the
time development, with each such rate dependent on the energy distribution of
interaction partners, the crosssection of a given process and the current atomic
state. These rates form the rate equation, describing the time evolution of a
atomic state distribution in a matrix differntial equation, the rate equation.
The interaction partner energy distributions are taken from the existing PIC
simulation, while an atomic state databank provides the random transition rates.

To modell the time evolution we solve the rate matrix in space and time starting
from a specified intial condition.


Electron States
---------------

The actual electron states encountered in a plasma are rather difficult to calculate,
since external fields, neighboring ions, free electrons and the bound electrons
itself distort the ion core potential.
The resulting atomic potential may vary on very short time- and space scales,
which are not resolved in PIC-simulations and computationally expensive to solve.

To avoid this, we are using the analytically known hydrogen electron states as
the basis of our description and include mean field energy corrections to account
for bound electrons, the so called screened hydrogen states.
The time dependent influences of external fields and free electrons are modelled
by random transitions between atomic states. This accounts for the fact that
used state basis we are using does not consist of eigen vectors of our
Hamiltonian and therefore our asusmed states are not stable in time.
To calculate these random rates we are using semi empirical fits of the tranitions
rates observed in plasmas.

Atomic States
-------------

Instead of the actual screened hydrogen electron configuration, we are using the
occupation numbers as the atomic state, a so called super configuration.
This reduces the memory required to store the atomic state considerably and allows
us ot include the atomic state directly in PIConGPU in the first place.
Occupation numbers have been choosen, since the energy of screened hydrogen states
only does depend on occupation numbers.
Grouping atomic states by energy is therefore equal to gruping by occupation number.

The atomic states are stored macro particle based, meaning the atomic state is
bound to macro particle in the PIC simulation.
This reflects the fact that atomic states are a property of ions and therfore
attached to the them, allowing a native transport description by following the
macro particle trajectories.
Every macro particle only stores one atomic states, instead of a distribution,
in ordr to require little engough memory to make this storage viable. The actual
atomic state distirbution is assumed to be sufficiently resolved in each super
cell.