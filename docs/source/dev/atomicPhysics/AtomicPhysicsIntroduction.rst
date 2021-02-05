==============
Atomic Physics
==============

Overview
--------

    Definitions:
    ------------

    I will not distinguish between atoms and ions, both will be refered to as
    ions irrespective of their actual charge, since both are handeled principally
    equally in PIConGPU.
    In addition I am going to denote an internal state of ions as atomic state,
    in contrast to the "external" charge state.

This module of PIConGPU covers excitations/deexcitations and ionization of bound
electrons in plasmas.

A bound electron in this context describes an electron bound to a specific ion.
Every bound electron occupies a quantum state, also called electron state,
inside the ions potential well, think quantum mechanical hydrogen modell.
The atomic state of an ion denotes the electron hull configuration of this ion,
the entirety of occupied electron states.

To describe a large number of ions, we use the number density of a given atomic
state. Different atomic states can be collected into an atomic state distribution
, the density of each possible atomic state, in a atomic state population
vector \vector{n}.

Electron States
---------------

The actual electron states encountered in a plasma are rather difficult to calculate,
since external fields, neighboring ions, free electrons and the bound electrons
itself distort the ion core potential.
The resulting atomic potential may vary on very short time and space scales,
which are not resolved in PIC-simulations and computationally expensive to solve.

To avoid this, we are using the analytically known hydrogen electron states as
the basis of our description and include mean field energy corrections to account
for bound electrons, the so called screened hydrogen states.
The time dependent influences of external fields and free electrons are modelled
by random transitions between atomic states. This accounts for the fact that the
used state basis does not consist of eigen vectors of our Hamiltonian and
therefore are not stable in time.

Dynamics of atomic states
-------------------------

The atomic state of an ion may changes due to
 - excitations:

   one or more electrons change their electron state, increasing the overall
   energy of the atomic state

 - deexcitations:

   the same as excitaion, but decreasing the overall energy

 - ionization:

   one or more electron is freed from the atomic potential, leaving their electron
   states empty.

 - capture:

   free electrons are captured by the ion and now occupy a electron state

These processes can be caused by interactions with photons, electrons or even occur randomly and can be assigned a rate of change between given atomic states.

    R_ij ... rate of change from state j to i

Based on their cause the different contributions to rates can be classified as
either interaction based or spontanous.

An atomic state database provides the random transition rates.

Interaction based contributions depend on the interaction partners energy, mostly,
and the ions crossection for a given process, while spontanous processes are
assumed to only depend on the current ion atomic state.

The interaction partner energy distributions are taken from the existing PIC
simulation.

These rates are combined into the rate equation, describing the time evolution
of an atomic state distribution in a matrix differntial equation.

    \frac{\delta n}{\delta t}= R \cdot \vector{n}

To modell the time evolution we solve the rate matrix in space and time starting
from a specified intial condition directly.

Atomic states in PIConGPU
-------------------------

Instead of the actual screened hydrogen electron configuration, we are using the
occupation number vector as the atomic state, a so called super configuration.

This reduces the memory required to store the atomic state considerably and allows
us to include the atomic state directly in PIConGPU in the first place.
Occupation numbers have been choosen, since the energy of screened hydrogen states
only does depend on occupation numbers. Grouping atomic states by energy is
therefore equal to gruping by occupation number.

The atomic states are stored macro particle based, meaning the atomic state is
bound to macro particle in the PIC simulation.
This reflects the fact that atomic states are a property of ions and therefore
attached to the them, allowing a native transport description, by following the
macro particle trajectories.
Every macro particle only stores one atomic states, instead of a distribution,
in order to require little enough memory to make storage viable. The actual
atomic state distribution is assumed to be sufficiently resolved in each super
cell.