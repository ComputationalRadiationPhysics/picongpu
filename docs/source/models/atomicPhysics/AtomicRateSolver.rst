===========================
Atomic rate equation solver
===========================

In principle what we want to solve the rate equation (1) for every grid point and
time step of our simultation.

    \frac{\delta n}{\delta t}= R \cdot \vector{n} (1)

We could do this using the standard matrix equation solver and be done with it,
but doing so is impractical.

  We would have to collate all atomic states of all macro ions into the atomic
  population vector, solve the rate equation and lastly split the resulting new
  population distribution amongst the macro ions.

  This is impractical since it is not parallel in macro ions and the macro ions
  posess differing weights,

    We want to initialise each cell with the same number of macro particles to get
    a somewhat constant phase space resolution. The density is then realised
    by varying the macro particle weight, resulting in differing weights of macro
    ions dependeing on their spatial origin.

  making splitting the distribution among macro ions difficult.

To avoid this we implemented the, the rate equation underlying, time evolution of
the atomic states directly. This allows us to reproduce the rate equation in the
macro ion population, without binding us to the atomic population vector detour.

The time evolution is modelled as a chain of transitions, starting with the
current atomic state, through zero to many randomly choosen intermediary atomic
states, to a final atomic state, reached when the PIC time step interval has
been used up by the preceding transition. This final state in turn becomes the
initial atomic state of the next PIC-step.

The time each transition uses up is equal to the inverse of the rate of this
transition, mean time between two such transitions. If the remaining time of our
current PIC time step is smaller than the time required the ratio between them
is used as a propability to decide wether a transition takes place. If the
transition takes place we have found the final state, if no transition takes
place a new transition is choosen randomly.

Transitions/intermediary states are choosen from all existing atomic states,
with each atomic state having equal probablity.
