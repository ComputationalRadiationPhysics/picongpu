```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
%matplotlib inline

import sys
```

```python
sys.path.append("./modules4picongpu/")

from bunchInit import vec3D
from bunchInit import addParticles2Checkpoint
```

# Set up the parameters for the electron bunch

```python
# define values

# grid values
delta_t = 1.6674254e-17
delta_x = 8.120995783129748e-8 * 4.0
delta_y = 5.0e-9

# number of cells
N_cellss = (512, 3840, 512)

# iterations till PIConGPU has inialized the fields
N_init_timesteps = 200000

# characteristic parameters of the electron bunch
mean_kinetic_energy = 15.0e6 # [eV]
norm_trans_emmitt_notSI = 0.5 # [mm mrad]
rel_energy_spread = 0.008
focus_position = 5.66e-3 + N_init_timesteps * delta_t * constants.speed_of_light # m
bunch_cross_section_radius = 7.0e-6 # m (rms)
bunch_duration = 17.5e-15 # s (rms)
bunch_charge = 35.0e-12 # C

```

```python
# compute position in simulation box
center_pos_x = delta_x * N_cells[0]/2.
center_pos_z = delta_x * N_cells[2]/2.
y_center = 5.5e-6 + focus_position

# number of macro particles in simulation
N_particles = np.int(bunch_charge / (100.*constants.elementary_charge) )

```

```python
# define helper functions

def gammaFromEkin_eV(Ekin_eV):
    Ekin_J = Ekin_eV * constants.elementary_charge
    return Ekin_J/(constants.electron_mass * constants.speed_of_light**2) +1

def betaFromGamma(gamma):
    return np.sqrt(1.0 - 1.0/gamma**2)

def momentumFromGammaBeta(gamma, beta):
    return constants.electron_mass * constants.speed_of_light * beta * gamma

```

```python
# derive values of electron bunch

# longitudinal
norm_trans_emmitt =  norm_trans_emmitt_notSI *1.e-3 * 1e-3 # m rad
t_delay = - focus_position / constants.speed_of_light

mean_gamma = gammaFromEkin_eV(mean_kinetic_energy)
mean_beta = betaFromGamma(mean_gamma)

# transversal
sigma_x = bunch_cross_section_radius
sigma_z = bunch_cross_section_radius

sigma_xp = norm_trans_emmitt / (mean_beta * mean_gamma * sigma_x)
sigma_zp = norm_trans_emmitt / (mean_beta * mean_gamma * sigma_z)

print("mean gamma = {:.2f}".format(mean_gamma))
```

```python
# initialize bunch

# weighting
weighting = np.ones(N_particles) * bunch_charge/(N_particles*constants.elementary_charge)

# distribution in bunch focus position (no coralation between momenta und positions)
focus_x = np.random.normal(loc=center_pos_x, scale=sigma_x, size=N_particles + N_particles//4)
focus_z = np.random.normal(loc=center_pos_z, scale=sigma_z, size=N_particles + N_particles//4)

focus_xp = np.random.normal(loc=0.0, scale=sigma_xp, size=N_particles + N_particles//4)
focus_zp = np.random.normal(loc=0.0, scale=sigma_zp, size=N_particles + N_particles//4)

focus_t = np.random.normal(loc=0.0, scale=bunch_duration, size=N_particles + N_particles//4)

gamma = np.random.normal(loc=mean_gamma, scale=rel_energy_spread*mean_gamma, size=N_particles + N_particles//4)
beta = betaFromGamma(gamma)

# filer: allow only particles within the 2*sigma region
my_filter = np.less(np.sqrt((((focus_x - center_pos_x)/sigma_xp)**2 + 
                             ((focus_z - center_pos_z)/sigma_xp)**2) +
                             (focus_t/bunch_duration)**2
                           ),
                   2.0 )

# derive beta
beta_x = focus_xp * beta
beta_z = focus_zp * beta
beta_y = np.sqrt(beta**2 - beta_x**2 - beta_z**2)

# generate momentum for checkpoint
p_x = (momentumFromGammaBeta(gamma, beta_x)[my_filter])[:N_particles]
p_y = (momentumFromGammaBeta(gamma, beta_y)[my_filter])[:N_particles]
p_z = (momentumFromGammaBeta(gamma, beta_z)[my_filter])[:N_particles]

# generate position for checkpoint
x = ((focus_x + t_delay * constants.speed_of_light * beta_x)[my_filter])[:N_particles]
z = ((focus_z + t_delay * constants.speed_of_light * beta_z)[my_filter])[:N_particles]
y = ((y_center + focus_t * constants.speed_of_light + t_delay * constants.speed_of_light * beta_y)[my_filter])[:N_particles]


```

```python
# plot spatial particle position

plt.title("x position")
TMP = plt.hist(x, bins=128)
plt.show()

plt.title("z position")
TMP = plt.hist(z, bins=128)
plt.show()

plt.title("y position")
TMP = plt.hist(y, bins=128)
plt.show()

```

```python
# convert to vec3D objects

pos = vec3D(x,y,z)
mom = vec3D(p_x, p_y, p_z)
```

# Manipulate PIConGPU checkpoint for field generation

Add the particles to a PIConGPU checkpoint without electromagnetic fields.

 - Generate a PIConGPU simulation with the same extend as the final simulation,
without particles in `./runs/bunch_init`.
 - Use the `particles::pusher::Free` pusher for the (not yet existing)
particles.
 - Run this simulation for one iteration to generate a 0th check point.
 - Add particles with the code below to this check point.
 - Run the simulation for ~200000 iterations



Runing PIConGPU  using the `particles::pusher::Free` pusher the particles are
not effected by missing or wrong electromagetic fields.
Due to their current deposition, the fields around the bunch are created and
approche the correct fields distribution with more and more iterations of the
PIC cycle.

After the initalization simulation is finished, the particle and field
distribution can be used for manupulationg the chekpoint of a PIConGPU
simulation using a physically correct pusher (`particles::pusher::Free`,
`particles::pusher::Boris`, ...)



```python
checkPoint = addParticles2Checkpoint("./runs/bunch_init/simOutput/checkpoints/checkpoint_0.h5")
```

```python
checkPoint.addParticles(pos, mom, weighting)

checkPoint.writeParticles()
```

Now **run the PIConGPU simulation** in `./runs/bunch_init`

# Manipulate PIConGPU checkpoint for final simulation

Add the previously computed particle and field distribution to the final
simulation.

 - Generate a PIConGPU simulation without particles in `/runs/real_simulation`
 - Use a physically correct pusher for the bunch particles.
 - Run this simulation for one iteration to generate a 0th check point.
 - copy checkpoint from hdf5 data output
 - run simulation

```python
checkPoint = addParticles2Checkpoint("./runs/real_simulation/simOutput/checkpoints/checkpoint_0.h5")
```

```python
checkPoint.copyCheckpoint("./runs/bunch_init/simOutput/h5/simData_200000.h5")
```

Now **run the PIConGPU simulation** in `./runs/real_simulation`

```python

```
