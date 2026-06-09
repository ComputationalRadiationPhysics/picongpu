.. SPDX-FileCopyrightText: PIConGPU contributors
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _model-pic:

The Particle-in-Cell Algorithm
==============================

.. sectionauthor:: Axel Huebl, Klaus Steiniger, Fabia Dietrich

Please also refer to the textbooks [BirdsallLangdon]_, [HockneyEastwood]_, our :ref:`latest paper on PIConGPU <usage-reference>` and the works in [Huebl2014]_ and [Huebl2019]_ .

Maxwell's Equations
-------------------

.. math::
   :label: Maxwell

   \nabla \cdot \mathbf{E} &= \frac{1}{\varepsilon_0}\sum_s \rho_s

   \nabla \cdot \mathbf{B} &= 0

   \nabla \times \mathbf{E} &= -\frac{\partial \mathbf{B}} {\partial t}

   \nabla \times \mathbf{B} &= \mu_0\left(\sum_s \mathbf{J}_s + \varepsilon_0 \frac{\partial \mathbf{E}} {\partial t} \right)

for multiple particle species :math:`s`.
:math:`\mathbf{E}(t)` represents the electric, :math:`\mathbf{B}(t)` the magnetic, :math:`\rho_s` the charge density and :math:`\mathbf{J}_s(t)` the current density field.

Except for normalization of constants, PIConGPU implements the governing equations in SI units.

Vlasov--Maxwell equation
---------------------------

The 3D3V particle-in-cell algorithm is used to describe many-body systems such as plasmas.
It simulates its behaviour by calculating the motion of its electrons and ions in the present electromagnetic fields with the Vlasov--Maxwell equation

.. math::
   :label: VlasovMaxwell

   \partial_t f_s(\mathbf{x},\mathbf{p},t) + \mathbf{v} \cdot \nabla_x f_s(\mathbf{x},\mathbf{p},t) + q_s \left[ \mathbf{E}(\mathbf{x},t)  + \frac{\mathbf{v}}{c} \times \mathbf{B}(\mathbf{x},t) \right] \cdot \nabla_p f_s(\mathbf{x},\mathbf{p},t) = 0

with :math:`f_s` as the distribution function of a particle species :math:`s`, :math:`\mathbf{x},\mathbf{p},t` as position, momentum and time and :math:`{q_s},{m_s}` the charge and mass of a species.
The velocity is related to the momentum by

.. math::
    \mathbf{v} = \frac{c \mathbf{p}}{\sqrt{(m_s c)^2 + \abs{p}^2}}.


Electro-Magnetic PIC Method
---------------------------

The distribution of **particles** is described by the distribution function :math:`f_s(\mathbf{x},\mathbf{p},t)` and tracked in a continuous 6D phase space (or 5D, for 2D3V simulations) :math:`(\mathbf{r},\mathbf{p})`.
As the number of particles involved in typical plasma simulations is usually extremely large, they are combined to so-called *macroparticles*, containing up to millions of particles of a species :math:`s`. 
Those macroparticles sample the distribution function :math:`f_s` by representing blobs of incompressible phase fluid with finite size and velocity moving in phase space.
Since the Lorentz force only depends on the charge-to-mass ratio :math:`q_s / m_s`, a macroparticle follows the same trajectory as an individual particle of the same species would. 

.. note::

   Particles in a particle species can have different charge states in PIConGPU.
   In the general case, :math:`\frac{q_s}{m_s}` is not required to be constant per particle species.

The temporal evolution of :math:`f_s` is simulated by advancing the macroparticles over time with :eq:`VlasovMaxwell`.

The charge of the macroparticles is assigned to neighboring mesh points with help of an *assignment function* :math:`W(\mathbf{x})`.
These assignment functions are called *shapes* in PIConGPU.
See :ref:`section Hierarchy of Charge Assignment Schemes <model-shapes>` for a list of available macroparticle shapes in PIConGPU.

The **fields** :math:`\mathbf{E}(t)` and :math:`\mathbf{B}(t)` are discretized on a regular mesh in Eulerian frame of reference (= static, see [EulerLagrangeFrameOfReference]_).
Their temporal evolution is computed with the :ref:`Finite-Difference Time-Domain (FDTD) method <model-AOFDTD>`, meaning that the partial space
and time derivatives occurring in Maxwell’s equations are approximated by centered finite differences. 
In this approach, the derivatives of field values are computed at positions intermediate to those where the field values are explicitly known. 
This results in a staggered grid arrangement, where the electric and magnetic field are offset by half a grid cell in space and half a time step.

The basic PIC cycle, representing a single simulation time step :math:`(n \to n + 1)`, comprises four consecutive steps, which are computed for all macroparticles involved in the simulation:

#. *Field to particle:* The fields :math:`\mathbf{E}^n`, :math:`\mathbf{B}^n` are interpolated onto the particle position :math:`\mathbf{x}^n` with help of the assignment function :math:`W`.
#. *Particle push:* By integrating the equation of motion obtained from the Lorentz force acting on the macroparticle, its change in position :math:`\mathbf{x}` and velocity :math:`\mathbf{u} = \gamma \mathbf{v}` is obtained:

   .. math::
       \mathbf{u}^{n+3/2} = \mathbf{u}^{n+1/2} + \Delta t \frac{q}{m} (\mathbf{E}^{n} + \bar{\mathbf{v}}^{n} \times \mathbf{B}^{n})
   .. math::
       \mathbf{x}^{n+1} = \mathbf{x}^n + \Delta t ~\mathbf{u}^{n+1/2}
   with :math:`\bar{\mathbf{v}}^{n}` as an approximated velocity at :math:`t = n`.
   There exist various particle pushers, differing in accuracy and performance and offering different approximation schemes for :math:`\bar{\mathbf{v}}^{n}`. Please refer to [Ripperda2018]_ for further information.
   The particle pusher can be chosen and configured in :ref:`pusher.param <usage-params-core>`.
#. *Current deposition:* The current :math:`\mathbf{J}` caused by the macroparticle movement is computed on the grid by solving the continuity equation of electrodynamics:

   .. math::
       \rho^{n+1} = \rho^n + \Delta t \cdot \nabla \mathbf{J}^{n + 1/2}
   The charge densities :math:`\rho^{n}, \rho^{n+1}` are obtained from the macroparticle position before and after movement with help of the assignment function :math:`W`. 
   In PIConGPU, the user can choose between Esirkepov's current deposition method [Esirkepov2001]_ and the performance-increased EZ method [Steiniger2023]_. 
   See :ref:`here <usage-params-core-currentdeposition>` for further details.
#. *Field update*: By inserting the current :math:`\mathbf{J}`, the electromagnetic fields can be updated by applying the third and fourth Maxwell equation :eq:`Maxwell` 
   using the :ref:`FDTD method <model-AOFDTD>`. 
   Thereby, the magnetic field update is split in two parts of each half a timestep, so that electric and magnetic fields are known at the same time. 
   This is necessary to caclulate the correct Lorentz force in the next run of the PIC cycle.

By running the cycle multiple times, a longer simulation duration is achieved. 

The field update requires only information from neighbouring or next-to neighbouring cells, depending on the FDTD scheme involved. 
This locality allows a parallelized computation of the algorithm by sub-dividing the simulation volume and mapping the underlying mesh onto local memory subsets, which can then be assigned to a single processing unit. 
Data transfer between processing units is required at every time step to transfer field data between neighbouring cells and when particles cross cell boundaries. 

The following flowchart provides a summary and visualization of the implementation of the PIC cycle within PIConGPU.

.. image:: media/picongpu-main-loop.svg
   :width: 700
   :alt: PIC cycle in PIConGPU

References
----------

.. [BirdsallLangdon]
        C.K. Birdsall, A.B. Langdon.
        *Plasma Physics via Computer Simulation*,
        McGraw-Hill (1985),
        ISBN 0-07-005371-5

.. [EulerLagrangeFrameOfReference]
        *Eulerian and Lagrangian specification of the flow field.*
        https://en.wikipedia.org/wiki/Lagrangian_and_Eulerian_specification_of_the_flow_field

.. [Esirkepov2001]
        T.Zh. Esirkepov,
        *Exact charge conservation scheme for particle-in-cell simulation with an arbitrary form-factor*,
        Computer Physics Communications 135.2 (2001): 144-153,
        `DOI:10.1016/S0010-4655(00)00228-9 <https://doi.org/10.1016/S0010-4655(00)00228-9>`_

.. [HockneyEastwood]
        R.W. Hockney, J.W. Eastwood.
        *Computer Simulation Using Particles*,
        CRC Press (1988),
        ISBN 0-85274-392-0

.. [Huebl2014]
        A. Huebl.
        *Injection Control for Electrons in Laser-Driven Plasma Wakes on the Femtosecond Time Scale*,
        Diploma Thesis at TU Dresden & Helmholtz-Zentrum Dresden - Rossendorf for the German Degree "Diplom-Physiker" (2014),
        `DOI:10.5281/zenodo.15924 <https://doi.org/10.5281/zenodo.15924>`_

.. [Huebl2019]
        A. Huebl.
        *PIConGPU: Predictive Simulations of Laser-Particle Accelerators with Manycore Hardware*,
        PhD Thesis at TU Dresden & Helmholtz-Zentrum Dresden - Rossendorf (2019),
        `DOI:10.5281/zenodo.3266820 <https://doi.org/10.5281/zenodo.3266820>`_

.. [Ripperda2018]
        B. Ripperda et. al.
        *A Comprehensive Comparison of Relativistic Particle Integrators*, 
        The American Astronomical Society (2018), 
        `DOI:10.3847/1538-4365/aab114 <https://dx.doi.org/10.3847/1538-4365/aab114>`_

.. [Steiniger2023]
        K. Steiniger et. al.
        *EZ: An efficient, charge conserving current deposition algorithm for electromagnetic Particle-in-Cell simulations*,
        Computer Physics Communications (2023)
        `DOI:10.1016/j.cpc.2023.108849 <https://doi.org/10.1016/j.cpc.2023.108849>`_
