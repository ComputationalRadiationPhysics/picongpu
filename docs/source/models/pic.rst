.. _model-pic:

The Particle-in-Cell Algorithm
==============================

.. sectionauthor:: Axel Huebl, Klaus Steiniger

Please also refer to the textbooks [BirdsallLangdon]_, [HockneyEastwood]_, our :ref:`latest paper on PIConGPU <usage-reference>` and the works in [Huebl2014]_ and [Huebl2019]_ .

System of Equations
-------------------

.. math::

   \nabla \cdot \mathbf{E} &= \frac{1}{\varepsilon_0}\sum_s \rho_s

   \nabla \cdot \mathbf{B} &= 0

   \nabla \times \mathbf{E} &= -\frac{\partial \mathbf{B}} {\partial t}

   \nabla \times \mathbf{B} &= \mu_0\left(\sum_s \mathbf{J}_s + \varepsilon_0 \frac{\partial \mathbf{E}} {\partial t} \right)

for multiple particle species :math:`s`.
:math:`\mathbf{E}(t)` represents the electric, :math:`\mathbf{B}(t)` the magnetic, :math:`\rho_s` the charge density and :math:`\mathbf{J}_s(t)` the current density field.

Except for normalization of constants, PIConGPU implements the governing equations in SI units.

Relativistic Plasma Physics
---------------------------

The 3D3V particle-in-cell method is used to describe many-body systems such as a plasmas.
It approximates the Vlasov--Maxwell--Equation

.. math::
   :label: VlasovMaxwell

   \partial_t f_s(\mathbf{x},\mathbf{v},t) + \mathbf{v} \cdot \nabla_x f_s(\mathbf{x},\mathbf{v},t) + \frac{q_s}{m_s} \left[ \mathbf{E}(\mathbf{x},t)  + \mathbf{v} \times \mathbf{B}(\mathbf{x},t) \right] \cdot \nabla_v f_s(\mathbf{x},\mathbf{v},t) = 0

with :math:`f_s` as the distribution function of a particle species :math:`s`, :math:`\mathbf{x},\mathbf{v},t` as position, velocity and time and :math:`\frac{q_s}{m_s}` the charge to mass-ratio of a species.
The momentum is related to the velocity by :math:`\mathbf{p} = \gamma m_s \mathbf{v}`.

The equations of motion are given by the Lorentz force as

.. math::

   \frac{\mathrm{d}}{\mathrm{d}t} \mathbf{V_s}(t) &= \frac{q_s}{m_s}  \left[ \mathbf{E}(\mathbf{X_s}(t),t) + \mathbf{V_s}(t) \times \mathbf{B}(\mathbf{X_s}(t),t) \right]\\
   \frac{\mathrm{d}}{\mathrm{d}t} \mathbf{X_s}(t) &= \mathbf{V_s}(t) .

.. attention::

   TODO: write proper relativistic form

:math:`\mathbf{X}_s = (\mathbf x_1, \mathbf x_2, ...)_s` and :math:`\mathbf{V}_s = (\mathbf v_1, \mathbf v_2, ...)_s` are vectors of *marker* positions and velocities, respectively, which describe the ensemble of particles belonging to species :math:`s`.

.. note::

   Particles in a particle species can have different charge states in PIConGPU.
   In the general case, :math:`\frac{q_s}{m_s}` is not required to be constant per particle species.

Electro-Magnetic PIC Method
---------------------------

**Fields** such as :math:`\mathbf{E}(t), \mathbf{B}(t)` and :math:`\mathbf{J}(t)` are discretized on a regular mesh in Eulerian frame of reference (see [EulerLagrangeFrameOfReference]_).
See :ref:`section Finite-Difference Time-Domain Method <model-AOFDTD>` describing how Maxwell's equations are discretized on a mesh in PIConGPU.

The distribution of **Particles** is described by the distribution function :math:`f_s(\mathbf{x},\mathbf{v},t)`.
This distribution function is sampled by *markers*, which are commonly referred to as *macroparticles*.
These markers represent blobs of incompressible phase fluid moving in phase space.
The temporal evolution of the distribution function is simulated by advancing the markers over time according to the Vlasov--Maxwell--Equation in Lagrangian frame (see eq. :eq:`VlasovMaxwell` and [EulerLagrangeFrameOfReference]_).
A marker has a finite-size and a velocity, such that it can be regarded as a cloud of particles, whose center of mass is the marker's position and whose mean velocity is the marker's velocity.
The cloud shape :math:`S^n(x)` of order :math:`n` of a marker describes its charge density distribution.
See :ref:`section Hierarchy of Charge Assignment Schemes <model-shapes>` for a list of available marker shapes in PIConGPU.

References
----------

.. [EulerLagrangeFrameOfReference]
        *Eulerian and Lagrangian specification of the flow field.*
        https://en.wikipedia.org/wiki/Lagrangian_and_Eulerian_specification_of_the_flow_field

.. [BirdsallLangdon]
        C.K. Birdsall, A.B. Langdon.
        *Plasma Physics via Computer Simulation*,
        McGraw-Hill (1985),
        ISBN 0-07-005371-5

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
