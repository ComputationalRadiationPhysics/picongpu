.. _model-fieldIonization:

Field Ionization
================

.. sectionauthor:: Marco Garten
.. moduleauthor:: Marco Garten

Get started here https://github.com/ComputationalRadiationPhysics/picongpu/wiki/Ionization-in-PIConGPU

PIConGPU features an adaptable ionization framework for arbitrary and combinable ionization models.

.. note::

    Most of the calculations and formulae in this section of the docs are done in the **Atomic Units (AU)** system.
    
.. math::

    \hbar = \mathrm{e} = m_\mathrm{e} = 1

.. table:: **Atomic Unit System**
    :widths: auto
    :name: atomic_units

    ================  ======================================================================
    AU                SI
    ================  ======================================================================
    length            :math:`5.292 \cdot 10^{-11}\,\mathrm{m}`
    time              :math:`2.419 \cdot 10^{-17}\,\mathrm{s}`
    energy            :math:`4.360 \cdot 10^{-18}\,\mathrm{J}\quad` (= 27.21 eV = 1 Rydberg)
    electrical field  :math:`5.142 \cdot 10^{11}\,\frac{\mathrm{V}}{\mathrm{m}}`
    ================  ======================================================================

Overview: Implemented Models
----------------------------
.. table::
    :widths: auto
    :name: implemented__field_ionization_models

    +---------------------+-----------------------------+---------------------------+
    | ionization regime   | implemented model           | reference                 |
    +=====================+=============================+===========================+
    | Multiphoton         | None, yet                   |                           |
    +---------------------+-----------------------------+---------------------------+
    | Tunneling           | * ``Keldysh``               | * [BauerMulser1999]_      |
    |                     | * ``ADKLinPol``             | * [DeloneKrainov]_        |
    |                     | * ``ADKCircPol``            | * [DeloneKrainov]_        |
    +---------------------+-----------------------------+---------------------------+
    | Barrier Suppression | * ``BSI``                   | * [MulserBauer2010]_      |
    |                     | * ``BSIEffectiveZ``         | * [ClementiRaimondi1963]_ |
    |                     |                             |   [ClementiRaimondi1967]_ |
    |                     | * ``BSIStarkShifted`` (R&D) | * [BauerMulser1999]_      |
    +---------------------+-----------------------------+---------------------------+

.. attention::

    Models marked with "(R&D)" are under *research and development* and should be used with care.

Usage
-----

Input for ionization models is defined in :ref:`speciesDefinition.param <usage-params-core>`, :ref:`ionizer.param and ionizationEnergies.param <usage-params-extensions>`.


Barrier Suppression Ionization
------------------------------

The so-called barrier-suppression ionization regime is reached for strong fields where the potential barrier binding an electron is completely suppressed.

Tunneling Ionization
--------------------

Tunneling ionization describes the process during which an initially bound electron quantum-mechanically tunnels through a potential barrier of finite height.

Keldysh
"""""""

.. math::

    \Gamma_\mathrm{K} = \frac{\left(6 \pi\right)^{1/2}}{2^{5/4}} E_\mathrm{ip} \left( \frac{F}{(2 E_\mathrm{ip})^{3/2}} \right)^{1/2} \exp\left(-\frac{2 \left(2 E_\mathrm{ip}\right)^{3/2}}{3 F}\right)

The Keldysh ionization rate has been implemented according to the equation (9) in [BauerMulser1999]_. See also [Keldysh]_ for the original work.

.. note::

    Assumptions:

        * low field - perturbation theory
        * :math:`\omega_\mathrm{laser} \ll E_\mathrm{ip}`
        * :math:`F \ll F_\mathrm{BSI}`
        * tunneling is instantaneous


Ammosov-Delone-Krainov (ADK)
""""""""""""""""""""""""""""

.. math::
   :nowrap:

    \begin{align}
        \Gamma_\mathrm{ADK} &= \underbrace{\sqrt{\frac{3 n^{*3} F}{\pi Z^3}}}_\text{lin. pol.} \frac{F D^2}{8 \pi Z} \exp\left(-\frac{2Z^3}{3n^{*3}F}\right) \\
        D &\equiv \left( \frac{4 \mathrm{e} Z^3}{F n^{*4}} \right)^{n^*} \hspace{2cm} n^* \equiv \frac{Z}{\sqrt{2 E_\mathrm{ip}}}
    \end{align}

We implemented equation (7) from [DeloneKrainov]_ which is a *simplified result assuming s-states* (since we have no atomic structure implemented, yet).
Leaving out the pre-factor distinguishes ``ADKCircPol`` from ``ADKLinPol``.
``ADKLinPol`` results from replacing an instantaneous field strength :math:`F` by :math:`F \cos(\omega t)` and averaging over one laser period.

    .. attention::

        Be aware that :math:`Z` denotes the **residual ion charge** and not the proton number of the nucleus!

In the following comparison one can see the ``ADKLinPol`` ionization rates for the transition from Carbon II to III (meaning 1+ to 2+).
For a reference the rates for Hydrogen as well as the barrier suppression field strengths :math:`F_\mathrm{BSI}` have been plotted.
They mark the transition from the tunneling to the barrier suppression regime.

.. plot:: models/field_ionization_comparison_c_ii_ionization.py

When we account for orbital structure in shielding of the ion charge :math:`Z` according to [ClementiRaimondi1963]_ in ``BSIEffectiveZ`` the barrier suppression field strengths of Hydrogen and Carbon-II are very close to one another.
One would expect much earlier ionization of Hydrogen due to lower ionization energy. The following image shows how this can be explained by the shape of the ion potential that is assumed in this model.

.. plot:: models/field_ionization_effective_potentials.py

References
----------
.. [DeloneKrainov]
        N. B. Delone and V. P. Krainov.
        *Tunneling and barrier-suppression ionization of atoms and ions in a laser radiation field*,
        Phys. Usp. 41 469–485 (1998),
        http://dx.doi.org/10.1070/PU1998v041n05ABEH000393

.. [BauerMulser1999]
        D. Bauer and P. Mulser.
        *Exact field ionization rates in the barrier-suppression regime from numerical time-dependent Schrödinger-equation calculations*,
        Physical Review A 59, 569 (1999),
        https://dx.doi.org/10.1103/PhysRevA.59.569

.. [MulserBauer2010]
        P. Mulser and D. Bauer.
        *High Power Laser-Matter Interaction*,
        Springer-Verlag Berlin Heidelberg (2010),
        https://dx.doi.org/10.1007/978-3-540-46065-7

.. [Keldysh]
        L.V. Keldysh.
        *Ionization in the field of a strong electromagnetic wave*,
        Soviet Physics JETP 20, 1307-1314 (1965),
        http://jetp.ac.ru/cgi-bin/dn/e_020_05_1307.pdf
        
.. [ClementiRaimondi1963]
        E. Clementi and D. Raimondi.
        *Atomic Screening Constant from SCF Functions*,
        The Journal of Chemical Physics 38, 2686-2689 (1963)
        https://dx.doi.org/10.1063/1.1733573

.. [ClementiRaimondi1967]
        E. Clementi and D. Raimondi.
        *Atomic Screening Constant from SCF Functions. II. Atoms with 37 to 86 Electrons*,
        The Journal of Chemical Physics 47, 1300-1307 (1967)
        https://dx.doi.org/10.1063/1.1712084
