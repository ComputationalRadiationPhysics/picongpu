TransitionRadiation : Transtion Radiation
=============================================

.. sectionauthor:: Finn-Ole Carstens <f.carstens (at) hzdr.de>

This example simulates the coherent and incoherent transition radiation created by an electron bunch in-situ.
The implemented transition radiation follows the studies from [Schroeder2004]_ and [Downer2018]_.
The transition radiation is computed for an infinitely large interface perpendicular to the y-axis of the simulation.

The electron bunch in this setup is moving with a 45° angle in the x-y plane with a Lorentz-factor of $\gamma = 100$.
The bunch has a Gaussian distribution with $\sigma_y = 3.0 \mu m$.
The results can be interpreted with the according python script `lib/python/picongpu/plugins/plot_mpl/transition_radiation_visualizer.py`.

References
----------

.. [Schroeder2004]
       Schroeder, C. B. and Esarey, E. and van Tilborg, J. and Leemans, W. P.
       *Theory of coherent transition radiation generated at a plasma-vacuum interface*,
       American Physical Society(2004),
       https://link.aps.org/doi/10.1103/PhysRevE.69.016501

.. [Downer2018]
       Downer, M. C. and Zgadzaj, R. and Debus, A. and Schramm, U. and Kaluza, M. C.
       *Diagnostics for plasma-based electron accelerators*,
       American Physical Society(2018),
       https://link.aps.org/doi/10.1103/RevModPhys.90.035002
