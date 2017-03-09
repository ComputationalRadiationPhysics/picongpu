Photons
=======

Radiation reaction and (hard) photons: why and when are they needed.
Models we implemented and verified:

* Landau-Lifschitz Model (semi-classical)
* QED Models (Synchrotron & Bremsstrahlung)

Would be great to add your Diploma Thesis talk with pictures and comments here.

Please add notes and warnings on the models' assumptions for an easy guiding on their usage :)

.. note::
   Assumptions in Furry-picture and Volkov-States: classical em wave part and QED "pertubation".
   EM fields on grid (Synchrotron) and density modulations (Bremsstrahlung) need to be locally constant compared to radiated coherence interval ("constant-crossed-field approximation").

.. attention::
   Bremsstrahlung: The individual electron direction and gamma emission are not correlated.
   (momentum is microscopically / per e- not conserved, only collectively.)

.. attention::
   "Soft" photons from low energy electrons will get underestimated in intensity below a threshold of ... .
   Their energy is still always conserved until cutoff (defined in ...).

.. note::
   An electron can only emit a photon with identical weighting.
   Otherwise, the statistical variation of their energy loss would be weighting dependent
   (note that the average energy loss is unaffected by that).

References
----------

.. [Gonoskov]
        A. Gonoskov, S. Bastrakov, E. Efimenko, A. Ilderton, M. Marklund, I. Meyerov, A. Muraviev, A. Sergeev, I. Surmin, E. Wallin.
        *Extended particle-in-cell schemes for physics in ultrastrong laser fields: Review and developments*,
        Physical Review E 92, 023305 (2015),
        https://dx.doi.org/10.1103/PhysRevE.92.023305

.. [Furry]
        W. Furry.
        *On bound states and scattering in positron theory*,
        Physical Review 81, 115 (1951),
        https://doi.org/10.1103/PhysRev.81.115

.. [Burau2016]
        H. Burau.
        *Entwicklung und Überprüfung eines Photonenmodells für die Abstrahlung durch hochenergetische Elektronen* (German),
        Diploma Thesis at TU Dresden & Helmholtz-Zentrum Dresden - Rossendorf for the German Degree "Diplom-Physiker" (2016),
        https://doi.org/10.5281/zenodo.192116
