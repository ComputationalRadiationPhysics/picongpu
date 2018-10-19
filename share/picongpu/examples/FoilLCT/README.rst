.. _usage-examples-foilLCT:

FoilLCT: Ion Acceleration from a Liquid-Crystal Target
======================================================

.. sectionauthor:: Axel Huebl
.. moduleauthor:: Axel Huebl, T. Kluge

The following example models a laser-ion accelerator in the [TNSA]_ regime.
An optically over-dense target (:math:`n_\text{max} = 192 n_\text{c}`) consisting of a liquid-crystal material *8CB* (4-octyl-4'-cyanobiphenyl) :math:`C_{21}H_{25}N` is used.

Irradiated with a high-power laser pulse with :math:`a_0 = 5` the target is assumed to be partly pre-ionized due to realistic laser contrast and pre-pulses to :math:`C^{2+}`, :math:`H^+` and :math:`N^{2+}` while being slightly expanded on its surfaces (modeled as exponential density slope).
The overall target is assumed to be initially quasi-neutral and the *8CB* ion components are are not demixed in the surface regions.
Surface contamination with, e.g. water vapor is neglected.

The laser is assumed to be in focus and approximated as a plane wave with temporally Gaussian intensity envelope of :math:`\tau^\text{FWHM}_I = 25` fs.

This example is used to demonstrate:

* an ion acceleration setup with
* :ref:`composite, multi ion-species target material <usage-workflows-compositeMaterials>`
* :ref:`quasi-neutral initial conditions <usage-workflows-quasiNeutrality>`
* ionization models for :ref:`field ionization <model-fieldIonization>` and :ref:`collisional ionization <model-collisionalIonization>`

with PIConGPU.

References
----------

.. [TNSA]
       S.C. Wilks, A.B. Langdon, T.E. Cowan, M. Roth, M. Singh, S. Hatchett, M.H. Key, D. Pennington, A. MacKinnon, and R.A. Snavely.
       *Energetic proton generation in ultra-intense laser-solid interactions*,
       Physics of Plasmas **8**, 542 (2001),
       https://dx.doi.org/10.1063/1.1333697
