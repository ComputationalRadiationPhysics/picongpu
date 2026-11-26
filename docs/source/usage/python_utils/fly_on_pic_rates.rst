.. _usage-python-utils-fly-on-pic-rates:

FLYonPIC Rate Reference Implementation
--------------------------------------

.. sectionauthor:: Julian Lenz

``picongpu.extra.utils.FLYonPICRateCalculationReference`` is the reference implementation of the rate and cross-section formulas of the FLYonPIC atomic-physics model, see :ref:`model-atomicPhysics`.
It is not used during a simulation.
Instead, its output is the ground truth against which the C++ implementation is tested, and it is kept here so that the reference values can be regenerated and inspected.

The package exposes one class per transition type, each providing the cross-section and rate methods for that process:

* :class:`~picongpu.extra.utils.FLYonPICRateCalculationReference.BoundBoundTransitions` — collisional excitation/deexcitation and spontaneous deexcitation,
* :class:`~picongpu.extra.utils.FLYonPICRateCalculationReference.BoundFreeCollisionalTransitions` — collisional ionization,
* :class:`~picongpu.extra.utils.FLYonPICRateCalculationReference.BoundFreeFieldTransitions` — field (ADK) ionization.

.. code:: python

   from picongpu.extra.utils import FLYonPICRateCalculationReference

   reference = FLYonPICRateCalculationReference.BoundFreeFieldTransitions
   reference.ADKRate_numpy(Z=5, E_Ip=5.0 / 27.211386245981, F=0.03)

The script ``calculatorMain.py`` next to these modules prints the complete reference set.
It uses flat imports of its sibling modules, so it has to be run from its own directory:

.. code-block:: bash

   cd lib/python/picongpu/extra/utils/FLYonPICRateCalculationReference
   python calculatorMain.py

.. autoclass:: picongpu.extra.utils.FLYonPICRateCalculationReference.BoundBoundTransitions
   :members:

.. autoclass:: picongpu.extra.utils.FLYonPICRateCalculationReference.BoundFreeCollisionalTransitions
   :members:

.. autoclass:: picongpu.extra.utils.FLYonPICRateCalculationReference.BoundFreeFieldTransitions
   :members:
