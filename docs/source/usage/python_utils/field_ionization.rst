.. _usage-python-utils-field-ionization:

Field Ionization Reference Formulas
-----------------------------------

.. sectionauthor:: Julian J. Lenz

The :class:`~picongpu.extra.utils.field_ionization.FieldIonization` class collects the analytic formulas behind PIConGPU's field-ionization models (barrier suppression, ADK and Keldysh) together with the atomic-unit conversions used by them.
All methods take and return values in atomic units unless stated otherwise; the ``atomic_unit`` dictionary holds the corresponding SI values.

.. code:: python

   from picongpu.extra.utils import FieldIonization

   fi = FieldIonization()
   fi.F_crit_BSI(Z=1, E_Ip=0.5)                      # critical field strength [AU]
   fi.ADKRate(Z=1, E_Ip=0.5, F=0.01)                 # ionization rate [1/AU(time)]
   fi.convert_a0_to_Intensity(E_in_a0=1.0)           # W/m^2 for an 800 nm laser

These formulas are used to derive the reference values of the corresponding PIConGPU models, see :ref:`model-fieldIonization`.

.. autoclass:: picongpu.extra.utils.field_ionization.FieldIonization
   :members:
