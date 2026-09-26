Diagnostics
===========

PIConGPU offers a range of diagnostics (plugins) that let you extract data
from your simulation.
They are all scheduled in time through the same
:class:`~picongpu.picmi.diagnostics.TimeStepSpec` mechanism,
and each page below describes one of them:

* the general :ref:`time-step scheduling <time-steps>`,
* the **file-based** diagnostics
  :ref:`openPMD output <openpmd>` (fields and particles),
  :ref:`phase space <phase-space>`,
  :ref:`energy histogram <energy-histogram>`,
  :ref:`macro-particle count <macro-particle-count>`,
  :ref:`binning <binning>`,
  :ref:`radiation <radiation>` and
  :ref:`checkpoints <checkpoint>`.

The :ref:`diagnostic output locations <diagnostic-output-locations>` are
summarized at the end of the time-step page.

.. toctree::
   :maxdepth: 2
   :hidden:

   time_steps
   openpmd
   phase_space
   energy_histogram
   macro_particle_count
   binning
   radiation
   checkpoint
