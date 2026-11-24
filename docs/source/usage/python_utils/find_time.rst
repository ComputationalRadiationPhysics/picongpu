.. _usage-python-utils-find-time:

Time to Iteration Conversion
----------------------------

.. sectionauthor:: Julian J. Lenz

The :class:`~picongpu.extra.utils.find_time.FindTime` class reads the time step from a finished PIConGPU run and converts between iterations (time steps) and simulation time in seconds.
It is the helper behind the plugin data readers, which use it to translate their ``time`` arguments, and can be used directly for the same purpose.

.. code:: python

   from picongpu.extra.utils import FindTime

   find_time = FindTime("/path/to/run")      # the path before ``simOutput/``
   find_time.get_dt()                        # time step in seconds
   find_time.get_time(iteration=200)         # simulation time in seconds
   find_time.get_iteration(t=1.0e-14)        # (iteration, time) for a given time

The constructor requires the run directory to contain a completed run with a ``simOutput/output`` file, because that is where the time step is read from.
The last call accepts the methods ``previous`` (default, closest iteration ``<= t``), ``closest`` and ``next`` and may optionally be restricted to a subset of iterations.

.. autoclass:: picongpu.extra.utils.find_time.FindTime
   :members:
