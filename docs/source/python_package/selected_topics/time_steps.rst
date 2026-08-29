.. _time-steps:

When to Run Diagnostics
========================

Almost every diagnostic described in this chapter accepts a ``period`` argument
that controls *on which time steps* the diagnostic writes its output.
All of them use the same class, :class:`picongpu.picmi.diagnostics.TimeStepSpec`.

.. literalinclude:: ../snippets/selected_topics/time_steps.py
   :language: python
   :start-after: from picongpu.picmi.diagnostics import TimeStepSpec
   :end-before: print("It worked!")

The syntax is a deliberate mix of familiar Python slicing and a few
PIConGPU-specific rules:

* The ``[]`` operator accepts **slices** (``start:stop:step``) and **integers**
  separated by commas.
* Slices are **inclusive on both ends**: ``TimeStepSpec[0:12:2]`` selects
  ``0, 2, 4, 6, 8, 10, 12`` (the endpoint is included if it is actually
  reached).
* A bare integer selects that single step.
* Comma-separated specifications are **unions**: ``TimeStepSpec[:5, 49:]``
  selects the first six steps *and* every step from ``49`` onwards.
* Negative ``start``/``stop`` count from the end of the run
  (``TimeStepSpec[-10:]`` is the last ten steps); ``stop=-1`` is the
  customary way of writing "to the end".
  A negative ``step`` is not allowed.

Units
-----

The default unit is **simulation steps**, which you select (or confirm) by
calling the specification with ``"steps"``.
You can also express a range in **physical time** in seconds with
``"seconds"``; it is translated into steps using your simulation's time step
size, rounded so that the interval is never clipped.

Because different units are meaningful in different contexts,
specifications in different units can be combined into one ``period`` with the
``+`` operator (a set union), as in the ``combined`` example above.

A few practical idioms:

``TimeStepSpec[::10]``
   Every 10th step (``0, 10, 20, ...``) -- the most common choice.

``TimeStepSpec[0:100:1]``
   The first hundred steps, every step.

``TimeStepSpec[-1]``
   Only the final step.

``TimeStepSpec[1e-15:1e-14:5e-16]("seconds")``
   Every 0.5 fs between 1 and 10 fs of physical time.

.. note::

   The radiation diagnostic cannot write at time step ``0`` (it needs a few
   steps of particle history); give its ``period`` a positive start,
   e.g. ``TimeStepSpec[2:-1:5]``.

.. _diagnostic-output-locations:

Where the Output Lands
----------------------

The diagnostics in this chapter write to different places:

* the **openPMD** diagnostics (see :ref:`openPMD Output <openpmd>`) write into
  ``simOutput/openPMD/``;
* the **checkpoint** plugin (see :ref:`Checkpoints <checkpoint>`) writes into
  a directory of your choice inside ``simOutput/``;
* the **phase space**, **energy histogram** and **macro-particle count**
  plugins write plain files directly into ``simOutput/``.

Recall from :ref:`Running Your Simulation <python_package/foundations/running_simulation:Running Your Simulation>`
that ``simOutput/`` lives in the run directory of the *submitted job*,
and is copied back to your local run directory when the job finishes.
