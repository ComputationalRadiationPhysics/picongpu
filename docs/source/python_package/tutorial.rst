Tutorial: Setting up a simple LWFA
==================================

We will now add some interesting physics to the minimal example
from the :ref:`Quick Start <python_package/quickstart:Quick Start>`.
This tutorial is supposed to give you a good introduction to the features
you will typically use in your daily work.
More details can be found in the
:ref:`Foundations <python_package/foundations/index:Foundations>` chapter
and in :ref:`Selected Topics <python_package/selected_topics/index:Selected Topics>`.

Extracting global constants
---------------------------

The minimal example was very concise and direct.
In practice, it is oftentimes more helpful to have access to common parameters in different parts of your input.
In order to do so, we extract some constants and decompose the definition of the solver:

.. literalinclude:: snippets/defining_simulation/lwfa_example.py
   :language: python
   :end-before: END-LWFA-CONSTANTS

Lasers
------

There are various lasers defined in `the PICMI standard <https://picmi-standard.github.io/>`__ and its :ref:`PIConGPU extension <python_package/selected_topics/lasers:Lasers>`.
We define a Gaussian laser as moving into positive ``y`` direction
(this is the convention PIConGPU is optimized for):

.. literalinclude:: snippets/defining_simulation/lwfa_example.py
   :language: python
   :start-after: BEGIN-LWFA-LASER
   :end-before: END-LWFA-LASER

Species and particles
---------------------

In the PICMI standard we define `abstract species <https://picmi-standard.github.io/>`__
and `distributions <https://picmi-standard.github.io/>`__ of particles belonging to such species among the cells.
The precise location of a particle inside of a cell is finally determined by `the layout <https://picmi-standard.github.io/>`__.
Thus, in order to add particles to our simulation we need three components:

.. literalinclude:: snippets/defining_simulation/lwfa_example.py
   :language: python
   :start-after: BEGIN-LWFA-SPECIES
   :end-before: END-LWFA-SPECIES

We add two species:
``hydrogen``, initialized from the ``GaussianDistribution``,
and ``electrons``, which is initially empty (``initial_distribution=None``).
The hydrogen is created in its ground state (``charge_state=0``, i.e. neutral),
so the plasma is charge neutral before ionization sets in.
The electron species does not receive any initial particles;
they are created by the ionization model below.

Interactions
------------

We can add various `interactions <https://picmi-standard.github.io/>`__ among our species.
As an example, we allow to ionize the hydrogen into the corresponding electron species:

.. literalinclude:: snippets/defining_simulation/lwfa_example.py
   :language: python
   :start-after: BEGIN-LWFA-ADK
   :end-before: END-LWFA-ADK

Diagnostics
-----------

Diagnostics, i.e. simulation output, are an important part of your simulation.
PIConGPU allows to define general diagnostics in a flexible way.
See :ref:`the diagnostics topic <python_package/selected_topics/index:Selected Topics>` for a full overview of the capabilities.
There are also various predefined diagnostics you can choose from.
Some of these provide quick access to heavily used features/debugging tools.
Others provide some optimized code for the diagnostic.
For example, we add a checkpoint and a macro-particle counter
(a useful tool for debugging the particle content of your simulation):

.. literalinclude:: snippets/defining_simulation/lwfa_example.py
   :language: python
   :start-after: BEGIN-LWFA-DIAGNOSTICS
   :end-before: END-LWFA-DIAGNOSTICS

Creating the simulation
-----------------------

We can now compose the above components
into a simulation setup:

.. literalinclude:: snippets/defining_simulation/lwfa_example.py
   :language: python
   :start-after: BEGIN-LWFA-SIMULATION
   :end-before: END-LWFA-SIMULATION

Running the simulation
----------------------

As a last step, we add the following lines to run the simulation upon execution of the script:

.. literalinclude:: snippets/defining_simulation/lwfa_example.py
   :language: python
   :start-after: BEGIN-LWFA-RUN
   :end-before: END-LWFA-RUN

The full script
---------------

.. dropdown:: Complete simulation script

   .. literalinclude:: snippets/defining_simulation/lwfa_example.py
      :language: python
