Interactions
============

Interactions describe the physics that acts on your particles
*in addition to* the electromagnetic fields
the solver provides:
ionization, binary collisions and radiation reaction.
They are passed to the simulation via the
``picongpu_interaction`` parameter:

.. code-block:: python

   sim = picmi.Simulation(max_steps=100, solver=solver, picongpu_interaction=[...])

.. note::

   The PICMI-standard method ``simulation.add_interaction()``
   is *not* supported by PIConGPU
   (it only emits an "unsupported" warning) --
   always use the ``picongpu_interaction`` parameter instead.

Each interaction is attached to the species it acts on
(and, where applicable, creates new species);
the species themselves are added to the simulation
as usual via ``add_species``.

.. _ionization:

Ionization
----------

An ionization model couples an *ion species*
(an element species with a fixed ``charge_state``)
to an *electron species* that receives the freed electrons.
In a typical setup the electron species starts out empty
(``initial_distribution=None``)
and is populated purely by ionization.
Each model is one of:

``picmi.ADK``
   ADK tunnel ionization
   (``ADK_variant`` selects ``picmi.ADKVariant.LinearPolarization``
   or ``picmi.ADKVariant.CircularPolarization``).
   This is the model of the :ref:`LWFA tutorial <tutorial>`.

``picmi.BSI``
   Barrier suppression ionization;
   ``BSI_extensions`` (a tuple) optionally adds
   ``picmi.BSIExtension.StarkShift`` or ``picmi.BSIExtension.EffectiveZ``.

``picmi.Keldysh``
   The quantum Keldysh model, which interpolates between
   tunnel (ADK) and multiphoton ionization.

All of them additionally take ``ionization_current``,
which selects how the ionization current
(the momentum carried away by the electrons)
is treated for energy conservation;
``None`` (the default) disables it.

.. literalinclude:: ../snippets/selected_topics/interactions.py
   :language: python
   :start-after: BEGIN-INTERACTIONS-ADK
   :end-before: END-INTERACTIONS-ADK

The same snippet also shows the BSI variant:

.. literalinclude:: ../snippets/selected_topics/interactions.py
   :language: python
   :start-after: BEGIN-INTERACTIONS-BSI
   :end-before: END-INTERACTIONS-BSI

.. note::

   ``picmi.ThomasFermi``
   (collisional ionization / electronic collisional equilibrium)
   exists in the API,
   but currently fails at input-file generation;
   it is therefore not documented here in detail.

   Deep dive:
   :ref:`the ionization models in the PIConGPU code <model-fieldIonization>`
   and :ref:`the collisional ionization model <model-collisionalIonization>`.

.. _collisions:

Collisions
----------

Binary collisions between particle species
(Coulomb collisions with a constant or dynamically computed
Coulomb logarithm) are represented by

``picmi.Collision``
   One collision between pairs of species
   (``species_pairs`` is a list of ``(lhs, rhs)`` pairs;
   convenience constructors ``construct_one_to_all``
   and ``construct_all_to_all`` are available).
   The ``functor`` selects the physics:
   ``picmi.ConstLogCollision(coulomb_log=...)``
   or ``picmi.DynamicLogCollision()``
   (the latter requires *screening species* to compute the log from).

``picmi.CollisionalPhysicsSetup``
   An optional container that holds several collisions
   together with the ``screening_species`` and numerical options
   (``precision``, ``cell_list_chunk_size``).
   If you pass bare ``Collision`` objects,
   they are combined into such a setup automatically;
   if you pass a setup, *all* collisions must be subsumed under it.

.. note::

   The collision API is currently broken:
   generating the input files for a simulation that contains
   a collision fails with an internal serialization error.
   The class and constructor signatures above are the intended interface;
   until the bug is fixed, collisions cannot be used from the
   Python interface.

   Deep dive:
   :ref:`the binary collision model in the PIConGPU code <model-binaryCollisions>`.

.. _synchrotron:

Synchrotron Radiation
---------------------

``picmi.Synchrotron``
   Couples an *electron species* to a *photon species*
   (``particle_type="photon"``; the species starts out empty
   and is populated by the radiation):
   the electrons emit synchrotron radiation
   and (optionally) receive the recoil.

.. literalinclude:: ../snippets/selected_topics/interactions.py
   :language: python
   :start-after: BEGIN-INTERACTIONS-SYNCHROTRON
   :end-before: END-INTERACTIONS-SYNCHROTRON

The optional ``synchrotron_parameters``
(a ``SynchrotronParams`` model)
exposes ``electron_recoil`` (default ``True``)
and ``min_energy``,
an energy high-pass filter for the generated photons.

Deep dive:
:ref:`the synchrotron radiation extension <synchrotronRadiation>`.
