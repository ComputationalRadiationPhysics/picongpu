.. _applied_fields:

Applied (Background) Fields
===========================

PIConGPU implements the PICMI-standard *applied fields* as **background
fields**: a field that is added to the grid ``E`` and ``B`` fields around the
particle push, so the particles feel it, while the field solver itself does not
evolve it. Only the **whole simulation domain** is supported; ``lower_bound``
and ``upper_bound`` must be left at their default (all ``None``), and region
restriction raises an ``UnsupportedFeatureError``.

Applied fields are attached declaratively via the ``applied_fields`` argument
of :class:`~picongpu.picmi.simulation.Simulation` (or with
:meth:`~picongpu.picmi.simulation.Simulation.add_applied_field`). Several fields
may be given; their contributions are summed per component into the single
background field that the C++ core evaluates:

.. literalinclude:: ../snippets/selected_topics/applied_fields.py
   :language: python
   :start-at: BEGIN-APPLIED-FIELD-ADD
   :end-before: END-APPLIED-FIELD-ADD

The field classes
-----------------

:class:`~picongpu.picmi.applied_field.ConstantAppliedField`
   A field that is constant in space and time. Its components use the
   PICMI-standard names ``Ex``, ``Ey``, ``Ez`` (in V/m) and ``Bx``, ``By``,
   ``Bz`` (in T).

:class:`~picongpu.picmi.applied_field.AnalyticAppliedField`
   A field given by Python expressions. Use the variables ``x``, ``y``, ``z``
   (position in m) and ``t`` (time in s); additional keyword arguments become
   named parameters inside the expressions. As in
   :class:`~picongpu.picmi.distribution.AnalyticDistribution.AnalyticDistribution`,
   each of the six components is backed by the shared
   :class:`~picongpu.picmi._FieldFunctor._FieldFunctor` and is exposed in all
   three interchangeable spellings: a sympy-parseable ``<component>_expression``
   string, a ``<component>_function`` callable, and the resolved
   ``<component>_sympy`` expression (see :doc:`functors`). Any one of them may be
   supplied; the others are computed from it, so all three are available and
   consistent after construction, and several spellings may be given for one
   component as long as they agree. Expressions are in V/m for ``E`` and T for
   ``B``, evaluated in SI units and converted to PIConGPU's internal units (see
   :ref:`units`).

.. literalinclude:: ../snippets/selected_topics/applied_fields.py
   :language: python
   :start-at: BEGIN-APPLIED-FIELD-CONSTANT
   :end-before: END-APPLIED-FIELD-CONSTANT

.. literalinclude:: ../snippets/selected_topics/applied_fields.py
   :language: python
   :start-at: BEGIN-APPLIED-FIELD-ANALYTIC
   :end-before: END-APPLIED-FIELD-ANALYTIC

All three spellings of a component are backed by the same
:class:`~picongpu.picmi._FieldFunctor._FieldFunctor`: supplying any one computes
the others, so they are available and consistent after construction, and
supplying several for one component is accepted as long as they agree. The
``<component>_expression``, ``<component>_function`` and ``<component>_sympy``
fields are all declared on the PIConGPU class itself:

.. literalinclude:: ../snippets/selected_topics/applied_fields.py
   :language: python
   :start-at: BEGIN-APPLIED-FIELD-SYMPY
   :end-before: END-APPLIED-FIELD-SYMPY

Evaluating the field
--------------------

Like :class:`~picongpu.picmi.distribution.AnalyticDistribution.AnalyticDistribution`,
every applied field is callable. Calling it with the coordinates ``x``, ``y``,
``z`` and ``t`` (SI units) returns the six components ``Ex`` … ``Bz`` as
evaluated by the same expression the C++ functor renders; a component that was
not set is ``None``. The arguments may be scalars or numpy arrays, so a whole
cell grid can be queried at once:

.. literalinclude:: ../snippets/selected_topics/applied_fields.py
   :language: python
   :start-at: BEGIN-APPLIED-FIELD-CALL
   :end-before: END-APPLIED-FIELD-CALL

Expressions may only reference the free variables ``x``, ``y``, ``z`` and ``t``
plus the named parameters passed as additional keyword arguments; any other
symbol is rejected with a ``ValueError`` before code generation. Parameter names
must not collide with those free variables or with generated identifiers such as
``cellIdx`` or ``sim`` (also a ``ValueError``); C++ keywords are escaped by the
PMAccPrinter rather than rejected.

Visibility knobs
----------------

Both classes accept two PIConGPU-specific visibility knobs (the ``picongpu_``
prefix marks code-specific PICMI inputs); they correspond to options of the
generated run configuration and default to ``True``:

* ``picongpu_influences_plugins`` — whether plugins see the background.
* ``picongpu_influences_dumps`` — whether dumps, including checkpoints, include
  the background.

The background is always applied to the grid around the particle push, so the
particles always feel it; there is no pusher-visibility switch. The two knobs
cover the electric **and** magnetic background together.

Because the knobs configure the *single* C++ background functor pair, all
applied fields of a simulation must agree on them. A mismatch is caught during
translation and rejected with an ``UnsupportedFeatureError`` — the same applies
to redefining a shared parameter with a different value:

.. literalinclude:: ../snippets/selected_topics/applied_fields.py
   :language: python
   :start-at: BEGIN-APPLIED-FIELD-INFLUENCE
   :end-before: END-APPLIED-FIELD-INFLUENCE

When no background field is configured, the plugin/dump options are not written
to the generated run configuration at all, so the core defaults apply unchanged.
