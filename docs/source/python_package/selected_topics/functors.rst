.. _functors:

Functors, Particle Functors and Filters
=======================================

Beyond the built-in distributions, PIConGPU can build quantities from
particle properties symbolically.
The same machinery is used for **analytic density profiles**,
**particle functors** (quantities computed per particle) and
**particle filters** (functor-valued predicates that select particles).

Analytic densities
------------------

The most direct use is an analytic density:
:class:`~picongpu.picmi.distribution.AnalyticDistribution`
takes a ``density_function`` of three sympy symbols ``x``, ``y``, ``z``
(in SI units) and returns a density expression
(also in SI units).
The expression is compiled into the simulation binary
and evaluated on the GPU at runtime:

.. literalinclude:: ../snippets/selected_topics/analytic_distribution.py
   :language: python
   :start-after: BEGIN-DENSITY-FUNCTION
   :end-before: END-DENSITY-FUNCTION

Instead of a callable you may pass the density as a sympy-parseable
string via the ``density_expression`` keyword;
it is string-normalised (as in the PICMI standard) and parsed with
``sympy.sympify``, so non-string values are coerced to their string form
(a bare number gives a constant density) and the result is exactly
equivalent to the matching ``density_function``:

.. literalinclude:: ../snippets/selected_topics/analytic_distribution.py
   :language: python
   :start-after: BEGIN-DENSITY-EXPRESSION
   :end-before: END-DENSITY-EXPRESSION

Provide exactly one of ``density_function`` or ``density_expression``.

Use ``sympy.Piecewise`` for conditional profiles;
the momentum parameters ``rms_velocity`` and ``directed_velocity``
are currently only partially supported
(``rms_velocity`` is pinned to zero; ``directed_velocity`` is accepted
but untested).

The same symbolic machinery is the natural building block for other
user-supplied, code-level expressions;
free-form lasers, for instance, are expected to reuse it later.

.. _particle-functors:

Particle functors
-----------------

A :class:`~picongpu.picmi.particle_functor.ParticleFunctor`
is a Python function of one (or two) arguments
that describes a particle property symbolically.
It is used as a decorator::

   @ParticleFunctor
   def gamma(particle):
       ...

The ``particle`` argument provides access to the particle's attributes
through ``particle.get("...")``:

* ``"position"``:
  a 3D vector; the keyword arguments ``origin``
  (``"total"`` (default), ``"local"``, ``"global"``,
  ``"moving_window"`` or ``"local_with_guards"``),
  ``precision`` (``"cell"`` (default) or ``"sub_cell"``)
  and ``unit`` (``"cell"`` (default), ``"pic"`` or ``"si"``)
  select the reference frame, the resolution and the units.
  (``"cell"`` is only available as an ``origin`` for derived-field and
  filter functors, not for binning functors.)
* ``"momentum"`` / ``"momentumPrev1"``:
  the 3D momentum (index it as
  ``px, py, pz = particle.get("momentum")``).
* ``"mass"``, ``"charge"``, ``"weighting"``:
  the particle's mass, charge and statistical weight.
* ``"gamma"``, ``"kinetic energy"``, ``"velocity"``:
  derived quantities, computed from mass and momentum.
* any further attribute name:
  rendered as a particle-attribute access in the generated code.

The function body is written with `sympy <https://www.sympy.org/>`__
expressions -- it is never executed as plain Python,
but *symbolically* (with dummy particle attributes)
and the resulting expression tree is compiled into the simulation binary,
where it is evaluated on the GPU.
The functor's ``name`` defaults to the function name;
use ``@ParticleFunctor(name="...")`` to override it.
Use ``return_type`` (e.g. ``int``) when the annotation of your function
is not enough, and ``unit_dimension``
(a :class:`~picongpu.picmi.particle_functor.UnitDimension`)
to declare the physical unit of the result.

.. literalinclude:: ../snippets/selected_topics/particle_functors.py
   :language: python
   :start-after: BEGIN-PARTICLE-FUNCTOR
   :end-before: END-PARTICLE-FUNCTOR

.. _particle-filters:

Particle filters
----------------

The same mechanism can be used to *select* particles instead of measuring them:
a :class:`~picongpu.picmi.particle_functor.ParticleFilter`
is a functor that must return a boolean,
and wrapping a species and a filter in a
:class:`~picongpu.picmi.particle_functor.FilteredSpecies`
gives a "species" that contains only the selected particles:

.. literalinclude:: ../snippets/selected_topics/particle_functors.py
   :language: python
   :start-after: BEGIN-PARTICLE-FILTER
   :end-before: END-PARTICLE-FILTER

``FilteredSpecies`` is accepted everywhere a ``Species`` is
(phase space, energy histogram, particle dump, binning, ...).
The name of the filtered species is
``<species name>_<filter name>``,
which is also what you will find in the output files.

.. note::

   Deep dive:
   functors are compiled into the binary -- binning expressions as
   ``ALPAKA_FN_ACC`` lambdas, derived attributes (file output) as
   ``DINLINE`` call operators of generated functor structs -- in
   ``binningSetup.param`` / ``fileOutput.param``
   (see :ref:`the binning plugin documentation <usage-plugins-binningPlugin>`
   for the underlying C++ implementation).
   A functor that cannot be expressed in terms of the available
   particle attributes will fail at input-file generation
   with a message pointing at the offending expression.
