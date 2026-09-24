.. _custom_input:

Custom User Input and Templates
===============================

Sometimes a setup needs a value that is not part of the PICMI interface --
for instance a project-specific parameter used by a customised template.
PIConGPU lets you register such values and your own templates
on the simulation.

Custom user input
-----------------

Create a :class:`~picongpu.pypicongpu.customuserinput.CustomUserInput`
and add key/value pairs under a tag,
then register it on the simulation with
``picongpu_add_custom_user_input``:

.. literalinclude:: ../snippets/selected_topics/custom_input.py
   :language: python
   :start-at: custom = CustomUserInput
   :end-before: simulation.write_input_file

The values become available to every template under the
``customuserinput`` key, referenced as
``{{{customuserinput.<key>}}}``
(or the longer ``{{{customuserinput.my_number}}}`` spelling).
Tags are collected separately under ``customuserinput.tags``.
Keys must not collide with different values, and tags must be unique.

Custom templates
----------------

``simulation.picongpu_template_dir`` points at one or more directories
that supply or shadow templates.
Each directory mirrors the layout of the packaged templates
(``etc/picongpu``, ``include/picongpu``, ``lib``, ...),
and templates use `mustache <https://mustache.github.io/>`__ syntax.
When unset, the templates shipped with the package are used.

.. literalinclude:: ../snippets/selected_topics/custom_input.py
   :language: python
   :start-at: template_dir = Path
   :end-before: custom = CustomUserInput

``write_input_file`` renders each ``*.mustache`` file and keeps the
template next to the rendered result, so the setup remains self-describing.

The rendering context
---------------------

The data available to a template is the *rendering context*:
a JSON object that is written to
``metadata/pypicongpu_rendering_context.json`` in every generated setup
(see :ref:`the serialization page <serialization>`).
Its top-level keys include ``species``, ``laser``, ``solver``, ``grid``,
``time_steps`` and ``customuserinput``.

Templates use `mustache <https://mustache.github.io/>`__ syntax:

* ``{{{name}}}`` substitutes a value
  (the triple braces avoid HTML escaping).
* ``{{#key}} ... {{/key}}`` is both an *if* and a *for-each*:
  the block is rendered once if ``key`` exists and is not null,
  and once per entry if ``key`` is a list.
  Inside the block, names are looked up in the current entry first,
  then in the enclosing scopes.

For example, to emit one line per species, iterating over the ``species``
list of the rendering context:

.. literalinclude:: ../snippets/selected_topics/custom_iteration.py
   :language: python
   :start-at: (template_dir / "include" / "picongpu" / "species_report.mustache")
   :end-before: grid = picmi.Cartesian3DGrid

The same template can combine that with custom user input, referenced as
``{{{customuserinput.<key>}}}``.

.. note::

   Custom user input is global to the simulation,
   and the ``picongpu_custom_user_input`` field is excluded from the
   (de-)serialized representation
   (see :ref:`the serialization page <serialization>`),
   because it may hold arbitrary, non-serializable values.
