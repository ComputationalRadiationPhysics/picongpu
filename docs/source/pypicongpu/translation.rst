.. _pypicongpu-translation:

Translation Process
===================

This document explains the semantic and syntactic transformation from
PICMI inputs to PIConGPU build files. It also describes when wich type
of error is caught or raised.

The data are transformed along the following pipeline:

1. User script

   -  is: python code, has ``from picongpu import picmi``
   -  purpose: input in generic programming language, editable by user
   -  defined in: user file (not part of this repo)

2. PICMI objects

   -  is: python objects specified by upstream ``picmistandard``

      -  note that the objects upstream are **inherited** (i.e. **NOT**
         copied)

   -  purpose: user interface
   -  located in: ``lib/python/pypicongpu/picmi``

3. PyPIConGPU objects (1-to-1 translateable to C++ code)

   -  is: python
   -  purposes:

      -  use PIConGPU representation instead of PICMI (number of cells
         vs extent in SI units etc.)
      -  consruct valid input parameters (where one can also modify data
         on a semantic level)
      -  possibly even modify parameters (e.g. “electron resolution”)

   -  “intermediate representation”
   -  no redundancies
   -  aims to be close to PIConGPU native input (in terms of structure)
   -  located in: ``lib/python/pypicongpu`` (everthing that is not
      PICMI)

4. JSON representation (“rendering context”)

   -  is: `JSON <https://www.json.org/json-en.html>`__ (simple
      hierarchical data format)
   -  purpose: passed to template engine to generate code (also:
      separate translation logic from code generation)
   -  slightly restricted version of json
   -  generated according to predefined schemas
   -  may contain rendundancies
   -  located in:

      -  generated in ``_get_serialized()`` implementations
      -  checked against schemas in ``share/pypicongpu/schema``
      -  stored (for debugging) in generated files as
         ``pypicongpu.json``

5. ``.param`` and ``.cfg`` files

   -  is: files as defined by PIConPGU
   -  purpose: fully compatible to vanilla PIConGPU
   -  generated using predefined template
   -  created using “templating engine” mustache, terminology:

      -  *render*: generate final code by applying a *context* to a
         template
      -  *context*: (JSON-like) data that is combined with a template
         during rendering
      -  a *templating engine* is a more powerful search replace;
         *rendering* describes transforming a *template* (where things
         are replaced) with a *context* (what is inserted)

   -  located in:

      -  generated directory
      -  templates (``*.mustache``) in ``share/pypicongpu/template``

6. (normal PIConGPU pipeline continues: ``pic-build``, ``tbg``)

   -  ``pic-build`` compile the code

      -  wrapper around cmake
      -  (would be run standalone for native PIConGPU)

   -  ``tbg`` runs the code

      -  enable usage on different platforms/hpc clustern/submit systems

Please see at the end of this document for an example.

Generally, each object has a method to transform it into the next step:
PICMI objects have a method ``get_as_pypicongpu()``, pypicongpu object
have a method ``get_rendering_context()`` etc.

The individual steps are fairly dumb (see: KISS), so if you do something
simple it should work – and anything complicated won’t.

The individual data formats and their transformations are outlined
below.

Practically this pipeline is invoked by the runner, see :doc:`the
corresponding documentation chapter <./running>`.

PICMI
-----

PICMI objects are objects as defined by the picmistandard.

A user script creates PICMI objects, all collected inside the
``Simulation`` object. Technically these objects are implemented by
pypicongpu, though (almost) all functionality is inherited from the
implementation inside the picmistandard.

**The content of the PICMI objects is assumed to be conforming to
standard.** PICMI (upstream) does not necessarily enforce this. The user
may define invalid objects. Such objects outside the standard produce
undefined behavior!

When invoking ``get_as_pypicongpu()`` on a PICMI object it is translated
to the corresponding pypicongpu object. Not all PICMI objects have a
direct pypicongpu translation, in which case the object owning the
non-translateable object has to manage this translation (e.g. the
simulation extracts the PICMI grid translation from the PICMI solver).
Before this translations happens, some simple checks are performed,
mainly for compatibility. When an incompatible parameter is encountered,
one of two things can happen:

1. A utility function is called to notify the user that the parameter is
   unsupported. This typically applies to parameters that are not
   supported at all.
2. An assertions is triggered. In this case python will create an
   ``AssertionError``, and an associated message is provided. These
   errors can’t be ignored. This typically applies to parameters for
   which only some values are supported.

The translation from PICMI to PyPIConGPU does not follow a “great design
plan”, the approach can be summed up by:

-  have unit tests for everything (which are in turn not well organized,
   but just huge lists of test cases)
-  trust PyPIConGPUs self-validation to catch configuration errors
-  translate as *local* as possible, i.e. translate as much as possible
   within the lowest hierachy level, and only rarely leave translation
   to the upper level (e.g. the all-encompassing simulation)

..

   The PICMI to PyPIConGPU translation itself is rather messy (and there
   is no way to entirely avoid that).

PyPIConGPU
----------

PyPIConGPU objects are objects defined in pypicongpu which resemble the
PIConGPU much more closely than plain PICMI. Hence, they (may) hold
different data/parameters/formats than the PICMI objects.

As PICMI, all PyPIConGPU objects are organized under the top-level
``Simulation`` object.

   Be aware that **both** PICMI **and** PyPIConGPU have a class named
   ``Simulation``, but these classes are **different**.

Typically, PyPIConGPU objects do not contain much logic – they are
structures to hold data. E.g. the 3D grid is defined as follows
(docstrings omitted here):

.. code:: python

   @typeguard.typechecked
   class Grid3D(RenderedObject):
       cell_size_x_si = util.build_typesafe_property(float)
       cell_size_y_si = util.build_typesafe_property(float)
       cell_size_z_si = util.build_typesafe_property(float)

       cell_cnt_x = util.build_typesafe_property(int)
       cell_cnt_y = util.build_typesafe_property(int)
       cell_cnt_z = util.build_typesafe_property(int)

       boundary_condition_x = util.build_typesafe_property(BoundaryCondition)
       boundary_condition_y = util.build_typesafe_property(BoundaryCondition)
       boundary_condition_z = util.build_typesafe_property(BoundaryCondition)

In particular, please note:

-  The annotation ``@typeguard.typechecked``: This is a decorator introduced by
   ``typeguard`` and it ensures that the type annotations of methods are
   respected. However, it does not perform typechecking for attributes,
   which is why the attributes are delegated to:
-  ``util.build_typesafe_property()`` …is a helper to build a
   `property <https://docs.python.org/3.10/library/functions.html#property>`__
   which automatically checks that only the type specified is used.
   Additionally, it **does not allow default values**, i.e. a value must
   be **set explicitly**. If it is read before a write an error is
   thrown.
-  The parent class ``RenderedObject``: Inheriting from
   ``RenderedObject`` means that this object can be translated to a
   *context* for the templating egine (see JSON representation). The
   inheriting Object ``Grid3D`` must implement a method
   ``_get_serialized(self) -> dict`` (not shown here), which returns a
   dictionary representing the internal state of this object for
   rendering. It is expected that two object which return the same
   result for ``_get_serialized()`` are equal. This method is used by
   ``RenderedObject`` to provide the ``get_rendering_context()`` which
   invokes ``_get_serialized()`` internally and performs additional
   checks (see next section).

Some objects only exist for processing purposes and do not (exclusively)
hold any simulation parameters, e.g. the ``Runner`` (see
:doc:`./running`) or the ``InitializationManager`` (see
:doc:`./species`).

JSON representation (“rendering context”)
-----------------------------------------

A JSON representation is fed into the templating engine to *render* a
template.

The rendering engine used here is
`Mustache <https://mustache.github.io/>`__, implemented by
`chevron <https://github.com/noahmorrison/chevron>`__.

   Think of a templating engine as a more powerful search-replace.
   Please refer to `the mustache
   documentation <http://mustache.github.io/mustache.5.html>`__ for
   further details. (The authoritative
   `spec <https://github.com/mustache/spec/>`__ outlines some additional
   features.)

We apply the Mustache standard more strictly than necessary, please see
below for further details.

   | Motivation: Why Mustache?
   | There are plenty of templating engines available. Most of these
     allow for much more logic inside the template than mustache does,
     even being turing complete. However, having a lot of logic inside
     of the template itself should be avoided, as they make the code
     much less structured and hence more difficult to read, test and
     maintain. Mustache itself is not very powerful, forcing the
     programmer to define their logic inside of PyPIConGPU. In other
     words: The intent is to disincentivize spaghetti template code.

The JSON representation is created inside of ``_get_serialized()``
implemented by classes inheriting from ``RenderedObject``. Before it is
passed to the templating engine, it has to go through three additional
steps:

1. check for general structure
2. check against schema
3. JSON preprocessor

..

   Notably, the schema check is actually performed *before* the general
   structure check, but conceptionally the general structure check is
   more general as the schema check.

Check General Structure
~~~~~~~~~~~~~~~~~~~~~~~

This check is implemented in ``Renderer.check_rendering_context()`` and
ensures that the python dict returned by ``_get_serialized()`` can be
used for Mustache as rendering context, in particular:

-  context is ``dict``
-  all keys are strings

   -  keys do **NOT** contain dot ``.`` (reserved for element-wise
      access)
   -  keys do **NOT** begin with underscore ``_`` (reserved for
      preprocessor)

-  values are either dict, list or one of:

   -  None
   -  boolean
   -  int or float
   -  string

-  list items must be dictionaries This is due to the nature of Mustache
   list processing (loops): The loop header for Mustache ``{{#list}}``
   does enter the context of the list items, e.g. for
   ``[{"num": 1"}, {"num": 2}]`` ``num`` is now defined after the loop
   header. This *entering the context* is not possible if the item is
   not a dict, e.g. for ``[1, 2]`` it is not clear to which variable
   name the value is bound after the loop header. Such simpe lists
   **can’t be handled by Mustache** and hence are caught during this
   check.

Simply put, this check ensures that the given dict can be represented as
JSON and can be processed by mustache. It is **independent** from the
origin of the dict.

   Notably native mustache actually *can*, in fact, handle plain lists.
   The syntax is not straight-forward though, hence we forbid it here.
   (For details see `mustache spec, “Implicit
   Iterators” <https://github.com/mustache/spec/blob/master/specs/sections.yml#L179>`__)

Schema Check
~~~~~~~~~~~~

This check is implemented in ``RenderedObject.get_rendering_context()``
as is performed on the dict returned by ``_get_serialized()``.

It ensures that the structure of this object conforms to a predefined
schema **associated with the generating class**.

   A *schema* in general defines a structure which some data follows.
   E.g. OpenPMD can be seen as a schema. Some database management
   systems call their tables *schemas*, XML has *schemas* etc. For the
   JSON data here `JSON schema <https://json-schema.org/>`__ is used. At
   the time of writing, JSON schema has not been standardized into an
   RFC, and the version **Draft 2020-12** is used throughout. To check
   the schemas, the library
   `jsonschema <https://github.com/Julian/jsonschema>`__ is employed.

   `live online JSON schema
   validator <https://www.jsonschemavalidator.net/>`__ \| `comprehensive
   guid <https://json-schema.org/understanding-json-schema/index.html>`__

The schemas that are checked against are located at
``share/pypicongpu/schema/``. All files in this directory are crawled
and added into the schema database. **One file may only define one
schema.**

To associate the classes to their schemas, their **Fully Qualified
Name** (FQN) is used. It is constructed from the modulename and the
class name, i.e. the PyPIConGPU simulation object has the FQN
``pypicongpu.simulation.Simulation``. The FQN is appended to the URL
``https://registry.hzdr.de/crp/picongpu/schema/`` which is used as
identifier (``$id``) of a schema.

E.g. the Yee Solver class’ schema is defined as:

.. code:: json

   {
       "$id": "https://registry.hzdr.de/crp/picongpu/schema/pypicongpu.solver.YeeSolver",
       "type": "object",
       "properties": {
           "name": {
               "type": "string"
           }
       },
       "required": ["name"],
       "unevaluatedProperties": false
   }

which is fullfilled by it serialization:

.. code:: json

   {"name": "Yee"}

The URL
(``https://registry.hzdr.de/crp/picongpu/schema/pypicongpu.solver.YeeSolver``)
can be used to refer to a serialized YeeSolver, e.g. by the PyPIConGPU
Simulation schema.

For all schema files, the following is checked:

-  Have an ``$id`` (URL) set (if not log error and skip file)
-  Have ``unevaluatedProperties`` set to ``false``, i.e. do not allow
   additional properties (if not log warning and continue)

If no schema can be found when translating an object to JSON operation
is aborted.

JSON preprocessor
~~~~~~~~~~~~~~~~~

If the created context object (JSON) passes all checks (structure +
schema) it is passed to the preprocessor.

   Before any preprocessing is applied (but after all checks have
   passed) the runner dumps the used context object into
   ``pypicongpu.json`` inside the setup directory.

The preprocessor performs the following tasks:

-  Translate all numbers to C++-compatible literals (stored as strings,
   using sympy)
-  Add the properties ``_first`` and ``_last`` to all list items, set to
   ``true`` or ``false`` respectively (to ease generation of list
   separators etc.)
-  Add top-level attributes, e.g. the current date as ``_date``.

Rendering Process
~~~~~~~~~~~~~~~~~

The rendering process itself is launched inside of Runner in
``Runner.generate()``. This creates the “setup directory” by copying it
from the template, which contains many ``NAME.mustache`` files. These
are the actual string templates, which will be rendered by the
templating engine.

After the setup directory copy the following steps are performed (inside
of ``Runner.__render_templates()``):

1. Retrieve rendering context of the all-encompassing PyPIConGPU
   simulation object

   -  The PyPIConGPU simulation object is responsible for calling
      translate-to-rendering-context methods of the other objects.
   -  This automatically (implicitly) checks against the JSON schema

2. The rendering context general structure is checked (see above)
3. Dump the fully checked rendering context into ``pypicongpu.json`` in
   the setup dir
4. preprocess the context (see above)
5. Render all files ``NAME.mustache`` to ``NAME`` using the context
   (including all child dirs)

   -  check that file ``NAME`` mustache does not exist, if it does abort
   -  check syntax according to rules outlined below, if violated warn
      and continue
   -  print warning on undefined variables and continue

6. rename the fully rendered template ``NAME.mustache`` to
   ``.NAME.mustache``; rationale:

   -  keep it around for debugging and investigation
   -  these files are fairly small, they don’t hurt (too bad)
   -  hide it from users, so they don’t confuse the generated file
      (which is actually used by PIConGPU) with the template
      ``NAME.mustache`` (which is entirely ignored by PIConGPU)

Due to the warnings on undefined variables optional parameters are
should **never** be omitted, but explicitly set to null if unused. E.g.
the laser in the PyPIConGPU simulation is expected by this (sub-)
schema:

.. code:: json

   "laser": {
       "anyOf": [
           {
               "type": "null"
           },
           {
               "$ref": "https://registry.hzdr.de/crp/picongpu/schema/pypicongpu.laser.GaussianLaser"
           }
       ]
   }

which makes both ``{..., "laser": null, ...}`` and
``{..., "laser": {...}, ...}`` valid – but in both cases the variable
``laser`` is defined.

Notably, from this process’ perspective “the rendering context” is the
rendering context of the PyPIConGPU simulation object.

.. _pypicongpu-translation-mustache:

Mustache Syntax
~~~~~~~~~~~~~~~

Mustache syntax is used as defined by the `Mustache
Spec <https://github.com/mustache/spec>`__, whith the following
exceptions:

   The `Mustache Language
   Documentation <https://mustache.github.io/mustache.5.html>`__ is the
   human-readable explanation, though it omits some details.

-  Variables are always inserted using **3** braces: ``{{{value}}}``.
   Using only two braces indicates that the value should be
   HTML-escaped, which is not applicable to this code generation. Before
   rendering, all code is checked by a (not-fully correct) regex, and if
   only two braces are found a warning is issued.
-  Subcomponents of objects can be accessed using the dot ``.``,
   e.g. ``nested.object.value`` returns ``4`` for
   ``{"nested": {"object": {"value": "4"}}}``. This is a mustache
   standard feature and therefore supported by the used library chevron,
   though it is not mentioned in the documentation linked above.
-  Unkown variables are explicitly warned about. Standard behavior would
   be to pass silently, treating them as empty string. Notably this also
   applies to variables used in conditions, e.g. ``{{^laser}}`` would
   issue a warning if laser is not set. Due to that **all used
   variables** should **always** be defined, if necessary set to null
   (``None`` in Python).
-  Partials are not available
-  Lambdas are not available

Example Sequence
----------------

These examples should demonstrate how the translation process works.

.. _pypicongpu-translation-example-boundingbox:

Bounding Box
~~~~~~~~~~~~

The Bounding Box is defined as a grid object. PICMI does not use a
global grid object, but PIConGPU does.

So when invoking ``picmi.Simulation.get_as_pypicongpu()`` it uses the
grid from the picmi solver:

.. code:: python

   # pypicongpu simulation
   s = simulation.Simulation()
   s.grid = self.solver.grid.get_as_pypicongpu()

``picmi.grid.get_as_pypicongpu()`` (here ``Cartesian3DGrid``) checks
some compatibility stuff, e.g. if the lower bound is correctly set to
0,0,0 (only values supported) and if the upper and lower boundary
conditions are the same for each axis. For unsupported features a util
method is called:

.. code:: python

   assert [0, 0, 0] == self.lower_bound, "lower bounds must be 0, 0, 0"
   assert self.lower_boundary_conditions == self.upper_boundary_conditions, "upper and lower boundary conditions must be equal (can only be chosen by axis, not by direction)"

   # only prints a message if self.refined_regions is not None
   util.unsupported("refined regions", self.refined_regions)

PIConGPU does not use bounding box + cell count but cell count + cell
size, so this is translated before returning a pypicongpu grid:

.. code:: python

   # pypicongpu grid
   g = grid.Grid3D()
   g.cell_size_x_si = (self.xmax - self.xmin) / self.nx
   g.cell_size_y_si = (self.ymax - self.ymin) / self.ny
   g.cell_size_z_si = (self.zmax - self.zmin) / self.nz
   g.cell_cnt_x = self.nx
   g.cell_cnt_y = self.ny
   g.cell_cnt_z = self.nz
   # ...
   return g

The pypicongpu ``Grid3D._get_serialized()`` now translates these
parameters to JSON (a python dict):

.. code:: python

   def _get_serialized(self) -> dict:
       return {
           "cell_size": {
               "x": self.cell_size_x_si,
               "y": self.cell_size_y_si,
               "z": self.cell_size_z_si,
           },
           "cell_cnt": {
               "x": self.cell_cnt_x,
               "y": self.cell_cnt_y,
               "z": self.cell_cnt_z,
           },
           "boundary_condition": {
               "x": self.boundary_condition_x.get_cfg_str(),
               "y": self.boundary_condition_y.get_cfg_str(),
               "z": self.boundary_condition_z.get_cfg_str(),
           }
       }

By invoking ``grid.get_rendering_context()`` in the owning
``Simulation`` object this is checked against the schema located in
``share/pypicongpu/schema/pypicongpu.grid.Grid3D.json``

.. code:: json

    {
       "$id": "https://registry.hzdr.de/crp/picongpu/schema/pypicongpu.grid.Grid3D",
       "description": "Specification of a (cartesian) grid of cells with 3 spacial dimensions.",
       "type": "object",
       "properties": {
           "cell_size": {
               "description": "width of a single cell in m",
               "type": "object",
               "unevaluatedProperties": false,
               "required": ["x", "y", "z"],
               "properties": {
                   "x": {
                       "$anchor": "cell_size_component",
                       "type": "number",
                       "exclusiveMinimum": 0
                   },
                   "y": {
                       "$ref": "#cell_size_component"
                   },
                   "z": {
                       "$ref": "#cell_size_component"
                   }
               }
           },
           "cell_cnt": {},
           "boundary_condition": {
               "description": "boundary condition to be passed to --periodic (encoded as number)",
               "type": "object",
               "unevaluatedProperties": false,
               "required": ["x", "y", "z"],
               "properties": {
                   "x": {
                       "$anchor": "boundary_condition_component",
                       "type": "string",
                       "pattern": "^(0|1)$"
                   },
                   "y": {
                       "$ref": "#boundary_condition_component"
                   },
                   "z": {
                       "$ref": "#boundary_condition_component"
                   }
               }
           }
       },
       "required": [
           "cell_size",
           "cell_cnt",
           "boundary_condition"
       ],
       "unevaluatedProperties": false
   }

This entire process has been launched by the ``Runner``, which now dumps
these parameters to the ``pypicongpu.json`` before continuing rendering:

.. code:: json

   {
       "grid": {
           "cell_size": {
               "x": 1.776e-07,
               "y": 4.43e-08,
               "z": 1.776e-07
           },
           "cell_cnt": {
               "x": 192,
               "y": 2048,
               "z": 12
           },
           "boundary_condition": {
               "x": "0",
               "y": "0",
               "z": "1"
           }
       },
   }

These are now used by the template from ``share/pypicongpu/template``,
e.g. in the ``N.cfg.mustache``:

.. code:: bash

   {{#grid.cell_cnt}}
   TBG_gridSize="{{{x}}} {{{y}}} {{{z}}}"
   {{/grid.cell_cnt}}

   TBG_steps="{{{time_steps}}}"

   {{#grid.boundary_condition}}
   TBG_periodic="--periodic {{{x}}} {{{y}}} {{{z}}}"
   {{/grid.boundary_condition}}

and in ``grid.param.mustache``:

.. code::

   constexpr float_64 DELTA_T_SI = {{{delta_t_si}}};

   {{#grid.cell_size}}
   constexpr float_64 CELL_WIDTH_SI = {{{x}}};
   constexpr float_64 CELL_HEIGHT_SI = {{{y}}};
   constexpr float_64 CELL_DEPTH_SI = {{{z}}};
   {{/grid.cell_size}}
