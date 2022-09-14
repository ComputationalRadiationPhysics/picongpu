How to Write a Schema
=====================

General Process
---------------

1. Write test for ``get_rendering_context()``

   -  checks passthru of values
   -  (implictly checks existance and against schema)

2. Run test

   -  must report NameError

3. Make checked object **inherit** from ``RenderedObject``

   -  located at: ``pypicongpu.rendering.RenderedObject``
   -  (use relative imports!)

4. Implement ``_get_serialized(self) -> dict``

   -  if using
      `TDD <https://en.wikipedia.org/wiki/Test-driven_development>`_:
      ``return {}``

5. Run tests

   -  must report ``RuntimeError`` with message ``schema not found``

6. Add schema (see below)
7. Run tests, must report validation error
8. Implement content of ``_get_serialized()``

Schema Writing Checklist
------------------------

All schemas must:

-  top-level object contains the following keys (properties):

   -  ``$id``: URI beginning with
      ``https://registry.hzdr.de/crp/picongpu/schema/`` + *fully qualified name* (FQN, see below)
   -  ``description``: Human-Readable description (avoid mentioning PyPIConGPU internals)
   -  ``type``: ``object``
   -  **also** has keys of next point (b/c is object)

-  every ``type: object`` has

   -  ``unevaluatedProperties``: ``false``, forbids additional properties
   -  ``required``: list of all properties defined, makes all properties mandatory

-  all properties are mandatory (in ``required``)

   -  exception: some other property indicates if said property is present or missing (see below)

-  every field has a ``description`` (unless pure reference ``$ref``)

-  **only one schema per file**, i.e. only one URI

   -  required for schema database loading

-  filename ends in ``.json``

   -  located in ``share/pypicongpu/schema``
   -  (no structure is enforce, typically ``FILE.CLASS.json`` in respective subdirectory)

-  references use absolute URIs (start with ``https://``)

-  When dealing with optional properties use this pattern:

   .. code:: json

      {
        "anyOf": [
          { "type": "null" },
          { "$ref": "URI" }
        ]
      }

   Rationale: As of the time of writing there is a bug in the
   ``jsonschema`` python lib when using ``oneOf``, this is a workaround.

..

   These are not enforced. Some of the points raise warnings if not
   fullfilled, but not all of them.

Schema Writing Background
-------------------------

-  Used version: Draft 2020-12
-  `JSON schema main page <https://json-schema.org/>`_

   -  **recommended** (includes many practical examples):
      `Full Reference <https://json-schema.org/understanding-json-schema/>`_ \|
      `print version <https://json-schema.org/understanding-json-schema/UnderstandingJSONSchema.pdf>`_
   -  `Learn JSON Schema <https://json-schema.org/learn/>`_ \|
      `tutorial <https://json-schema.org/learn/>`_ \|
      `examples <https://json-schema.org/learn/miscellaneous-examples.html>`_

-  `Online live validator <https://www.jsonschemavalidator.net/>`_
-  URIs are used purely for **identification**, i.e. they don’t have to
   resolve to anything meaningful

   -  Avoid collisions at all costs
   -  URI vs URL: uniform resource *identifier* vs. *locator* – here we identify, hence URI
   -  PyPIConGPU URIs: Base URI + classname suffix
   -  base uri ``https://registry.hzdr.de/crp/picongpu/schema/``
   -  suffix: *FQN* (fully qualified name) – full module+class path, let python generate it

-  refer to schemas using ``$ref`` key

   -  `recommended section from manual <https://json-schema.org/understanding-json-schema/structuring.html>`_

-  only use absoulte URIs

   -  (technically speaking relative references would work, but they are prone to introduce headaches)

Polymorphy and Schemas
----------------------

-  Polymorphy: Concept of Object Oriented Programming, where *the same
   name* can refer to *different implementations*.

   -  E.g.: all operations inherit the same ``check_preconditions()``
      method, but each implementation is differnt
   -  E.g.: All density profiles inherit from ``DensityProfile``, but
      have differnt implementations
   -  (not polymorphy: multiple species objects – not polymorphy b/c
      they use the same implementation)

-  Translating every object to a JSON version results in different
   possible versions

   -  e.g.: valid profile could be ``{density_si: 17}`` as well as
      ``{expression: "x+y"}``
   -  solution: explicitly state type
   -  problem: for templating engine, type must be encoded in **key**
      (Mustache does **NOT** access values)

      -  workaround: one key per type

   -  problem: templating engine complains on undefined keys
      (i.e. checks like “if the key exists do this” produce warnings and
      **must be avoided**)

      -  workaround: always define all keys which are checked against
         (these checks indicate if other, optional, keys are available)
      -  (note: fewer optionals make schema definition easier)

-  possible solutions:

   1. externally specify types (e.g. as for density profiles)

      .. code:: json

         {
           "type": {
             "uniform": false,
             "gaussian": true,
             "analytic": false
           },
           "data": { }
         }

   2. one key per type (e.g. as for operations)

      .. code:: json

         {
           "simple_density": [ {"name": "OP1"}, {"name": "OP2"} ],
           "simple_momentum": [ {"name": "OP3"} ]
         }

-  problem: need to provide wrapper which generates these structures

   -  do **NOT** change the rendering interface,
      i.e. ``_get_serialized()`` should **ONLY** provide the data (not
      the type)
   -  DO: add extra wrapper method
      (e.g. ``get_generic_profile_rendering_context()`` for profiles,
      calls original rendering interface internally)
   -  DO: add type data from rendering object (e.g. as in init manager)
