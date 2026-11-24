.. _legacy_migration:

Migrating from Legacy ``.param``/``.cfg`` Setups
================================================

.. sectionauthor:: Julian J. Lenz

This page is for users who already run PIConGPU through the legacy C++ interface:
a setup created with ``pic-create``, edited in
``include/picongpu/param/*.param`` and ``etc/picongpu/*.cfg``, built with
``pic-build`` and submitted with ``tbg``.
It maps the legacy files in those setups to the PICMI Python classes that can
produce their content in this repository.

The page is a *user-facing mapping*, not a rewrite of the shipped example
setups. Converting the examples themselves is tracked separately;
the two efforts complement each other.

.. seealso::

   The :ref:`Legacy C++ interface (advanced) <usage-index>` chapter
   still documents the ``.param``/TBG setup you are coming from,
   including the :ref:`TBG documentation <usage-tbg>`.

The layers you are moving between
----------------------------------

A PICMI input file is not rendered directly into C++.
It passes through several layers before it becomes the files the legacy
workflow edited by hand:

* the **PICMI frontend** (``picongpu.picmi``), which you write,
* the runtime configuration (RCParams), which is orthogonal to the physics,
* the **PyPIConGPU** middle layer, which represents the input close to the
  C++ interface,
* the **template layer**, which renders PyPIConGPU objects into
  ``.param``/``.cfg`` files and C++ headers,
* the **PIConGPU core**.

The :ref:`background chapter <python_package/foundations/background:Background>`
describes this architecture in detail.
For migration, the practical consequence is:
whenever a legacy file is still written by the Python package, some PICMI
class feeds a PyPIConGPU object that a mustache template renders into that
file.

Which legacy file maps to which PICMI classes?
-----------------------------------------------

The table below is generated from the mustache templates and the PyPIConGPU
models, not maintained by hand.
For every legacy file it lists the PICMI input classes that can *currently*
produce content in it.
Read it as:
"If I changed file *A* in my legacy setup, which PICMI classes could I use to
express that change in a PICMI input file?"

For class usage and parameters, see the
`PICMI standard documentation <https://picmi.readthedocs.io/en/latest/standard/standard.html>`__
and :ref:`Selected Topics <python_package/selected_topics/index:Selected Topics>`.

.. legacy-migration-table:: files

The ``etc/picongpu/*.cfg`` files collect TBG macros from many independent
inputs and are listed separately:

.. legacy-migration-table:: cfg

Some legacy parameter files have no dedicated template. Their settings are
either folded into one of the files above (for example species constants and
attributes are written by the generated ``speciesDefinition.param``) or they
have no PICMI counterpart yet.

.. legacy-migration-table:: missing

Two remarks on reading the table:

* Several rows list more than one class because a file is fed by multiple
  independent inputs. For example, ``density.param`` is written from the
  density profiles of the distributions, and ``etc/picongpu/*.cfg`` collects
  the settings of many diagnostics.
* The mapping is deliberately *coarse*: it is about which classes can render
  content into a file, not about a one-to-one replacement of every line.
  Names of classes and parameters are those of the PICMI dialect in this
  repository; see the :ref:`API documentation
  <python_package/api/index:API Documentation>` for the exact fields.
