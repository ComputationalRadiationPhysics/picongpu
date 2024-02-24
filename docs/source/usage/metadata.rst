.. _usage-metadata:

Dumping Metadata
================

Starting your simulation with

.. literalinclude:: ../../../share/picongpu/tests/metadataFromLaserWakefield/bin/ci.sh
   :language: bash
   :start-after: doc-include-start: cmdline
   :end-before: doc-include-end: cmdline
   :dedent:

will dump a `json`_ respresentation of some metadata to `<filename>`. If no `<filename>` is given, the default value

.. literalinclude:: ../../../include/picongpu/MetadataAggregator.hpp
   :language: C++
   :start-after: doc-include-start: metadata default filename
   :end-before: doc-include-end: metadata default filename
   :dedent:

is used. This feature might in a future revision default to being active.

You might want to dump metadata without actually running a simulation. In this case, you can add the
`--no-start-simulation` flag which will make the code skip the actual simulation.

The dumping happens after all initialisation immediately before the simulation starts (or is skipped). This implies that

 * No dynamic information about the simulation can be included (e.g. information about the state at time step 10).
 * The dumped information will represent the actual parameters used which might be different from the parameters given
   in the input files, e.g., due to :ref:`automatic adjustment<???>`.

.. note::

  Since we are still performing all initialisation work in order to get to the actual parameter values that affect the
  simulation, it might be necessary to run this feature in a (very short) full batch submission with all resources (like
  GPUs, etc.) available as for the full simulation run even when running with `--no-start-simulation`.

The content of the output is a **summary of the physical content of the simulated conditions** and the format is
described :ref:`below <???>`. It is important to note that the structure of the content is aligned with its
categorisation in the physical context and not (enforced to be) aligned with the internal code structure.

.. note::

  The scope of this feature is to provide a human- and machine-readable **summary of the physical content of the
  simulated conditions**. The envisioned use cases are:

    * a theoretician quickly getting an overview over their simulation data,
    * an experimentalist comparing with simulation data or
    * a database using such information for tagging, filtering and searching.

  The following related aspects are out of scope (for the PIConGPU development team):

    * Reproducibility: The only faithful, feature-complete representation of the input necessary to reproduce a
      PIConGPU simulation is the complete input directory. If a more standardised and human-readable repesentation is
      desired, :ref:`PICMI<???>` provides access to a small subset of features.
    * Completeness: This feature is intended to be fed with well-structured information considered important by the
      researchers. It is :ref:`customisable<???>` but the design does not allow to ensure any form of completeness with
      appropriate maintenance effort. We therefore do not aim to describe simulations exhaustively.
    * (De-)Serialisation: We do not provide infrastructure to fully or partially reconstruct C++ objects from the
      dumped information.
    * Standardisation or generalisation of the format: The format and content are the result of our best effort to be
      useful. Any form of standardisation or generalisation beyond this scope requires a resources commitment from
      outside the PIConGPU development team. We are happy to implement any such result.


The Format
----------

The created file is a human-readable text file containing valid `json` the content of which is partially
:ref:`customisable<???>`. We do not enforce a particular format but suggest that you stick as closely as possible to the
naming conventions from :ref:`PyPIConGPU<???>` and :ref:`PICMI<???>`. By default, the output has the following
high-level structure which might be supplemented with further details as appropriate for the described elements of the
simulation:

``???``

Customisation
-------------

Content Creation
^^^^^^^^^^^^^^^^

The main customisation point for adding and adjusting the output related to some `typename TObject`, say a Laser or the
`Simulation` object itself, is providing a specialisation for `picongpu::traits::GetMetadata` defaulting to

.. literalinclude:: ../../../include/picongpu/traits/GetMetadata.hpp
   :language: C++
   :start-after: doc-include-start: GetMetdata trait
   :end-before: doc-include-end: GetMetdata trait
   :dedent:

For example, customising the metadata for `MyClass` with some runtime as well as some compiletime information could look
something like this

.. literalinclude:: ../../../share/picongpu/unit/metadata/metadataDescription.cpp
   :language: C++
   :start-after: doc-include-start: metadata customisation
   :end-before: doc-include-end: metadata customisation
   :dedent:

This can be put anywhere in the code where `MyClass` is known, e.g., in a pertinent `.param` file or directly below the
declaration of `MyClass` itself.

The `json` object returned from `description()` is related to the final output via a `merge_patch`_ operation but we do
not guarantee any particular order in which these are merged. So it is effectively the responsibility of the programmer
to make sure that no metadata entries overwrite each other.

These external classes might run into access restrictions when attempting to dump `private`_ or `protected`_ members.
These can be circumvented in three ways:

1. If `MyClass` already implements a `.metadata()` method, it might already provide the necessary information through
   that interface, e.g.

  .. literalinclude:: ../../../share/picongpu/unit/metadata/metadataDescription.cpp
     :language: C++
     :start-after: doc-include-start: reusing default metadata
     :end-before: doc-include-end: reusing default metadata
     :dedent:

  This is the preferred way of handling this situation (if applicable). The default implementation of
  `picongpu::traits::GetMetadata` forwards to such `.metadata()` methods anyway.

2. Declare `picongpu::traits::GetMetadata<MyClass>` a `friend`_ of `MyClass`, e.g.

  .. literalinclude:: ../../../share/picongpu/unit/metadata/metadataDescription.cpp
     :language: C++
     :start-after: doc-include-start: declare metadata as friend
     :end-before: doc-include-end: declare metadata as friend
     :dedent:

  This way is minimally invasive and preferred if your change is only applicable to your personal situation and is
  not intended to land into mainline.

3. Implement/adjust the `.metadata()` member function of `MyClass`, e.g.

  .. literalinclude:: ../../../share/picongpu/unit/metadata/metadataDescription.cpp
     :language: C++
     :start-after: doc-include-start: adapting metadata
     :end-before: doc-include-end: adapting metadata
     :dedent:

  This method is preferred if your change is general enough to make it into the mainline. If so, you are invited to
  :ref:`open a pull request<???>`. It is also the approach used to provide you with default implementations to build
  upon.

Content Registration
^^^^^^^^^^^^^^^^^^^^

If you are not only adjusting existing output but instead you are adding metadata to a class that did not report any in
the past, this class must register itself **before the simulation starts**. Anything that experiences some form of
initialisation at runtime, e.g., :ref:`plugins <???>` should register themselves after their initialisation. To stick
with the example, a plugin could implement

  .. literalinclude:: ../../../include/picongpu/simulation/control/Simulation.hpp
     :language: C++
     :start-after: doc-include-start: metadata pluginLoad
     :end-before: doc-include-end: metadata pluginLoad
     :dedent:

Classes that only affect compile-time aspects of the program need to be registered in
`include/picongpu/param/metadata.param` by extending the compile-time list `MetadataRegisteredAtCT`. Remember: Their
specialisation of `picongpu::traits::GetMetadata` does not hold a reference and must have a static `description` method.

Classes that get instantiated within a running simulation (and not in the initialisation phase) cannot be included
(because they are dynamic information, see above) unless their exact state could be forseen at compile time in which
case they can be handled exactly as compile-time-only classes.

.. _json: https://www.json.org
.. _merge_patch: https://datatracker.ietf.org/doc/html/rfc7396
.. _private: https://en.cppreference.com/w/cpp/language/access
.. _protected: https://en.cppreference.com/w/cpp/language/access
.. _friend: https://en.cppreference.com/w/cpp/language/friend
