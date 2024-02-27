.. _usage-metadata:

Dumping Metadata
================

Starting your simulation with

.. literalinclude:: ../../../share/picongpu/tests/metadataFromLaserWakefield/bin/ci.sh
   :language: bash
   :start-after: doc-include-start: cmdline
   :end-before: doc-include-end: cmdline
   :dedent:

will dump a `json`_ respresentation of some metadata to `${OPTIONAL_FILENAME}`. If no `${OPTIONAL_FILENAME}` is given,
the default value

.. literalinclude:: ../../../include/picongpu/MetadataAggregator.hpp
   :language: C++
   :start-after: doc-include-start: metadata default filename
   :end-before: doc-include-end: metadata default filename
   :dedent:

is used. This feature might in a future revision default to being active.

You might want to dump metadata without actually running a simulation. In this case, you can add the
`--no-start-simulation` flag which will make the code skip the actual simulation. If your intention is instead to also
run the simulation afterwards, just leave it out and the program will proceed as normal.

The dumping happens after all initialisation work is done immediately before the simulation starts (or is skipped). This implies that

 * No dynamic information about the simulation can be included (e.g. information about the state at time step 10).
 * The dumped information will represent the actual parameters used which might be different from the parameters given
   in the input files, e.g., due to automatic adjustment of the grid size (see :ref:`autoAdjustGrid<usage-cfg>`).

.. note::

  Since we are still performing all initialisation work in order to get to the actual parameter values that affect the
  simulation, it might be necessary to run this feature in a (very short) full batch submission with all resources (like
  GPUs, etc.) available as for the full simulation run even when running with `--no-start-simulation`.

The content of the output is a **summary of the physical content of the simulated conditions** and the format is
described :ref:`below <format-subsection>`. It is important to note that the structure of the content is aligned with its
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
      desired, `PICMI`_ provides access to a small subset of features.
    * Completeness: This feature is intended to be fed with well-structured information considered important by the
      researchers. It is :ref:`customisable<customisation-subsection>` but the design does not allow to ensure any form of completeness with
      appropriate maintenance effort. We therefore do not aim to describe simulations exhaustively.
    * (De-)Serialisation: We do not provide infrastructure to fully or partially reconstruct C++ objects from the
      dumped information.
    * Standardisation or generalisation of the format: The format and content are the result of our best effort to be
      useful. Any form of standardisation or generalisation beyond this scope requires a resource commitment from
      outside the PIConGPU development team. We are happy to implement any such result.


.. _format-subsection:

The Format
----------

The created file is a human-readable text file containing valid `json` the content of which is partially
:ref:`customisable<customisation-subsection>`. We do not enforce a particular format but suggest that you stick as
closely as possible to the naming conventions from :ref:`PyPIConGPU<pypicongpu>` and `PICMI`_. For example, the
`LaserWakefield` example dumps the following metadata which might be supplemented with further details as appropriate
for the described elements of the simulation:

.. literalinclude:: ../../../share/picongpu/tests/metadataFromLaserWakefield/picongpu-metadata.json.reference
   :language: json

.. _customisation-subsection:

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

For example, customising the metadata for `MyClass` with some runtime (RT) as well as some compiletime (CT) information could look
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

Tackling Acess Restrictions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
  open a pull request (see `CONTRIBUTING.md`_ for more details). It is also the approach used to provide you with default implementations to build
  upon.

Content Registration
^^^^^^^^^^^^^^^^^^^^

If you are not only adjusting existing output but instead you are adding metadata to a class that did not report any in
the past, this class must register itself **before the simulation starts**. Anything that experiences some form of
initialisation at runtime, e.g., :ref:`plugins <usage-plugins>` should register themselves after their initialisation. To stick
with the example, a plugin could add

  .. literalinclude:: ../../../include/picongpu/simulation/control/Simulation.hpp
     :language: C++
     :start-after: doc-include-start: metadata pluginLoad
     :end-before: doc-include-end: metadata pluginLoad
     :dedent:

at the end of its `pluginLoad()` method (see the `Simulation`_ class or an example).

Classes that only affect compiletime aspects of the program need to be registered in
`include/picongpu/param/metadata.param` by extending the compiletime list `MetadataRegisteredAtCT`. Remember: Their
specialisation of `picongpu::traits::GetMetadata` does not hold a reference.

Classes that get instantiated within a running simulation (and not in the initialisation phase) cannot be included
(because they are dynamic information, see above) unless their exact state could be forseen at compile time in which
case they can be handled exactly as compiletime-only classes.

Metadata Handling Via Policies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is sometimes convenient to have different strategies for handling metadata at hand which can be applied to
independent of the exact content. Despite them not being formal entities in the code, an approach via `policies` can
come in handy. For example, the `AllowMissingMetadata` policy can wrap any CT type in order to handle cases when no
metadata is available. This is its implementation:

.. literalinclude:: ../../../include/picongpu/traits/GetMetadata.hpp
   :language: C++
   :start-after: doc-include-start: AllowMissingMetadata
   :end-before: doc-include-end: AllowMissingMetadata
   :dedent:

Another example is the categorisation of different incident fields which -- by themselves -- cannot report from which
direction they are incident (see the `GetMetadata`_ trait). The `IncidentFieldPolicy` provides a strategy to gather all
pertinent metadata and assemble the `incidentField` subtree of the output by providing the necessary context.

Handling Custom Types
^^^^^^^^^^^^^^^^^^^^^

The `nlohmann-json`_ library in use allows to serialise arbitrary types as described `here`_. As an example, we have
implemented a serialisation for `pmacc::math::Vector` in `GetMetadata`_.

.. _json: https://www.json.org
.. _merge_patch: https://datatracker.ietf.org/doc/html/rfc7396
.. _private: https://en.cppreference.com/w/cpp/language/access
.. _protected: https://en.cppreference.com/w/cpp/language/access
.. _friend: https://en.cppreference.com/w/cpp/language/friend
.. _PICMI: https://picmi-standard.github.io/
.. _CONTRIBUTING.md: https://github.com/ComputationalRadiationPhysics/picongpu/blob/dev/CONTRIBUTING.md
.. _Simulation: https://github.com/ComputationalRadiationPhysics/picongpu/blob/dev/include/picongpu/simulation/control/Simulation.hpp
.. _GetMetadata: https://github.com/ComputationalRadiationPhysics/picongpu/blob/dev/include/picongpu/traits/GetMetadata.hpp
.. _nlohmann-json: https://json.nlohmann.me/
.. _here: https://json.nlohmann.me/features/arbitrary_types/
