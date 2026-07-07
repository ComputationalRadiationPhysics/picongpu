Background
==========

PIConGPU at its core is an advanced, exascale HPC code written in C++.
For maximal performance and configurability,
PIConGPU's core is not distributed and deployed
as an executable binary application but as a source code from which
a highly-optimized tailored executable gets compiled for each simulation setup you want to run.

Core Capabilities
-----------------

The PIConGPU Python frontend is a thin Python wrapper with 3 core capabilities:

#. Translating your simulation setup
   from a high-level, standardized Python representation
   to low-level C++ source code and configuration files
#. Managing your runtime environment
   in order to have the right compilers and optimized libraries
   available on every machine you want to run on
#. Orchestrating the workflow of compiling and running your simulation

Components
----------

The PIConGPU Python package has several, different components
working together to produce your final product.
For the end user, it can be helpful to know their names and functionalities
in discussion or when debugging your input files.
But generally, users should only interact with the topmost frontend:

Frontend/PICMI
  The user-facing frontend is our implementation of the PICMI standard.
  Most users will only ever interact with this component (and the runtime configuration).
  The PICMI standard aims at harmonising the terminology and interfaces to PIC simulation codes.
  Whenever functionality of PIConGPU can be represented in terms of the PICMI standard,
  we strive to expose it in this standard-compliant way.
  When there's no equivalent definition in the PICMI standard,
  we expose it through PIConGPU-specific extensions.

Runtime configuration (RCParams)
  The runtime configuration system handles
  configuration concerns that are orthogonal to your physical/numerical setup.
  These include aspects like which system to run on (compilers, libraries, ...)
  and metadata to be automatically recorded in order to facilitate FAIRness of your research.
  Users are supposed to configure this (roughly) once for each system/project they are on.

PyPIConGPU middlelayer
  Input to the PICMI frontend is translated
  into a middlelayer Python representation called PyPIConGPU.
  Users might occasionally interact with elements from this layer
  either for fine-grained control and customization
  or while debugging their input files.
  It is not considered public interface but should be relatively stable
  because it closely follows core PIConGPU's C++ interface which is mostly stable at this point.

Template layer
  The source code and configuration files are rendered
  from the PyPIConGPU representation via mustache template files.
  Normal users will not interact with these files directly
  but advanced users can write custom template files to expose functionality of PIConGPU's core
  that is not (yet) accessible from the PICMI frontend.
  Such users are kindly asked to contribute back their changes
  to make such functionality available to a broader userbase.

PIConGPU core
  This is our name for the C++ source files (including build system, etc.)
  that comprise the actual PIC code.
  It is distributed as source files when you install the Python package.
  Users are not intended to interact with this component in any form.
  This is developers' realm.

Configuration templates and presets
  We are continuously curating
  a large library of configuration files
  for running PIConGPU on any HPC system our developers have access to.
  They are exposed to the user in the form of simple presets.
  Direct interaction is only intended
  if you are in the process of running PIConGPU on a system for which no preset exists.
  In such cases, we kindly ask to feed back your configuration files into our library.
