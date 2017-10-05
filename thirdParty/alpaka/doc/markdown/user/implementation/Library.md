[:arrow_up: Up](../Implementation.md)

Library Interface
=================

As described in the chapter about the Abstraction, the general design of the library is very similar to *CUDA* and *OpenCL* but extends both by some points, while not requiring any language extensions.
General interface design as well as interface implementation decisions differentiating *alpaka* from those libraries are described in the Rationale section.
It uses C++ because it is one of the most performant languages available on nearly all systems.
Furthermore, C++11 allows to describe the concepts in a very abstract way that is not possible with many other languages.
The *alpaka* library extensively makes use of advanced functional C++ template meta-programming techniques.
The Implementation Details  section discusses the C++ library and the way it provides extensibility and optimizability.

1. [Structure](library/Structure.md)
2. [Usage](library/Usage.md)
2. [Rationale](library/Rationale.md)
3. [Details](library/Details.md)
