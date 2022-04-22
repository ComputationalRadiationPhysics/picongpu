How to Add a Species Constant (“Flag”)
======================================

Species constants are constants associated to all particles of a
species. Hence they are included in the PIConGPU species type
definition, located in the “Species Flags”.

   Replace ``CONST`` with your new constant.

1. add constant in ``lib/python/picongpu/pypicongpu/species/constant/CONST.py``

   -  inherit from ``Constant``

2. add test, checking **at least** for:

   -  rendering from ``get_rendering_context()``
   -  check (even if check is empty)
   -  typesafety
   -  (possibly empty) dependencies on species, attributes, other constants

3. (implement tests, add schema)
4. add constant passthrough test in species test

   -  add constant in map ``expected_const_by_name`` in test
      ``test_rendering_constants()``
   -  background: constants are not automatically exposed by the
      PyPIConGPU species class

5. (implement test by adding passthrough from species.py)
6. write PICMI test

   -  new constant is created from PICMI

7. (implement test)
8. adjust code generation

   -  keep in mind that your new constant is optional, i.e. use

      ::

         {{#constants.CONST}}
            my-code-to-generate-CONST = {{{value}}};
         {{/constants.CONST}}

9. create & compile an example

Note that constants are not stored in a list, but every constant type
has a key in a dictionary associated to it. This means **when adding a
new constant** the **species class** has to be adjusted to pass the new
constant into its rendering context.
