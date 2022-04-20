.. _picmi-custom-generation:

Custom Code Generation
======================

The PIConGPU code generation works by encoding all parameters into a JSON representation,
which is combined with a *template* to *render* the final configuration.

You can supply a **custom template** to **modify the generated code**,
e.g. to enable additional output.

.. note::

   It is even possible to use the JSON data during the generation.
   To get started read the :ref:`explanation of the translation process <pypicongpu-translation>`,
   and especially the section on the :ref:`templating engine *Mustache* <pypicongpu-translation-mustache>`.

   To see an example of the JSON data that is used either generate an input set with PICMI and examine the generated ``pypicongpu.json`` or simply have a look at the :ref:`example in the documentation <pypicongpu-translation-example-boundingbox>`.

Step-by-Step Guide
------------------

1. Create a copy of the template: ``cp -r $PICSRC/share/picongpu/pypicongpu/template my_template_dir``
2. Change whatever you need in the template

   e.g. ``vim my_template_dir/etc/picongpu/N.cfg.mustache``

   find ``pypicongpu_output_with_newlines`` and insert:

   .. code::

      --openPMD.period 10
      --openPMD.file simData
      --openPMD.ext bp
      --checkpoint.backend openPMD
      --checkpoint.period 100
      --checkpoint.restart.backend openPMD

3. supply your template dir in your PICMI script:

   .. code:: python

      # other setup...
      sim = picmi.Simulation(
          time_step_size=9.65531e-14,
          max_steps=1024,
          solver=solver,
          # sets custom template dir
          picongpu_template_dir="my_template_dir")

      sim.write_input_file("generated_input")

4. run PICMI script
5. inspect generated files

   e.g. ``less generated_input/etc/picongpu/N.cfg`` now contains the output added above

.. warning::

   It is highly discouraged to incorporate editing generated PIConGPU input files **after** generation -- just because it is very easy to make mistakes this way.
   Try to use the process outlined here.
