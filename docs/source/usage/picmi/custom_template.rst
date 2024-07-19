.. _picmi-custom-generation:

Custom Code Generation
======================

The PICMI interface of PIConGPU uses code generation to generate the c++ ``.param`` and bash ``.cfg`` files that form a PIConGPU setup.

This code generation is done by combining a *template*, a pattern with placeholders for variables into which values are to be substituted, and a *context*, a dictionary of assigning placeholder names a value.
The result is a *render* of the template for the given *context*.

In the case of PIConGPU the *context* is a JSON representation of the data contained in the python simulation object and the *template* is a pre-written c++/bash file with placeholders for values to be substituted.
In addition to pure substitution we may also include a code block zero or more times, depending on the *context*, with each instance of a code block possibly using different values for substitution.

Usually the PIConGPU PICMI interface uses a fixed set of *templates* but it is also possible to supply **custom templates** to **modify code generation**, to for example enable additional output or allow additional configuration.

Fundamentally there are three different approaches to custom template, differing in complexity, maintenance overhead and flexibility:

- hard coded custom templates
- variable custom templates
- variable custom templates utilising custom user input

To get started take a look at the examples below, have a look at :ref:`step-by-step guide <step_by_step>`, or have at the detailed :ref:`rendering example <pypicongpu-translation-example-boundingbox>` in the developer documentation.

Hard Coded Custom templates
---------------------------

A Hard coded custom template differs from the default template only by the addition/change of template parts that do not depend on the *context*.
This is usually the easiest type of custom templates, since only an understanding of standard picongpu templates is required.

An example of a hard coded custom template would be a change of the output from the PIConGPU simulation.

The output of the picongpu simulation in the default templates is configured in the file ``N.cfg.mustache`` in the `lines 83-109  <https://github.com/ComputationalRadiationPhysics/picongpu/blob/c16e76a00dc36fe413dbbaae7d8611a5c732169d/share/picongpu/pypicongpu/template/etc/picongpu/N.cfg.mustache#L83-L109>`_.

.. code:: bash

  pypicongpu_output_with_newlines="
      {{#output.auto}}
          --fields_energy.period {{{period}}}
          --sumcurr.period {{{period}}}
          $USED_CHARGE_CONSERVATION_FLAGS


          {{#species_initmanager.species}}
              --{{{name}}}_macroParticlesCount.period {{{period}}}

              --{{{name}}}_energy.period {{{period}}}
              --{{{name}}}_energy.filter all

              --{{{name}}}_energyHistogram.period {{{period}}}
              --{{{name}}}_energyHistogram.filter all
              --{{{name}}}_energyHistogram.binCount 1024
              --{{{name}}}_energyHistogram.minEnergy 0
              --{{{name}}}_energyHistogram.maxEnergy 256000

              {{#png_axis}}
                  --{{{name}}}_png.period {{{period}}}
                  --{{{name}}}_png.axis {{{axis}}}
                  --{{{name}}}_png.slicePoint 0.5
                  --{{{name}}}_png.folder png_{{{name}}}_{{{axis}}}
              {{/png_axis}}
          {{/species_initmanager.species}}

      {{/output.auto}}

If we now replaces these lines with the following

.. code:: bash

  pypicongpu_output_with_newlines="
      {{#output.auto}}
         --Cu_energyHistogram.period 100
         --Cu_energyHistogram.filter all
         --Cu_energyHistogram.binCount 1024
         --Cu_energyHistogram.minEnergy 0
         --Cu_energyHistogram.maxEnergy 256000
      {{/output.auto}}

We are replacing the previous output for every species with just an energy histogram output every 100 steps for the species ``Cu``, thereby reducing the total amount of data that needs to be stored significantly.
Alternatively we may of course also add a new plugin and hard code its parameters as we wish.

This example shows both the advantages and disadvantages of this approach.

Hard coding values is comparatively easy but not flexible, for example if were to also want a macro particle count from every species of the simulation we would have to add the ``macroParticlesCount`` plugin for each species by hand.

.. code:: bash

  pypicongpu_output_with_newlines="
      {{#output.auto}}
         --Cu_energyHistogram.period 100
         --Cu_energyHistogram.filter all
         --Cu_energyHistogram.binCount 1024
         --Cu_energyHistogram.minEnergy 0
         --Cu_energyHistogram.maxEnergy 256000

         --Cu_macroParticlesCount.period 1
         --eth_macroParticlesCount.period 1
         --Ni_macroParticlesCount.period 1
      {{/output.auto}}

And if we add fourth species we have to remember to add them by hand.


Variable Custom Templates
-------------------------

Instead of hard coding the output we might want to automatically generate one instance of the ``macroParticlesCount`` plugin for every species in our simulation for this we modify the above example to the following.

.. code:: bash

  pypicongpu_output_with_newlines="
      {{#output.auto}}
         --Cu_energyHistogram.period 100
         --Cu_energyHistogram.filter all
         --Cu_energyHistogram.binCount 1024
         --Cu_energyHistogram.minEnergy 0
         --Cu_energyHistogram.maxEnergy 256000

          {{#species_initmanager.species}}
              --{{{name}}}_macroParticlesCount.period 1
          {{/species_initmanager.species}}
      {{/output.auto}}

Let's go in detail through the above example.

The ``{{#<property>}}`` indicates the start of a block which ends at the corresponding ``{{/<property>}}`` and that this block will exist if ``<property>`` exists, i.e. it is contained as a key in the *context* and the corresponding value is not ``None``(python)/``Null``(json).
In Addition if ``<property>`` happens to be a list as it is in our case we will repeat the in the block enclosed code once for every entry, ``#`` indicates in fact both an *if* and a *for-each* loop.

Therefore the block ``--{{{name}}}_macroParticlesCount.period 1`` will exist once per species in the simulation.

Once we are in a block we also move to its context, meaning that all further property names are now first sought inside of it
Only we have exhausted all sub-levels will we search for a name in higher levels, like for example the ``output`` level.
In this block we now replace ``{{{name}}}``, replacement indicated by the enclosing ``{{{``, with the name of the species this block belongs to, since we first search for the property ``name`` inside this species block.

To get a deeper understanding of how templates are rendered see the documentation of the :ref:`templating engine *Mustache* <pypicongpu-translation-mustache>`.

And for a deeper understanding of the *context* structure take a look at the upon ``write_input_file()``-call from your PICMI *user script* generated ``pypicongpu.json`` or take a look at the json-schemas describing the general ``pypicongpu.json`` structure located `here <https://github.com/ComputationalRadiationPhysics/picongpu/tree/dev/share/picongpu/pypicongpu/schema>`_.
For further information about schema see `the translation process section on schemas <https://picongpu.readthedocs.io/en/latest/pypicongpu/translation.html#schema-check>`_ and the tutorial on `how to write a schema <https://picongpu.readthedocs.io/en/latest/pypicongpu/howto/schema.html>`_.

Variable Templates with Custom User Input
-----------------------------------------

While powerful variable templates are still limited by the predefined context available for rendering them, a template may not reference a quantity which is not contained in the *context*.

For example we can not easily configure the number of bins of our energy histogram from PICMI, since this information is not encoded in the default *context*.

To circumvent this limitation we may pass create in the *user script* custom user input containing additional global information to the simulation.

.. code:: python
  import picongpu

  # create and configure PICMI simulation object
  picmi_simulation = picongpu.picmi.Simulation( ... )

  # create CustomUserInput object
  custom_input_number_bins = picongpu.pypicongpu.customuserinput.CustomUserInput()
  custom_input_number_bins.addToCustomInput(custom_input={"numberbins": 1023}, tag="energy_histogram_configuration_number_bins")

  # add custom User Input to simulation
  picmi_simulation.picongpu_add_custom_user_input(custom_input_number_bins)

  picmi_simulation.write_to_file()

This input will be included directly in the *context* created by the simulation.

.. code:: json

  pypicongpu.json =
  {
      ...,
      "customuserinput":{
          "numberbins":1
          "tags":["energy_histogram_configuration_number_bins"]
      }
  }

And of course it will be available as such in the rendering of the template which allows us to use it in the template.

.. code:: bash

  pypicongpu_output_with_newlines="
      {{#output.auto}}
         --Cu_energyHistogram.period 100
         --Cu_energyHistogram.filter all
         --Cu_energyHistogram.binCount {{{customuserinput.numberbins}}}
         --Cu_energyHistogram.minEnergy 0
         --Cu_energyHistogram.maxEnergy 256000

          {{#species_initmanager.species}}
              --{{{name}}}_macroParticlesCount.period 1
          {{/species_initmanager.species}}
      {{/output.auto}}

.. warning::

  In contrast to the default **context** the PIConGPU PICMI interface by default does not perform checks on custom user input.

  If you use custom user input, you are responsible for its physical and structural correctness.

Multiple Custom User Inputs
---------------------------
A user may:

- add more than one custom user input to the same simulation
- may add more than one dictionary to the same custom user input in separate ``addToCustomInput()``

so long as the do not conflict, i.e. assign differing values to the same key.

All will be serialized as expected preserving sub sub-structure within a custom input but not between custom inputs as exemplified below,

.. code:: python

  i_1 = customuserinput.CustomUserInput()
  i_2 = customuserinput.CustomUserInput()

  i_1.addToCustomInput({"test_data_1": 1}, "tag_1")
  i_2.addToCustomInput({"test_data_2": 2}, "tag_2")

  i_1.addToCustomInput({"test_data_3": 3}, "tag_3")

  picmi_simulation.add_custom_user_input(i_1)
  picmi_simulation.add_custom_user_input(i_2)

  ---------------------------------------------------------

  simulation.get_rendering_context()["customuserinput"] == {
      "test_data_1": 1,
      "test_data_2": 2,
      "test_data_3": 3,
      "tags" : ["tag_1", "tag_2", "tag_3"]}

Adding User Specfied Checks
---------------------------

A user may define custom checks on his custom user inputs by inheriting from ``picongpu.pypicongpu.customuserinput.CustomUserInput`` and overwriting the ``check()`` method.

See `here <https://github.com/ComputationalRadiationPhysics/picongpu/blob/dev/lib/python/picongpu/pypicongpu/customuserinput.py>`_ for the interface definition and implementation details.

Defining a new Custom User Input Class
--------------------------------------

A user may define a new Implementation of ``picongpu.pypicongpu.customuserinput.InterfaceCustomUserInput`` to sidestep the need to serialize his custom input by hand before passing it to the Simulation and check the serialization with a custom schema.

See `here <https://github.com/ComputationalRadiationPhysics/picongpu/blob/dev/lib/python/picongpu/pypicongpu/customuserinput.py>`_ for the interface definition and implementation details requirements.

Step-by-Step Guide
------------------
.. _step_by_step:

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
