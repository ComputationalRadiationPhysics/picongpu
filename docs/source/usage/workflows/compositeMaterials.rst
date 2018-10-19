.. _usage-workflows-compositeMaterials:

Definition of Composite Materials
---------------------------------

.. sectionauthor:: Axel Huebl

The easiest way to define a composite material in PIConGPU is starting relative to an idealized full-ionized electron density.
As an example, lets use :math:`\text{C}_{21}\text{H}_{25}\text{N}` (*"8CB"*) with a plasma density of :math:`n_\text{e,max} = 192\,n_\text{c}` contributed by the individual ions relatively as:

* Carbon: :math:`21 \cdot 6 / N_{\Sigma \text{e-}}`
* Hydrogen: :math:`25 \cdot 1 / N_{\Sigma \text{e-}}`
* Nitrogen: :math:`1 \cdot 7 / N_{\Sigma \text{e-}}`

and :math:`N_{\Sigma \text{e-}} = 21_\text{C} \cdot 6_{\text{C}^{6+}} + 25_\text{H} \cdot 1_{\text{H}^+} + 1_\text{N} \cdot 7_{\text{N}^{7+}} = 158`.

Set the idealized electron density in :ref:`density.param <usage-params-core>` as a reference and each species' relative ``densityRatio`` from the list above accordingly in :ref:`speciesDefinition.param <usage-params-core>` (see the input files in the :ref:`FoilLCT example <usage-examples-foilLCT>` for details).

In order to initialize the electro-magnetic fields self-consistently, read :ref:`quasi-neutral initialization <usage-workflows-quasiNeutrality>`.
