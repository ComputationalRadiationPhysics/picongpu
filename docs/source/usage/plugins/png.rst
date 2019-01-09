.. _usage-plugins-PNG:

PNG
---

This plugin generates **images in the png format** for slices through the simulated volume.
It allows to draw a **species density** together with electric, magnetic and/or current field values.
The exact field values, their coloring and their normalization can be set using ``*.param`` files.
It is a very rudimentary and useful tool to get a first impression on what happens in the simulation and to verify that the parameter set chosen leads to the desired physics.

.. note::

   In the near future, this plugin might be replaced by the ISAAC interactive 3D visualization.

External Dependencies
^^^^^^^^^^^^^^^^^^^^^

The plugin is available as soon as the :ref:`PNGwriter library <install-dependencies>` is compiled in.

.cfg file
^^^^^^^^^

For **electrons** (``e``) the following table describes the command line arguments used for the visualization.

====================== ==========================================================================================================
Command line option    Description
====================== ==========================================================================================================
``--e_png.period``     This flag requires an integer value that specifies at what periodicity the png pictures should be created.
                       E.g. setting ``--e_png.period 100`` generates images for the *0th, 100th, 200th, ...* time step.
                       There is no default.
                       If flags are not set, no pngs are created.
``--e_png.axis``       Set 2D slice through 3D volume that will be drawn.
                       Combine two of the three dimensions ``x``, ``y``and ``z``, the define a slice.
                       E.g. setting ``--e_png.axis yz`` draws both the y and z dimension and performs a slice in x-direction.
``--e_png.slicePoint`` Specifies at what ratio of the total depth of the remaining dimension, the slice should be performed.
                       The value given should lie between ``0.0`` and ``1.0``.
``--e_png.folder``     Name of the folder, where all pngs for the above setup should be stored.
====================== ==========================================================================================================

These flags use ``boost::program_options``'s ``multitoken()``.
Therefore, **several setups** can be specified e.g. to draw different slices.
The order of the flags is important in this case.
E.g. in the following example, two different slices are visualized and stored in different directories:

.. code:: bash

   picongpu [more args]
     # first
     --e_png.period 100
     --e_png.axis xy
     --e_png.slicePoint 0.5
     --e_png.folder pngElectronsXY
     # second
     --e_png.period 100
     --e_png.axis xz
     --e_png.slicePoint 0.5
     --e_png.folder pngElectronsXZ

.param files
^^^^^^^^^^^^

The two param files :ref:`png.param <usage-params-plugins>` and :ref:`pngColorScales.param <usage-params-plugins>` are used to specify the desired output.

**Specifying the field values using** ``png.param``

Depending on the used prefix in the command line flags, electron and/or ion density is drawn.
Additionally to that, three field values can be visualized together with the particle density.
In order to set up the visualized field values, the ``png.param`` needs to be changed.
In this file, a variety of other parameters used for the PngModule can be specified.

The ratio of the image can be set.

.. code:: cpp

   /* scale image before write to file, only scale if value is not 1.0 */
   const double scale_image = 1.0;

   /* if true image is scaled if cellsize is not quadratic, else no scale */
   const bool scale_to_cellsize = true;

In order to scale the image, ``scale_to_cellsize`` needs to be set to ``true`` and ``scale_image`` needs to specify the reduction ratio of the image.

.. note::

   For a 2D simulation, even a 2D image can be a quite heavy output.
   Make sure to reduce the preview size!

It is possible to draw the borders between the GPUs used as white lines.
This can be done by setting the parameter ``white_box_per_GPU`` in ``png.param`` to ``true``

.. code:: cpp

   const bool white_box_per_GPU = true;

There are three field values that can be drawn: ``CHANNEL1``, ``CHANNEL2`` and ``CHANNEL3``.

Since an adequate color scaling is essential, there several option the user can choose from.

.. code:: cpp

   // normalize EM fields to typical laser or plasma quantities
   //-1: Auto: enable adaptive scaling for each output
   // 1: Laser: typical fields calculated out of the laser amplitude
   // 2: Drift: typical fields caused by a drifting plasma
   // 3: PlWave: typical fields calculated out of the plasma freq.,
   // assuming the wave moves approx. with c
   // 4: Thermal: typical fields calculated out of the electron temperature
   // 5: BlowOut: typical fields, assuming that a LWFA in the blowout
   // regime causes a bubble with radius of approx. the laser's
   // beam waist (use for bubble fields)
   #define EM_FIELD_SCALE_CHANNEL1 -1
   #define EM_FIELD_SCALE_CHANNEL2 -1
   #define EM_FIELD_SCALE_CHANNEL3 -1

In the above example, all channels are set to **auto scale**.
**Be careful**, when using a normalization other than auto-scale, depending on your setup, the normalization might fail due to parameters not set by PIConGPU.
*Use the other normalization options only in case of the specified scenarios or if you know, how the scaling is computed.*


You can also add opacity to the particle density and the three field values:

.. code:: cpp

   // multiply highest undisturbed particle density with factor
   float_X const preParticleDens_opacity = 0.25;
   float_X const preChannel1_opacity = 1.0;
   float_X const preChannel2_opacity = 1.0;
   float_X const preChannel3_opacity = 1.0;

and add different coloring:

.. code:: cpp

   // specify color scales for each channel
   namespace preParticleDensCol = colorScales::red;  /* draw density in red */
   namespace preChannel1Col = colorScales::blue;     /* draw channel 1 in blue */
   namespace preChannel2Col = colorScales::green;    /* draw channel 2 in green */
   namespace preChannel3Col = colorScales::none;     /* do not draw channel 3 */

The colors available are defined in ``pngColorScales.param`` and their usage is described below.
If ``colorScales::none`` is used, the channel is not drawn.


In order to specify what the three channels represent, three functions can be defined in ``png.param``.
The define the values computed for the png visualization.
The data structures used are those available in PIConGPU.

.. code:: cpp

   /* png preview settings for each channel */
   DINLINE float_X preChannel1( float3_X const & field_B, float3_X const & field_E, float3_X const & field_J )
   {
       /* Channel1
        * computes the absolute value squared of the electric current */
       return math::abs2(field_J);
   }

   DINLINE float_X preChannel2( float3_X const & field_B, float3_X const & field_E, float3_X const & field_J )
   {
       /* Channel2
        * computes the square of the x-component of the electric field */
       return field_E.x() * field_E.x();
   }

   DINLINE float_X preChannel3( float3_X const & field_B, float3_X const & field_E, float3_X const & field_J )
   {
       /* Channel3
        * computes the negative values of the y-component of the electric field
        * positive field_E.y() return as negative values and are NOT drawn */
       return -float_X(1.0) * field_E.y();
   }

Only positive values are drawn. Negative values are clipped to zero.
In the above example, this feature is used for ``preChannel3``.


**Defining coloring schemes in** ``pngColorScales.param``

There are several predefined color schemes available:

- none (do not draw anything)
- gray
- grayInv
- red
- green
- blue

But the user can also specify his or her own color scheme by defining a namespace with the color name that provides an ``addRGB`` function:

.. code:: cpp

   namespace NameOfColor /* name needs to be unique */
   {
       HDINLINE void addRGB( float3_X& img, /* the already existing image */
                             const float_X value, /* the value to draw */
                             const float_X opacity ) /* the opacity specified */
       {
           /* myChannel specifies the color in RGB values (RedGreenBlue) with
            * each value ranging from 0.0 to 1.0 .
            * In this example, the color yellow (RGB=1,1,0) is used. */
           const float3_X myChannel( 1.0, 1.0, 0.0 );

           /* here, the previously calculated image (in case, other channels have already
            * contributed to the png) is changed.
            * First of all, the total image intensity is reduced by the opacity of this
            * channel, but only in the color channels specified by this color "NameOfColor".
            * Then, the actual values are added with the correct color (myChannel) and opacity. */
           img = img
                 - opacity * float3_X( myChannel.x() * img.x(),
                                       myChannel.y() * img.y(),
                                       myChannel.z() * img.z() )
                 + myChannel * value * opacity;
       }
   }

For most cases, using the predefined colors should be enough.

Memory Complexity
^^^^^^^^^^^^^^^^^

Accelerator
"""""""""""

locally, memory for the local 2D slice is allocated with 3 channels in ``float_X``.

Host
""""

as on accelerator.
Additionally, the master rank has to allocate three channels for the full-resolution image.
This is the original size **before** reduction via ``scale_image``.

Output
^^^^^^

The output of this plugin are pngs stored in the directories specified by ``--e_png.folder`` or ``--i_png.folder``.
There can be as many of these folders as the user wants.
The pngs follow a naming convention:

.. code::

   <species>_png_yx_0.5_002000.png

First, either ``<species>`` names the particle type.
Following the 2nd underscore, the drawn dimensions are given.
Then the slice ratio, specified by ``--e_png.slicePoint`` or ``--i_png.slicePoint``, is stated in the file name.
The last part of the file name is a 6 digit number, specifying the simulation time step, at which the picture was created.
This naming convention allows to put all pngs in one directory and still be able to identify them correctly if necessary.

Analysis Tools
^^^^^^^^^^^^^^

Data Reader
"""""""""""

You can quickly load and interact with the data in Python with:

.. code:: python

   from picongpu.plugins.data import PNGData


   png_data = PNGData('path/to/run_dir')

   # get the available iterations for which output exists
   iters = png_data.get_iterations(species="e", axis="yx")

   # get the available simulation times for which output exists
   times = png_data.get_times(species="e", axis="yx")

   # pngs as numpy arrays for multiple iterations (times would also work)
   pngs = png_data.get(species="e", axis="yx", iteration=iters[:3])

   for png in pngs:
       print(png.shape)

Matplotlib Visualizer
"""""""""""""""""""""

If you are only interested in visualizing the generated png files it is
even easier since you don't have to load the data manually.

.. code:: python

   from picongpu.plugins.plot_mpl import PNGMPL
   import matplotlib.pyplot as plt


   # create a figure and axes
   fig, ax = plt.subplots(1, 1)

   # create the visualizer
   png_vis = PNGMPL('path/to/run_dir', ax)

   # plot
   png_vis.visualize(iteration=200, species='e', axis='yx')

   plt.show()

The visualizer can also be used from the command line by writing

 .. code:: bash

    python png_visualizer.py

with the following command line options

=================================  ==================================================================
Options                            Value
=================================  ==================================================================
-p                                 Path and to the run directory of a simulation.
-i                                 An iteration number
-s                                 Particle species abbreviation (e.g. 'e' for electrons)
-f (optional, defaults to 'e')     Species filter string
-a (optional, defaults to 'yx')    Axis string (e.g. 'yx' or 'xy')
-o (optional, defaults to 'None')  A float between 0 and 1 for slice offset along the third dimension
=================================  ==================================================================

Jupyter Widget
""""""""""""""

If you want more interactive visualization, then start a jupyter notebook and make
sure that ``ipywidgets`` and ``ìpympl`` are installed.

After starting the notebook server write the following

.. code:: python

   # this is required!
   %matplotlib widget
   import matplotlib.pyplot as plt
   # deactivate interactive mode
   plt.ioff()

   from IPython.display import display
   from picongpu.plugins.jupyter_widgets import PNGWidget

   # provide the paths to the simulations you want to be able to choose from
   # together with labels that will be used in the plot legends so you still know
   # which data belongs to which simulation
   w = PNGWidget(run_dir_options=[
           ("scan1/sim4", scan1_sim4),
           ("scan1/sim5", scan1_sim5)])
   display(w)

and then interact with the displayed widgets.
