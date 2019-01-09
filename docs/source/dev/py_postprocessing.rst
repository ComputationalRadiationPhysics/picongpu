.. _development-pytools:

Python Postprocessing Tool Structure
====================================

Each plugin should implement at least the following Python classes.

1. A data reader class responsible for loading the data from the simulation directory
2. A visualizer class that outputs a matplotlib plot
3. A jupyter-widget class that exposes the parameters of the matplotlib visualizer to the user via other widgets.

The repository directory for PIConGPU Python modules for plugins is ``lib/python/picongpu/plugins/``.

Data Reader
~~~~~~~~~~~

The data readers should reside in the ``lib/python/picongpu/plugins/data`` directory.
There is a base class in ``base_reader.py`` defining the interface of a reader.
Each reader class should derive from this class and implement the interface functions not implemented in this class.

.. autoclass:: picongpu.plugins.data.base_reader.DataReader
   :members:
   :private-members:

To shorten the import statements for the readers, please also add an entry in the ``__init__.py`` file of the ``data`` directory.

Matplotlib visualizer
~~~~~~~~~~~~~~~~~~~~~

The visualizers should reside in the ``lib/python/picongpu/plugins/plot_mpl/`` directory.
The module names should end on ``_visualizer.py`` and the class name should only be ``Visualizer``.

To shorten the import statements for the visualizers, please also add an entry in the ``__init__.py`` file of the ``plot_mpl`` directory with an alias that ends on "MPL".

There is a base class for visualization found in ``base_visualizer.py`` which already handles the plotting logic.
It uses (possibly multiple) instances of the data reader classes for accessing the data. Visualizing data simultaneously for more than one scan is supported by creating as many readers and plot objects as there are simulations for visualization.
After getting the data, it ensures that (for performance reasons) a matplotlib artist is created only for the first plot and later only gets updated with fresh data.

.. autoclass:: picongpu.plugins.plot_mpl.base_visualizer.Visualizer
   :members:
   :private-members:

All new plugins should derive from this class.

When implementing a new visualizer you have to perform the following steps:

1. Let your visualizer class inherit from the ``Visualizer`` class in ``base visualizer.py`` and call the base class constructor with the correct data reader class.

2. Implement the ``_create_plt_obj(self, idx)`` function.
This function needs to access the plotting data from the ``self.data`` member (this is the data structure as returned by the data readers ``.get(...)`` function, create some kind of matplotlib artist by storing it in the ``self.plt_obj`` member variable at the correct index specified by the idx variable (which corresponds to the data of the simulation at position idx that is passed in construction.

3. Implement the ``_update_plt_obj(self, idx)`` function.
This is called only after a valid ``self.plt_obj`` was created.
It updates the matplotlib artist with new data.
Therefore it again needs to access the plotting data from the ``self.data`` member and call the data update API for the matplotlib artist (normally via ``.set_data(...)``.

Jupyter Widget
~~~~~~~~~~~~~~

The widget is essentially only a wrapper around the matplotlib visualizer that allows dynamical adjustment of the parameters the visualizer accepts for plotting.
This allows to adjust e.g. species, filter and other plugin-dependent options without having to write new lines of Python code.

The widgets should reside in the ``lib/python/picongpu/plugins/jupyter_widgets/`` directory.
The module names should end on ``_widget.py``.

To shorten the import statements for the widgets, please also add an entry in the ``__init__.py`` file of the ``jupyter_widget`` directory.

There is a base class for visualization found in ``base_widget.py`` which already handles most of the widget logic.

.. autoclass:: picongpu.plugins.jupyter_widgets.base_widget.BaseWidget
   :members:
   :private-members:

It allows to switch between visualizations for different simulation times (iterations) and different simulations.

When implementing a new widget you have to perform the following steps:

1. Let the widget class inherit from the ``BaseWidget`` class in ``base_widget.py`` and call the base class constructor with the correct matplotlib visualizer class.

.. code:: python

   from .base_widget import BaseWidget

   class NewPluginWidget(BaseWidget):

2. In the constructor, call the base class constructor with the matplotlib visualizer class as ``plot_mpl_cls`` keyword.

   The base class will then create an instance of the visualizer class and delegate the plotting job to it.

.. code:: python

   # taken from lib/python/picongpu/plugins/jupyter_widgets/energy_histogram_widget.py
   from .base_widget import BaseWidget
   from picongpu.plugins.plot_mpl import EnergyHistogramMPL

   class EnergyHistogramWidget(BaseWidget):
       def __init__(self, run_dir_options, fig=None, **kwargs):

            BaseWidget.__init__(self,
                                EnergyHistogramMPL,
                                run_dir_options,
                                fig,
                                **kwargs)

3. implement the  ``_create_widgets_for_vis_args(self)`` function.

    This function has to define jupyter widgets as member variables of the class to allow interactive manipulation of parameters the underlying matplotlib visualizer is capable of handling.
    It needs to return a dictionary using the parameter names the matplotlib visualizer accepts as keys and the widget members that correspond to these parameters as values.

.. code:: python

   # taken from lib/python/picongpu/plugins/jupyter_widgets/energy_histogram_widget.py
    def _create_widgets_for_vis_args(self):
        # widgets for the input parameters
        self.species = widgets.Dropdown(description="Species",
                                        options=["e"],
                                        value='e')
        self.species_filter = widgets.Dropdown(description="Species_filter",
                                               options=['all'],
                                               value="all")

        return {'species': self.species,
                'species_filter': self.species_filter}
