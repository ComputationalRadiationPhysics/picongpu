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

Visualizer
~~~~~~~~~~

The visualizers should reside in the ``lib/python/picongpu/plugins/plot_mpl/`` directory.
The module names should end on ``_visualizer.py`` and the class name should only be ``Visualizer``.

To shorten the import statements for the visualizers, please also add an entry in the ``__init__.py`` file of the ``plot_mpl`` directory.

There is a base class for visualization found in ``base_visualizer.py`` which already handles the plotting logic.
It uses the data reader classes for accessing the data.
After getting the data, it ensures that (for performance reasons) a matplotlib artist is created only for the first plot and later only gets updated with fresh data.

.. autoclass:: picongpu.plugins.plot_mpl.base_visualizer.Visualizer
   :members:
   :private-members:

The complete implementation logic of the ``visualize`` function is pretty simple.

.. code:: python

   def visualize(self, **kwargs):
        self.data = self.data_reader.get(**kwargs)
        if self.plt_obj is None:
            self._create_plt_obj()
        else:
            self._update_plt_obj()

All new plugins should derive from this class.

When implementing a new visualizer you have to perform the following steps:

1. Let your visualizer class inherit from the ``Visualizer`` class in ``base visualizer.py``.

2. Implement the ``_create_data_reader(self, run_directory)`` function.
This function should return a data reader object (see above) for this plugin's data.

3. Implement the ``_create_plt_obj(self)`` function.
This function needs to access the plotting data from the ``self.data`` member (this is the data structure as returned by the data readers ``.get(...)`` function, create some kind of matplotlib artist by storing it in the ``self.plt_obj`` member variable and set up other plotting details (e.g. a colorbar).

4. Implement the ``_update_plt_obj(self)`` function.
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
It also implements widgets for scan, simulation and iteration selection.

.. autoclass:: picongpu.plugins.jupyter_widgets.base_widget.BaseWidget
   :members:
   :private-members:

When created with an ``experiment_path`` argument, it assumes that there will be a number of subdirectories containing scan results named ``scan_<nr>``.
Within each such directory it further assumes one or many PIConGPU simulation output directories named ``sim_<nr>``.
The base class provides functionality to easily allow switching between visualizations for different scans or simulations.

When implementing a new widget you have to perform the following steps:

1. Let the widget class inherit from the base visualizer

.. code:: python

   from .base_visualizer import BaseWidget

   class NewPluginWidget(BaseWidget):

2. In the constructor, call the base class constructor with the matplotlib visualizer class as ``pic_vis_cls`` keyword.

   The base class will then create an instance of the visualizer class and delegate the plotting job to it.

.. code:: python

   # taken from lib/python/picongpu/plugins/jupyter_widgets/energy_histogram_visualizer.py
   from .base_visualizer import BaseWidget
   from picongpu.plugins.plot_mpl import EnergyHistogramMPL

   class EnergyHistogramWidget(BaseWidget):
       def __init__(self, experiment_path, fig=None, your_additional_args):

           BaseWidget.__init__(
               self,
               experiment_path,
               pic_vis_cls=EnergyHistogramMPL,
               fig=fig)
           # do stuff with your_additional_args


3. (optional) adjust the ``_basic_ui_elements(self)`` function.
    The base class already creates dropdown widgets for interactive selection of scan and simulation directories as well as a slider to adjust the simulation time (i.e. the iteration).
    If your class does not need some of those widgets, feel free to override this function.
    It needs to return a list of widgets.
    Note that scan and simulation dropdowns should probably be returned in any case to allow switching between different simulations.

4. implement the  ``_extra_ui_elements(self)`` function.

    This function has to define jupyter widgets as member variables of the class to allow interactive manipulation of parameters the underlying matplotlib visualizer is capable of handling.
    It needs to return a list of those widgets.

.. code:: python

   # taken from lib/python/picongpu/plugins/jupyter_widgets/energy_histogram_visualizer.py
   def _extra_ui_elements(self):
       """
       Create and return as a list the individual widgets that are necessary
       for this special visualizer.
       """
       self.species = widgets.Dropdown(description="Species",
                                       options=["e"],
                                       value='e')
       self.species_filter = widgets.Dropdown(description="Species_filter",
                                              options=['all'],
                                              value="all")
       return [self.species, self.species_filter]

5. implement the ``_extra_ui_elements_value(self)`` function.

    This function acts as the connection between the widgets specified above and transfering their values to the underlying matplotlib visualizer instance.
    It therefore needs to return a dictionary with an entry for every widget specified in 4.
    The keys must be the parameter names that the matplotlib visualizer accepts in its call to the ``visualize`` function.
    The values will be the corresponding values of the widget elements for this parameter.

.. code:: python

   # taken from lib/python/picongpu/plugins/jupyter_widgets/energy_histogram_visualizer.py
   def _extra_ui_elements_value(self):
       """
       Map widget values to parameter names for the plot_mpl visualizer.
       """
       return {'species': self.species.value,
               'species_filter': self.species_filter.value}
