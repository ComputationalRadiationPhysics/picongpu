.. _development-pytools:

Python Postprocessing Tool Structure
====================================

Each plugin should implement at least the following Python classes.

1. A data reader class responsible for loading the data from the simulation directory
2. A visualizer class that outputs a matplotlib plot

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

