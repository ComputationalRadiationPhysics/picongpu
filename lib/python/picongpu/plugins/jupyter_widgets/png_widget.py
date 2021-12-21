"""
This file is part of the PIConGPU.

Copyright 2017-2021 PIConGPU contributors
Authors: Sebastian Starke
License: GPLv3+
"""

from .base_widget import BaseWidget
from picongpu.plugins.plot_mpl import PNGMPL

from ipywidgets import widgets


class PNGWidget(BaseWidget):
    """
    From within jupyter notebook this widget can be used in the following way:

        %matplotlib widget
        import matplotlib.pyplot as plt
        plt.ioff() # deactivate instant plotting is necessary!

        from picongpu.plugins.jupyter_widgets import PNGWidget

        display(PNGWidget(run_dir_options="path/to/outputs"))
    """
    def __init__(self, run_dir_options, fig=None,
                 output_widget=None, **kwargs):

        BaseWidget.__init__(self,
                            PNGMPL,
                            run_dir_options,
                            fig,
                            output_widget,
                            **kwargs)

    def _create_sim_dropdown(self, options):
        """
        The widget for displaying the available simulations.
        This should be a Dropdown since the underlying
        visualizer can not deal with multiple run_directories!
        """
        sim_drop = widgets.Dropdown(
            description="Sims", options=options, value=None)

        return sim_drop

    def _create_widgets_for_vis_args(self):
        """
        Create the widgets that are necessary for adjusting the
        visualization parameters of this special visualizer.

        Returns
        -------
        a dict mapping keyword argument names of the PIC visualizer
        to the jupyter widgets responsible for adjusting those values.
        """
        self.species = widgets.Dropdown(description="Species",
                                        options=["e"],
                                        value='e')
        self.species_filter = widgets.Dropdown(description="Species_filter",
                                               options=['all'],
                                               value="all")
        self.axis = widgets.Dropdown(description="Axis",
                                     options=["yx", "yz"],
                                     value="yx")

        return {'species': self.species,
                'species_filter': self.species_filter,
                'axis': self.axis}
