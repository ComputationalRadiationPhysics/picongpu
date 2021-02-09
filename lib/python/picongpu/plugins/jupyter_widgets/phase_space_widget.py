"""
This file is part of the PIConGPU.

Copyright 2017-2021 PIConGPU contributors
Authors: Sebastian Starke
License: GPLv3+
"""

from .base_widget import BaseWidget
from picongpu.plugins.plot_mpl import PhaseSpaceMPL

from ipywidgets import widgets


class PhaseSpaceWidget(BaseWidget):
    """
    From within jupyter notebook this widget can be used in the following way:

        %matplotlib widget
        import matplotlib.pyplot as plt
        plt.ioff() # deactivate instant plotting is necessary!

        from picongpu.plugins.jupyter_widgets import PhaseSpaceWidget

        display(PhaseSpaceWidget(run_dir_options="path/to/outputs"))
    """
    def __init__(self, run_dir_options, fig=None,
                 output_widget=None, **kwargs):

        BaseWidget.__init__(self,
                            PhaseSpaceMPL,
                            run_dir_options,
                            fig,
                            output_widget,
                            **kwargs)

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
        self.ps = widgets.Dropdown(description="ps",
                                   options=['ypy'],
                                   value='ypy')

        return {'species': self.species,
                'species_filter': self.species_filter,
                'ps': self.ps}
