"""
This file is part of the PIConGPU.

Copyright 2017-2021 PIConGPU contributors
Authors: Sebastian Starke
License: GPLv3+
"""

import matplotlib.pyplot as plt

from ipywidgets import widgets
from datetime import datetime
from warnings import warn


class BaseWidget(widgets.VBox):
    """
    Basic widget class that wraps a corresponding plot_mpl visualizer.
    It handles selection of scans, simulations and iterations.
    It also allows to expose the parameters of the
    corresponding plot_mpl visualizer via jupyter widgets to the user.
    Only classes derived from this base class should be used!

    Note: In order to work, those objects should be used in %matplotlib widget
    mode and interactive plotting should be switched off (by using plt.ioff()).
    """
    def capture_output(fun):
        """Used as decorator for capturing output of member functions."""
        def decorated_member_fun(self, *args, **kwargs):
            self.output_widget.clear_output()
            with self.output_widget:
                return fun(self, *args, **kwargs)

        return decorated_member_fun

    def __init__(self, plot_mpl_cls, run_dir_options=None, fig=None,
                 output_widget=None, **kwargs):
        """
        Parameters
        ----------
        run_dir_options: list
            list of tuples with label and filepath
        plot_mpl_cls: a valid plot_mpl class handle (not string!)
            Specifies the underlying plot_mpl visualizer that will be used.
        fig: a matplotlib figure instance.(optional)
            If no figure is provided, the widget will create its own figure.
            Otherwise the widget will draw into the provided figure and will
            not create its own.
        output_widget: None or instance of ipywidgets.Output
            used for capturing messages from the visualizers.
            If None, a new output widget is created and displayed
            as a child. If not None, it is not displayed but only
            used for capturing output and the owner of the output
            is responsible for displaying it.
        kwargs: options for plotting, passed on to matplotlib commands.
        """
        widgets.VBox.__init__(self)

        if output_widget is not None:
            # for a given output widget we don't add
            # it to the children but rely on the owner
            # of the widget to display it somewhere
            assert isinstance(output_widget, widgets.Output)
            add_out_to_children = False
        else:
            output_widget = widgets.Output(
                layout={'border': '1px solid black'})
            add_out_to_children = True
        self.output_widget = output_widget

        # if no figure is given, we create one and add it to children
        # if one is given, we don't add it as part of children
        add_fig_to_children = fig is None
        # setting of figure and ax
        self.fig = None
        self.ax = None
        self._init_fig_and_ax(fig, **kwargs)

        # create the PIConGPU visualizer instance but do not set
        # any run directories yet
        self.plot_mpl = plot_mpl_cls(None, ax=self.ax)

        self.label_path_lut = None
        self._create_label_path_lookup(run_dir_options)

        # widgets for selecting the simulation and the simulation step
        # dependent on the derived class which widget it should be
        # use the simulation labels of the plot_mpl visualizer from picongpu
        self.sim_drop = self._create_sim_dropdown(
            sorted(list(self.label_path_lut.keys())))
        self.sim_drop.observe(
            self._handle_run_dir_selection_callback, names='value')

        self.sim_time_slider = widgets.SelectionSlider(
            description='Time [s]',
            options=[""],
            continuous_update=False,
            # layout=widgets.Layout(width='65%', height='10%'))
        )
        self.sim_time_slider.observe(self._visualize_callback, names='value')

        # widgets that this specific widget might need
        # to expose the parameters of the plot_mpl object.
        self.widgets_for_vis_args = \
            self._create_widgets_for_vis_args()
        # Its changes will result in changes to the plot
        for _, widg in self.widgets_for_vis_args.items():
            widg.observe(self._visualize_callback, names='value')

        # register the ui elements that will be displayed
        # as children of this object.
        vis_widgets = widgets.VBox(
            children=[
                self.sim_drop,
                widgets.VBox(
                    children=list(self.widgets_for_vis_args.values()))])
        if add_fig_to_children:
            top = widgets.HBox(children=[
                vis_widgets, self.fig.canvas])
        else:
            top = vis_widgets

        if add_out_to_children:
            bottom = widgets.VBox(children=[
                self.sim_time_slider, self.output_widget])
        else:
            bottom = self.sim_time_slider

        self.children = [top, bottom]

    @capture_output
    def _create_label_path_lookup(self, run_dir_options):
        """
        Creates the lookup table from simulation labels to their paths.

        Parameters
        ----------
        run_dir_options: str or list of str or list of tuples of (str, str)
                         with a label and the path to the simulation.
        """
        # conversion from single element to list
        if not isinstance(run_dir_options, list):
            run_dir_options = [run_dir_options]

        if len(run_dir_options) < 1:
            warn("Empty run_dir_options list was passed!")
            lut = {}
        else:
            if isinstance(run_dir_options[0], str):
                # enumeration as labels
                lut = {str(i): path for i, path in enumerate(run_dir_options)}
            else:
                # assume run_dir_options is a list of tuples of length two
                lut = {label: path for label, path in run_dir_options}

        # lookup table from label strings to paths
        self.label_path_lut = lut

    def _show_run_dir_options_in_dropdown(self):
        """
        Make the labels of the run_dir_options lookup table available for
        selection as options for the dropdown.
        """
        sim_options = sorted(list(self.label_path_lut.keys()))
        self.sim_drop.unobserve(
            self._handle_run_dir_selection_callback, names='value')
        # set the UI
        self.sim_drop.options = sim_options
        # don't select a value yet but leave it to the user.
        # this needs to be handled differently
        # for single and multi selection
        if isinstance(self.sim_drop, widgets.Dropdown):
            self.sim_drop.value = None
        else:
            # we assume widgets.SelectMultiple instance here
            self.sim_drop.value = ()
        # re-enable the callback functions
        self.sim_drop.observe(
            self._handle_run_dir_selection_callback, names='value')

    @capture_output
    def set_run_dir_options(self, run_dir_options):
        """
        Re-set the selectable simulations and everything that is affected
        by it in the dropdown menu.

        Parameters
        ----------
        run_dir_options: list
            list of tuples of length two with
            a label and a path to a PIConGPU simulation
        """
        # renew the lookup table
        self._create_label_path_lookup(run_dir_options)
        # set the options in the dropdown
        self._show_run_dir_options_in_dropdown()

        # clear the ax (this is done by the current plot_mpl instance)
        self._clean_ax()

        # create a fresh plot_mpl object since the old
        # one had some run directories which are outdated now
        plot_mpl_class = type(self.plot_mpl)
        self.plot_mpl = plot_mpl_class(
            run_directories=None,
            ax=self.ax)

        # the user has not yet chosen any simulation
        # so we have no option about which times are available
        self.sim_time_slider.options = ('',)

    def _init_fig_and_ax(self, fig, **kwargs):
        """
        Creates the figure and the ax as members.
        """
        if fig is None:
            # create a new figure and add them to the plot
            self.fig, self.ax = plt.subplots(1, 1, **kwargs)
        else:
            # Take the figure and do NOT add it to the children list but
            # instead just draw straight into it.
            # This way other elements can compose figure
            # and this widget in any way they like.

            # still clear the figure first to get rid of stuff that
            # still might be there
            # from some other plotting
            fig.clear()
            self.fig = fig
            # provide unique label so new axes are always created
            lab = (str(type(self)) +
                   "_" + datetime.now().strftime("%Y%m%d%H%M%S"))
            self.ax = fig.add_subplot(111, label=lab)

# interface functions that can be overridden
    def _create_sim_dropdown(self, options):
        """
        Provide the widget for selection of simulations.
        Can be overridden in derived classes if some of those widgets
        are not necessary.
        Note: Make sure that no value of the widget is selected initially
        since otherwise initial plotting after creation of the widget might
        not work (since the constructor sets the value to the first available
        which leads to the visualization callback being triggered.)

        Returns
        -------
        a jupyter widget that allows selection of value(s)
        """
        sim_drop = widgets.SelectMultiple(
            description="Sims", options=options, value=())

        return sim_drop

    def _create_widgets_for_vis_args(self):
        """
        Provide additional UI widget elements that expose the parameters
        of the underlying plot_mpl visualizer instance that the user should
        be able to modify.

        Returns
        -------
        a dict mapping keyword argument names of the PIC visualizer
        to the widgets that are used for adjusting those values.
        Those widgets should be created in this function as well.

        Note: no callback for plotting needs to be registered, this is done
        automatically during construction.
        """
        return {}

    def _get_widget_args(self):
        """
        Returns
        -------
        a dict mapping keyword argument names of the PIC visualizer
        to the values of the corresponding widget elements.
        """
        return {arg: widg.value
                for arg, widg in self.widgets_for_vis_args.items()}

# functions that should NOT be overridden
    @capture_output
    def _handle_run_dir_selection(self, run_dir_selection):
        """
        Update the run_directories and the common simulation times.
        """
        # 1. need to set the run directories
        self._update_plot_mpl_run_dir(run_dir_selection)

        # 2. need to compute the simulation time common to all selected
        #    simulations and set the slider value to one of those values
        #    (without triggering the callback)
        self._update_available_sim_times()

    def _handle_run_dir_selection_callback(self, change):
        """
        Callback function when user selects a subset of the
        available simulations.
        """
        selected_sims_labels = change['new']

        self._handle_run_dir_selection(selected_sims_labels)

        # 3. visualize
        self.visualize()

    def _update_plot_mpl_run_dir(self, selected_sims):
        """
        Passes the selected simulations to the
        visualizer instance.

        Parameters
        ----------
        selected_sims: list
            list of simulation labels which will be translated to their path.
        """
        if isinstance(selected_sims, tuple):
            # handle the case for a single selected value from
            # a widgets.Dropdown instance
            selected_sims = list(selected_sims)

        if not isinstance(selected_sims, list):
            selected_sims = [selected_sims]

        # print("in _update_plot_mpl_run_dir, selected_sims =", selected_sims)

        run_dirs = [self.label_path_lut[label] for label in selected_sims]
        # prepare nicer labels for run_dirs in the plots
        run_dirs = list(zip(selected_sims, run_dirs))

        self.plot_mpl.set_run_directories(run_dirs)

    def _update_available_sim_times(self):
        """
        Computes the intersection of simulation times that are present
        in all simulations currently selected.
        It automatically plots the iteration step that best matches the
        specified simulation time.
        """
        # get the available iterations for every simulation
        # and build the intersection
        all_sim_times = []  # list of sets
        for reader in self.plot_mpl.data_reader:
            # query the reader for the available sim_times
            try:
                vis_params = self._get_widget_args()
                times_avail = reader.get_times(
                    **vis_params)
            except IOError as e:
                print("Getting times failed!", e)
                raise e
            # print(reader.run_directory, ":", times_avail)
            all_sim_times.append(set(times_avail))

        # the simulation times shared by all selected simulations
        common_sim_times = sorted(list(set.intersection(*all_sim_times)))
        if len(common_sim_times) == 0:
            common_sim_times = [""]
        # print("common_sim_times = ", common_sim_times)
        # Note: this would trigger the callback function of the
        # iteration slider and we deactivate it here first
        self.sim_time_slider.unobserve(self._visualize_callback,
                                       names='value')
        # changing the option might silently also
        # adjust the .value attribute!
        self.sim_time_slider.options = common_sim_times
        last_selected_val = self.sim_time_slider.value
        # if the previously used iteration is available, use it as value
        # otherwise start from first available
        if last_selected_val in self.sim_time_slider.options:
            self.sim_time_slider.value = last_selected_val
        else:
            print("last selected time {0} not present with newly"
                  " selected sim_times, set to {1}".format(
                        last_selected_val, common_sim_times[0]))
            self.sim_time_slider.value = common_sim_times[0]

        # re-enable the callback
        self.sim_time_slider.observe(self._visualize_callback,
                                     names='value')

    def _visualize_callback(self, change):
        """
        Callback that is executed when one of the extra_ui_elements
        changes or the iteration changes."""
        self.visualize()

    @capture_output
    def visualize(self, **kwargs):
        """
        Draw the plot by getting all necessary parameter values from the
        exposed widgets.

        Parameters
        ----------
        kwargs: dict
            Ignored at the moment.
        """
        time = self.sim_time_slider.value

        # the case where no valid iteration is provided
        if time is None or time == "":
            return

        vis_params = self._get_widget_args()
        try:
            self.plot_mpl.visualize(time=time,
                                    **vis_params)
        except Exception as e:
            warn("{}: visualization failed! Reason: {} ".format(type(self), e))
            # raise e

        # since interactive mode should be turned off, we have
        # to update the figure explicitely
        try:
            self.update_plot()
        except ValueError as e:
            warn("{}: drawing the plot failed! Reason: {}".format(
                type(self), e))
            # raise e

    def update_plot(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _make_drop_val_compatible(self, val):
        """
        Depending on the type of self.sim_drop we have to
        assign a single value or a tuple to the self.sim_drop.value.
        This function converts the 'val' in a way to be compatible
        with the expected type.
        """
        # handle conversion from SelectMultiple which expects
        # tuples to ordinary dropdown which accepts a single value
        if isinstance(self.sim_drop, widgets.Dropdown):
            # handle SelectMultiple -> Dropdown
            if isinstance(val, tuple):
                val = val[0]
        elif isinstance(self.sim_drop, widgets.SelectMultiple):
            # handle Dropdown -> SelectMultiple
            if not isinstance(val, tuple):
                val = (val,)

        return val

    @capture_output
    def _use_options_from_other(self, other):
        """
        Uses the options from another instance derived from this class
        to instantly set e.g. the simulation time or the selected simulation.

        This should only be used when switching the visualizer but keeping
        the run_dir_options as they are.
        """
        other_sim = other.sim_drop.value
        # skip this if other_sim is None (when coming from Dropdown)
        # or empty (when coming from a SelectMultiple)
        if other_sim is None or len(other_sim) == 0:
            return

        # setting the previously selected simulations
        # (without executing callback)
        self.sim_drop.unobserve(
            self._handle_run_dir_selection_callback, names='value')
        self.sim_drop.value = self._make_drop_val_compatible(other_sim)
        self.sim_drop.observe(
            self._handle_run_dir_selection_callback, names='value')

        # manually update the run directories and the simulation times.
        # Needed since we deactivated the callback to prevent plotting
        self._handle_run_dir_selection(self.sim_drop.value)

        other_time = other.sim_time_slider.value
        # check if other_time is available before setting it
        # this also triggers visualization
        self.sim_time_slider.unobserve(
            self._visualize_callback, names='value')

        if other_time in self.sim_time_slider.options:
            self.sim_time_slider.value = other_time
        else:
            # we dont have the selected time of the other widget
            # so we choose the first available
            self.sim_time_slider.value = self.sim_time_slider.options[0]

        self.sim_time_slider.observe(
            self._visualize_callback, names='value')

        # species, filter and other stuff (extra_ui_elements)
        for arg, other_widget in other.widgets_for_vis_args.items():
            # if the keyword matches, we assign the value
            # of the other widget to our widget
            if arg in self.widgets_for_vis_args:
                other_value = other_widget.value
                own_widget = self.widgets_for_vis_args[arg]
                # prevent triggering the callback each time
                # but defer it until all args are set
                own_widget.unobserve(self._visualize_callback, names='value')
                own_widget.value = other_value
                own_widget.observe(self._visualize_callback, names='value')

    @capture_output
    def _clean_ax(self):
        self.plot_mpl._clean_ax()
        # refresh the figure since we are not in interactive mode
        self.update_plot()
