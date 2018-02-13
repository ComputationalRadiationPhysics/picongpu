import matplotlib.pyplot as plt


class Visualizer(object):
    def __init__(self, run_directory):

        if run_directory is None:
            raise ValueError('The run_directory parameter can not be None!')

        self.plt_obj = None
        self.data_reader = self._create_data_reader(run_directory)
        self.data = None

    def _create_data_reader(self, run_directory):
        raise NotImplementedError

    def _create_plt_obj(self, ax):
        """
        Creates a matplotlib figure of some kind which can be updated
        by feeding new values into it. Only called on the first call for
        visualization.
        """
        raise NotImplementedError

    def _update_plt_obj(self):
        """
        Take the 'data' member which was filled before,
        interpret it and feed it into an existing 'plt_obj'
        """
        raise NotImplementedError

    def visualize(self, ax=plt.gca(), **kwargs):
        """
        1. Creates the 'plt_obj' if it does not exist
        2. Fills the 'data' parameter by using the reader
        3. Updates the 'plt_obj' with the new data.
        """

        if self.plt_obj is None:
            self._create_plt_obj(ax)

        # TODO: this requires all the readers to consume **kwargs as well
        self.data = self.data_reader.get(**kwargs)
        self._update_plt_obj()
