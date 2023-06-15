.. _testing-new:

How to setup a new test
=======================

General
-------

.. note::

   Of course, all functionalities can be used to design your own test without resorting to the structure with Data.py.
   This is assumed to be known and is therefore not discussed further here.
   Please refer to the documentation for the individual functionalities.
    
A template for developing a new test is available under ``.lib/python/test/testsuite/Template``.
The files it contains can now be adapted to the new test in an extra folder. 
The main.py file is only used to run the test and does not need to be changed any further.
Much more important is the adaptation of Data.py. This is explained in more detail in the next section.
If the test case is to be checked automatically by the ci afterwards, ci.sh and, if possible, a validate.sh must also be written after adapting Data.py.
To do this, adapt the existing ci.sh or validate.sh files for the sake of simplicity.

Using Data.py
-------------

The data.py file provides the test suite with all the essential data for running the test.
The most important are the functions for calculating the data from theory and from the simulation.
To better illustrate Data.py, the usage will be explained in more detail here using the MI example.

1. Data not relevant to the test
""""""""""""""""""""""""""""""""

.. code-block:: python

    title = "KHI Growthrate (2D MI)"
    author = "Mika Soren Voss"
    
These parameters are all optional.
They serve to provide a better overview and a better presentation of the results after the test.
As already mentioned, there are two variants of documenting the test results, the text format as a .log file and a plot.
The author is only mentioned in the .log file, while the title is used as a heading for the test.
If the plot_title parameter is not set or is defined as None, the title is used as the plot title.

2. paths to files
"""""""""""""""""

.. code-block:: python

    # directory specified
    resultDirection = None
    paramDirection = None
    dataDirection = None
    
In order to be able to evaluate the test, the test suite needs to know what data it can use and where it gets it from.
To ensure this, the paths to the individual folders can be specified here.
If data from these files is required, these parameters must be set unless they are passed to main.py. 
(See the :ref:`main.py documentation <usage>` for this).

``resultDirection`` designates the folder in which the result of the test is to be saved.
If an automatic test is to be carried out and you are interested in the results in detail, this value should be set.

``dataDirection`` gives the path to the .txt files of the simulation.
``paramDirection`` and ``jsonDirection`` are available to include .param files and .json files in the evaluation.

The test suite then needs to know which parameters it has to read out. 
The parameters ``data_Parameter``, ``param_Parameter`` and ``json_Parameter`` are available for this.
In the MI example, no ``json_parameter`` is required:

.. code-block:: python

    # parameter information found in .param files
    param_Parameter = ["gamma", "BASE_DENSITY_SI", "DELTA_T_SI"]
    data_Parameter = ["Bz", "step"]
    
3. test condition
"""""""""""""""""

In order for the test suite to be able to decide whether a test passed or not, it needs an acceptance range.
This is done in the form of the parameter ``acceptance``, which contains the maximum deviation.
An acceptance range of 0.2 thus describes a maximum deviation of 20% from an expected value in which the test is passed.

.. code-block:: python

    # acceptance in percentage / 100
    acceptance = 0.2
    
4. Plot data
""""""""""""

.. code-block:: python

    # plot data
    plot_xlabel = r"$t[\omega_{pe}^ {-1}]$"
    plot_ylabel = r"$\Gamma_\mathrm{Fi}$"
    # if None or not defined the standard type will be used, see documentation
    plot_type = None
    # if None or not defined the time will be used
    plot_xaxis = None
    # for more values see the documentation (e.g. 2D plot needs zaxis and yaxis)
    


In this section, all data required for the plot display are transmitted.
Possible parameters are ``plot_title``, ``plot_xlabel``, ``plot_ylabel``, ``plot_type`` and ``plot_xaxis``.
The first three define the annotation of the plot.

``plot_type`` can assume the values 1D or 2D and describes the dimensionality of the plot.
If no value is specified, 1D is used by default.
``plot_xaxis`` describes the values of the x-axis(default: time).

5. Functions
""""""""""""

Finally, there are two more functions.
The first calculates the value of the theory and must contain this as a return value. 
The second calculates the values from the simulation.
Since both functions have a similar structure, we only consider the theoretical function here:

.. code-block:: python

    def theory(gamma, *args):
        """
        this function indicates how the theoretical values
        can be calculated from the data. It must be filled out
        and have the theory as return value.

        All parameters that are read from the test-suite must
        be given the same names as in the parameter lists.

        Return:
        -------
        out : theoretical values!
        """
        # gamma is calculated automatically, does not have to be passed
        v = ts.Math.physics.calculateV_O()

        return (v / (c * np.sqrt(gamma)))
        
