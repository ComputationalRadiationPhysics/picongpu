import os
import sys
sys.path.insert(0, os.path.abspath('../../../../lib/python'))

from picongpu.plugins.plot_mpl.xray_diffraction_visualizer import Visualizer

visualizer = Visualizer("e_intensity10.dat")
visualizer.plot_xy(markerSize=6)

visualizer2 = Visualizer("e_intensity10_4.dat")
visualizer.plot_xy(markerSize=6)
