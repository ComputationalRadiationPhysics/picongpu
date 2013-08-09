#!/usr/bin/env python2.7
#
# Copyright 2013 Richard Pausch
#
# This file is part of PIConGPU.
#
# PIConGPU is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PIConGPU is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PIConGPU.
# If not, see <http://www.gnu.org/licenses/>.
#

from numpy import *
    



for angle_id_extern in arange(481):
    N_phi_split = 32
    N_theta = 16
    
    my_index_theta = angle_id_extern / N_phi_split
    my_index_phi = angle_id_extern % N_phi_split
    
    phi_range   = pi
    theta_range = pi/2.0

    delta_phi   = phi_range   / (N_phi_split - 1)
    delta_theta = theta_range / (N_theta - 1)

    theta =  my_index_theta * delta_theta  + 0.5*pi 
    phi   =  my_index_phi   * delta_phi  


    x = sin(theta)*cos(phi)
    y = sin(theta)*sin(phi)
    z = cos(theta)
    print around([x, y, z], 3)
    
