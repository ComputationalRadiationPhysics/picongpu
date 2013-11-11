/**
 * Copyright 2013 Heiko Burau, Rene Widera
 *
 * This file is part of PIConGPU. 
 * 
 * PIConGPU is free software: you can redistribute it and/or modify 
 * it under the terms of the GNU General Public License as published by 
 * the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version. 
 * 
 * PIConGPU is distributed in the hope that it will be useful, 
 * but WITHOUT ANY WARRANTY; without even the implied warranty of 
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
 * GNU General Public License for more details. 
 * 
 * You should have received a copy of the GNU General Public License 
 * along with PIConGPU.  
 * If not, see <http://www.gnu.org/licenses/>. 
 */ 
 



#pragma once

#include "types.h"
#include "simulation_defines.hpp"

namespace picongpu
{
  namespace radiation_observer
  {
    DINLINE vec2 observation_direction(const int theta_id_extern)
    {
        
              //vec2 look;
      const int N_angle_split = 16;

      const int my_index_theta = theta_id_extern / N_angle_split;
      const int my_index_phi = theta_id_extern % N_angle_split;

      const numtype2 angle_range= picongpu::PI/2.0;
      const numtype2 delta_angle =  1.0 * angle_range / (N_angle_split-1);

      const numtype2 theta(  my_index_theta * delta_angle  + 0.5*picongpu::PI ); // off axis angle
      const numtype2 phi(    my_index_phi   * delta_angle  ); // off axis angle

      return vec2( sinf(theta)*cosf(phi) , sinf(theta)*sinf(phi) , cosf(theta) ) ;

      /*
      if(theta_id_extern < (parameters::N_theta/2))
	{
	  // step with for theta (looking angle):
	  const int my_theta_id = theta_id_extern;
	  const numtype2 delta_theta =  2.0 * 1.5/5.0 / (parameters::N_theta/2);
	  const numtype2 theta(my_theta_id * delta_theta + picongpu::PI - 1.5/5.0); // off axis angle
	  look = vec2(sinf(theta), cosf(theta), 0.0);
	}
      else
	{
	  // step with for theta (looking angle):
	  const int my_theta_id = theta_id_extern % (parameters::N_theta/2);
	  const numtype2 delta_theta =  2.0 * 1.5/5.0 / (parameters::N_theta/2);
	  const numtype2 theta(my_theta_id * delta_theta + picongpu::PI - 1.5/5.0); // off axis angle
	  look = vec2(0.0, cosf(theta), sinf(theta));
	}
      
      return look;

      */
    }
    
  } // end namespace radiation_observer
} // end namespace picongpu
