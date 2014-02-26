/**
 * Copyright 2013 Heiko Burau, Richard Pausch
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

namespace picongpu
{
  namespace radiation_observer
  {
    DINLINE vec2 observation_direction(const int theta_id_extern)
    {
       //returns vec2 look;

	  const int my_theta_id = theta_id_extern;
	  const numtype2 delta_theta =  2.0 * 1.5/5.0 / (parameters::N_theta);
	  const numtype2 theta(my_theta_id * delta_theta + picongpu::PI - 1.5/5.0); // off axis angle
	  return vec2(sinf(theta), cosf(theta), 0.0);
      


      
    }
    
  } // end namespace radiation_observer
} // end namespace picongpu
