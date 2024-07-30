/* Copyright 2024 Rene Widera
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

// required to get the definition of SIMDIM
#include "picongpu/simulation_defines.hpp"

#if(SIMDIM == DIM3 && PIC_ENABLE_FFTW3 == 1 && ENABLE_OPENPMD == 1)

#    include "picongpu/plugins/PluginRegistry.hpp"
#    include "picongpu/plugins/multi/multi.hpp"
#    include "picongpu/plugins/shadowgraphy/Shadowgraphy.hpp"


PIC_REGISTER_PLUGIN(picongpu::plugins::multi::Master<picongpu::plugins::shadowgraphy::Shadowgraphy>);
#endif
