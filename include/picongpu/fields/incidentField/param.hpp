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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

< < < < < < < < HEAD : include / picongpu / fields / incidentField / param.hpp
#include "picongpu/defines.hpp"
#include "picongpu/param/incidentField.param"
    == == == ==
#include <cstdint>


    namespace picongpu{
        namespace simulation{namespace stage{//! Initialize particles
                                             struct ParticleInit{/** Initialize particles dependent of the given step
                                                                  *
                                                                  * @param step index of time iteration
                                                                  */
                                                                 void
                                                                 operator()(uint32_t const step) const;
}
;
} // namespace stage
} // namespace simulation
} // namespace picongpu
>>>>>>>> 8f6920fcc(compile unit : stage refactoring) : include / picongpu / simulation / stage / ParticleInit.hpp
