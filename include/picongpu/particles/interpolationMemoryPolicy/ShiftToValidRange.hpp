/* Copyright 2016-2021 Richard Pausch
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


namespace picongpu
{
    namespace particles
    {
        namespace interpolationMemoryPolicy
        {
            /** Shift position to valid range [0,1)
             *  and repositions memory accordingly.
             *  This is necessary if a particle moves
             *  outside of its cell during a sub-stepping cycle
             *  Returns: shifted position and shifted memory. */
            struct ShiftToValidRange
            {
                template<typename T_MemoryType, typename T_PosType>
                HDINLINE T_MemoryType memory(const T_MemoryType& mem, const T_PosType& pos) const
                {
                    const T_PosType pos_floor = math::floor(pos);
                    return mem(precisionCast<int>(pos_floor));
                }

                template<typename T_PosType>
                HDINLINE T_PosType position(const T_PosType& pos) const
                {
                    const T_PosType pos_floor = math::floor(pos);
                    return pos - pos_floor;
                }
            };

        } // namespace interpolationMemoryPolicy

    } // namespace particles
} // namespace picongpu
