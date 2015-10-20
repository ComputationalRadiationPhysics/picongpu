/**
 * Copyright 2015 Richard Pausch
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



/** functor for particle field interpolator
 *
 * This functor is a simplification of the full
 * field to particle interpolator that can be used in the
 * particle pusher
 */

namespace picongpu
{

using namespace PMacc;


template< typename T_Field2PartInt, typename T_MemoryType, typename T_FieldPosition >
struct InterpolationForPusher
{
    typedef T_Field2PartInt Field2PartInt;

    HDINLINE
    InterpolationForPusher( const T_MemoryType& mem, const T_FieldPosition& fieldPos )
        : m_mem( mem ), m_fieldPos( fieldPos )
    {
    }

    template< typename T_PosType >
    HDINLINE
    float3_X operator()( const T_PosType& pos ) const
    {
        /* to do: memory shift, position shift */
        return Field2PartInt()( m_mem, pos, m_fieldPos );
    }

private:
    PMACC_ALIGN( m_mem, T_MemoryType );
    const PMACC_ALIGN( m_fieldPos, T_FieldPosition );
};


/** functor to create particle field interpolator
 *
 * required to get interpolator for pusher
 */
template<typename T_Field2PartInt>
struct CreateInterpolationForPusher
{
    template< typename T_MemoryType, typename T_FieldPosition >
    HDINLINE
    InterpolationForPusher< T_Field2PartInt, T_MemoryType, T_FieldPosition >
    operator()( const T_MemoryType& mem, const T_FieldPosition& fieldPos )
    {
        return InterpolationForPusher< T_Field2PartInt, T_MemoryType, T_FieldPosition >( mem, fieldPos );
    }
};

} // namespace picongpu
