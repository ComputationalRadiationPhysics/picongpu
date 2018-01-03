/* Copyright 2015-2018 Axel Huebl, Benjamin Worpitz
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

#include "picongpu/simulation_defines.hpp"

#include "picongpu/fields/currentInterpolation/None/None.def"
#include "picongpu/algorithms/DifferenceToUpper.hpp"
#include "picongpu/algorithms/LinearInterpolateWithUpper.hpp"
#include <pmacc/traits/GetComponentsType.hpp>

#include "picongpu/fields/MaxwellSolver/Yee/Curl.hpp"


namespace picongpu
{
namespace currentInterpolation
{
using namespace pmacc;

namespace detail
{
    template<uint32_t T_simDim, uint32_t T_plane>
    struct LinearInterpolateComponentPlaneUpper
    {
        static constexpr uint32_t dim = T_simDim;

        /* UpperMargin is actually 0 in direction of T_plane */
        typedef typename pmacc::math::CT::make_Int<dim, 0>::type LowerMargin;
        typedef typename pmacc::math::CT::make_Int<dim, 1>::type UpperMargin;

        template<typename DataBox>
        HDINLINE float_X operator()(DataBox field) const
        {
            const DataSpace<dim> self;
            DataSpace<dim> up;
            up[(T_plane + 1) % dim] = 1;

            using Avg = LinearInterpolateWithUpper< dim >;

            const typename Avg::template GetInterpolatedValue< (T_plane + 2) % dim > avg;

            return float_X(0.5) * ( avg(field)[T_plane] + avg(field.shift(up))[T_plane] );
        }
    };

    /* shift a databox along a specific direction
     *
     * returns the identity (assume periodic symmetry) if direction is not
     * available, such as in a 2D simulation
     *
     * \todo accept a full CT::Vector and shift if possible
     * \todo call with CT::Vector of correct dimensionality that was created
     *       with AssignIfInRange...
     *
     * \tparam T_simDim maximum dimensionality of the mesh
     * \tparam T_direction (0)X (1)Y or (2)Z for the direction one wants to
     *                     shift to
     * \tparam isShiftAble auto-filled value that decides if this direction
     *                     is actually non-existent == periodic
     */
    template<uint32_t T_simDim, uint32_t T_direction, bool isShiftAble=(T_direction<T_simDim) >
    struct ShiftMeIfYouCan
    {
        static constexpr uint32_t dim = T_simDim;
        static constexpr uint32_t dir = T_direction;

        HDINLINE ShiftMeIfYouCan()
        {
        }

        template<class T_DataBox >
        HDINLINE T_DataBox operator()(const T_DataBox& dataBox) const
        {
            DataSpace<dim> shift;
            shift[dir] = 1;
            return dataBox.shift(shift);
        }
    };

    template<uint32_t T_simDim, uint32_t T_direction>
    struct ShiftMeIfYouCan<T_simDim, T_direction, false>
    {
        HDINLINE ShiftMeIfYouCan()
        {
        }

        template<class T_DataBox >
        HDINLINE T_DataBox operator()(const T_DataBox& dataBox) const
        {
            return dataBox;
        }
    };

    /* that is not a "real" yee curl, but it looks a bit like it */
    template<class Difference>
    struct ShiftCurl
    {
        using LowerMargin = typename Difference::OffsetOrigin;
        using UpperMargin = typename Difference::OffsetEnd;

        template<class DataBox >
        HDINLINE typename DataBox::ValueType operator()(const DataBox& mem) const
        {
            const typename Difference::template GetDifference<0> Dx;
            const typename Difference::template GetDifference<1> Dy;
            const typename Difference::template GetDifference<2> Dz;

            const ShiftMeIfYouCan<simDim, 0> sx;
            const ShiftMeIfYouCan<simDim, 1> sy;
            const ShiftMeIfYouCan<simDim, 2> sz;

            return float3_X(Dy(sx(mem)).z() - Dz(sx(mem)).y(),
                            Dz(sy(mem)).x() - Dx(sy(mem)).z(),
                            Dx(sz(mem)).y() - Dy(sz(mem)).x());
        }
    };
} /* namespace detail */

template<uint32_t T_simDim>
struct NoneDS
{
    static constexpr uint32_t dim = T_simDim;

    typedef typename pmacc::math::CT::make_Int<dim, 0>::type LowerMargin;
    typedef typename pmacc::math::CT::make_Int<dim, 1>::type UpperMargin;

    template<typename DataBoxE, typename DataBoxB, typename DataBoxJ>
    HDINLINE void operator()(DataBoxE fieldE,
                             DataBoxB fieldB,
                             DataBoxJ fieldJ )
    {
        typedef typename DataBoxJ::ValueType TypeJ;
        typedef typename GetComponentsType<TypeJ>::type ComponentJ;

        const DataSpace<dim> self;

        const ComponentJ deltaT = DELTA_T;
        const ComponentJ constE = (float_X(1.0)  / EPS0) * deltaT;
        const ComponentJ constB = (float_X(0.25) / EPS0) * deltaT * deltaT;

        const detail::LinearInterpolateComponentPlaneUpper<dim, 0> avgX;
        const ComponentJ jXavg = avgX(fieldJ);
        const detail::LinearInterpolateComponentPlaneUpper<dim, 1> avgY;
        const ComponentJ jYavg = avgY(fieldJ);
        const detail::LinearInterpolateComponentPlaneUpper<dim, 2> avgZ;
        const ComponentJ jZavg = avgZ(fieldJ);

        const TypeJ jAvgE = TypeJ(jXavg, jYavg, jZavg);
        fieldE(self) -= jAvgE * constE;

        using CurlRight = yeeSolver::Curl< DifferenceToUpper< dim > >;
        using ShiftCurlRight = detail::ShiftCurl< DifferenceToUpper< dim > >;
        CurlRight curl;
        ShiftCurlRight shiftCurl;

        const TypeJ jAvgB = curl(fieldJ) + shiftCurl(fieldJ);
        fieldB(self) += jAvgB * constB;
    }

    static pmacc::traits::StringProperty getStringProperties()
    {
        pmacc::traits::StringProperty propList( "name", "none" );
        return propList;
    }
};

} /* namespace currentInterpolation */

} /* namespace picongpu */
