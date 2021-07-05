/* Copyright 2020-2021 Pawel Ordyna
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

#include "picongpu/plugins/externalBeam/AxisSwap.hpp"

#include <boost/mpl/integral_c.hpp>

namespace picongpu
{
    namespace plugins
    {
        namespace externalBeam
        {
            //! Probing along the PIC x basis vector.
            struct XSide
            {
                static constexpr float_X beamStartPosition[3] = {0.0, 0.5, 0.5};
                static constexpr unsigned axisMap[3] = {2, 1, 0};
                static constexpr bool reverse[3] = {true, false, false};
            };
            constexpr float_X XSide::beamStartPosition[3];
            constexpr bool XSide::reverse[3];
            constexpr unsigned XSide::axisMap[3];

            //! Probing against the PIC x basis vector.
            struct XRSide
            {
                static constexpr float_X beamStartPosition[3] = {1.0, 0.5, 0.5};
                static constexpr unsigned axisMap[3] = {2, 1, 0};
                static constexpr bool reverse[3] = {true, true, true};
            };
            constexpr float_X XRSide::beamStartPosition[3];
            constexpr bool XRSide::reverse[3];
            constexpr unsigned XRSide::axisMap[3];

            //! Probing along the PIC y basis vector.
            struct YSide
            {
                static constexpr float_X beamStartPosition[3] = {0.5, 0.0, 0.5};
                static constexpr unsigned axisMap[3] = {2, 0, 1};
                static constexpr bool reverse[3] = {true, true, false};
            };
            constexpr float_X YSide::beamStartPosition[3];
            constexpr bool YSide::reverse[3];
            constexpr unsigned YSide::axisMap[3];


            //! Probing against the PIC y basis vector.
            struct YRSide
            {
                static constexpr float_X beamStartPosition[3] = {0.5, 1.0, 0.5};
                static constexpr unsigned axisMap[3] = {2, 0, 1};
                static constexpr bool reverse[3] = {true, false, true};
            };
            constexpr float_X YRSide::beamStartPosition[3];
            constexpr bool YRSide::reverse[3];
            constexpr unsigned YRSide::axisMap[3];

            //! Probing along the PIC z basis vector.
            struct ZSide
            {
                static constexpr float_X beamStartPosition[3] = {0.5, 0.5, 0.0};
                static constexpr unsigned axisMap[3] = {1, 0, 2};
                static constexpr bool reverse[3] = {true, false, false};
            };
            constexpr float_X ZSide::beamStartPosition[3];
            constexpr bool ZSide::reverse[3];
            constexpr unsigned ZSide::axisMap[3];

            //! Probing against the PIC z basis vector.
            struct ZRSide
            {
                static constexpr float_X beamStartPosition[3] = {0.5, 0.5, 0.0};
                static constexpr unsigned axisMap[3] = {1, 0, 2};
                static constexpr bool reverse[3] = {true, true, true};
            };
            constexpr float_X ZRSide::beamStartPosition[3];
            constexpr bool ZRSide::reverse[3];
            constexpr unsigned ZRSide::axisMap[3];

            namespace detail
            {
                namespace mpl = boost::mpl;

                template<bool x, bool y, bool z>
                struct GetBeamIdx;
                template<>
                struct GetBeamIdx<true, false, false>
                {
                    using type = mpl::integral_c<uint32_t, 0>;
                };
                template<>
                struct GetBeamIdx<false, true, false>
                {
                    using type = mpl::integral_c<uint32_t, 1>;
                };
                template<>
                struct GetBeamIdx<false, false, true>
                {
                    using type = mpl::integral_c<uint32_t, 2>;
                };
            } // namespace detail

            /** Create an an AxisSwap instance for the given beam placement
             *
             * @tparam T_Side defines from which simulation side a beam enters the simulation
             *  possible options are XSide, XRSide, YSide, YRSide, ZSide, ZRSide
             */
            template<typename T_Side>
            struct ProbingSideCfg
            {
                using Side = T_Side;

                struct AxisSwapRT : public AxisSwap
                {
                    HINLINE AxisSwapRT()
                        : AxisSwap{
                            Side::axisMap[0],
                            Side::axisMap[1],
                            Side::axisMap[2],
                            Side::reverse[0],
                            Side::reverse[1],
                            Side::reverse[2]}
                    { }
                };

                HINLINE AxisSwap getAxisSwap()
                {
                    return AxisSwap{
                        Side::axisMap[0],
                        Side::axisMap[1],
                        Side::axisMap[2],
                        Side::reverse[0],
                        Side::reverse[1],
                        Side::reverse[2]};
                }


                struct AxisSwapCT
                {
                    template<typename T_CtVec>
                    struct Swap
                    {
                    private:
                        using IntegralType = typename T_CtVec::type;

                        template<IntegralType x, IntegralType y, IntegralType z>
                        using ReturnVector = pmacc::math::CT::Vector<
                            mpl::integral_c<IntegralType, x>,
                            mpl::integral_c<IntegralType, y>,
                            mpl::integral_c<IntegralType, z>>;

                        static constexpr IntegralType xNew{T_CtVec::template at<Side::axisMap[0]>::type::value};
                        static constexpr IntegralType yNew{T_CtVec::template at<Side::axisMap[1]>::type::value};
                        static constexpr IntegralType zNew{T_CtVec::template at<Side::axisMap[2]>::type::value};

                    public:
                        using type = ReturnVector<xNew, yNew, zNew>;
                    };

                    template<typename T_CtVec>
                    struct ReverseSwap
                    {
                    private:
                        using IntegralType = typename T_CtVec::type;
                        using ReturnVec0 = pmacc::math::CT::Vector<>;

                        template<unsigned idx>
                        using GetAssignIdx = mpl::integral_c<unsigned, Side::axisMap[idx]>;

                        using ReturnVec1 =
                            typename pmacc::math::CT::Assign<ReturnVec0, GetAssignIdx<0>, typename T_CtVec::x>::type;
                        using ReturnVec2 =
                            typename pmacc::math::CT::Assign<ReturnVec1, GetAssignIdx<1>, typename T_CtVec::y>::type;
                        using ReturnVec3 =
                            typename pmacc::math::CT::Assign<ReturnVec2, GetAssignIdx<2>, typename T_CtVec::z>::type;

                    public:
                        using type = ReturnVec3;
                    };

                    template<uint32_t idxBeam>
                    struct BeamToPicIdx
                    {
                        using type = mpl::integral_c<uint32_t, Side::axisMap[idxBeam]>;
                    };

                    template<uint32_t idxPic>
                    struct PicToBeamIdx
                    {
                        using type = typename detail::GetBeamIdx<
                            idxPic == Side::axisMap[0],
                            idxPic == Side::axisMap[1],
                            idxPic == Side::axisMap[2]>::type;
                    };
                };
            };
        } // namespace externalBeam
    } // namespace plugins
} // namespace picongpu
