/* Copyright 2025-2025 Tapish Narwal
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

#include "picongpu/plugins/binning/UnitConversion.hpp"
#include "picongpu/plugins/binning/axis/Axis.hpp"

#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace picongpu::plugins::binning::axis
{
    /**
     * Symlog (symmetric log) axis with linear behavior near zero and logarithmic behavior for large values.
     * Similar to matplotlib's symlog scale.
     * The axis has a linear region within [-linearThreshold, linearThreshold] and logarithmic regions outside.
     * Bins are distributed to maintain smooth transitions at the threshold boundaries.
     * If overflow bins are enabled, allocates 2 extra bins for under and overflow.
     */
    template<typename T_Attribute, typename T_AttrFunctor>
    class SymlogAxis
    {
    private:
        /**
         * Transform value to symlog space
         * For |val| <= threshold: returns val/threshold (normalized linear)
         * For |val| > threshold: returns sign(val) * (1 + log(|val|/threshold))
         */
        static double symlogTransform(double val, double threshold)
        {
            if(std::abs(val) <= threshold)
            {
                return val / threshold;
            }
            else
            {
                return std::copysign(1.0 + std::log(std::abs(val) / threshold), val);
            }
        }

        /**
         * Inverse transform from symlog space to real space
         */
        static double symlogInverseTransform(double y, double threshold)
        {
            if(std::abs(y) <= 1.0)
            {
                return y * threshold;
            }
            else
            {
                return std::copysign(threshold * std::exp(std::abs(y) - 1.0), y);
            }
        }

        void initBinEdges()
        {
            binEdges.reserve(axisSplit.nBins + 1);

            // Transform min and max to symlog space
            double yMin = symlogTransform(axisSplit.m_range.min, linearThresholdSI);
            double yMax = symlogTransform(axisSplit.m_range.max, linearThresholdSI);
            double yRange = yMax - yMin;

            // Create evenly spaced bins in transformed space
            for(size_t i = 0; i <= axisSplit.nBins; i++)
            {
                double y = yMin + (i * yRange) / axisSplit.nBins;
                double edge = symlogInverseTransform(y, linearThresholdSI);
                binEdges.emplace_back(edge);
            }
        }

    public:
        using Type = T_Attribute;
        using ScalingType = std::conditional_t<
            std::is_integral_v<T_Attribute>,
            std::conditional_t<sizeof(T_Attribute) <= 4, float_X, double>,
            T_Attribute>;

        AxisSplitting<T_Attribute> axisSplit;
        /** Axis name, written out to OpenPMD */
        std::string label;
        /** Units(Dimensionality) of the axis */
        std::array<double, numUnits> units;
        /** Linear threshold in SI units */
        double linearThresholdSI;
        /** axisSplit.nBins + 1 bin edges in SI units */
        std::vector<double> binEdges;

        struct SymlogAxisKernel
        {
            /** Function to place particle on axis */
            T_AttrFunctor getAttributeValue;
            /** Linear threshold in PIC units */
            ScalingType linearThresholdPIC;
            /** 1/linearThresholdPIC */
            ScalingType invLinearThreshold;
            /** Range in PIC units in symlog space */
            Range<ScalingType> symlogPicRange;
            // 1/yRange
            ScalingType invYRange;
            /** Enable or disable allocation of extra bins for out of range particles */
            bool overflowEnabled;
            /** Number of bins excluding overflow */
            uint32_t nBins;

            constexpr SymlogAxisKernel(
                T_AttrFunctor const& attrFunc,
                AxisSplitting<T_Attribute> const& axisSplit,
                double linearThreshold,
                std::array<double, numUnits> const& unitsArr)
                : getAttributeValue{attrFunc}
                , linearThresholdPIC{static_cast<ScalingType>(toPICUnits(linearThreshold, unitsArr))}
                , invLinearThreshold{ScalingType(1) / linearThresholdPIC}
                , symlogPicRange{symlogTransformKernel(toPICUnits(axisSplit.m_range.min, unitsArr)), symlogTransformKernel(toPICUnits(axisSplit.m_range.max, unitsArr))}
                , invYRange{ ScalingType(1) / (symlogPicRange.max - symlogPicRange.min)}
                , overflowEnabled{axisSplit.enableOverflowBins}
                , nBins{axisSplit.nBins}
            {
            }

            /**
             * Kernel-side symlog transform
             */
            ALPAKA_FN_HOST_ACC ScalingType symlogTransformKernel(ScalingType val) const
            {
                ScalingType absVal = math::abs(val);

                // Compute both branches
                ScalingType linear = val * invLinearThreshold;
                ScalingType logarithmic = math::copysign(ScalingType(1) + math::log(absVal * invLinearThreshold), val);

                // Check if in linear region
                ScalingType isLinear = ScalingType(absVal <= linearThresholdPIC);

                return isLinear ? linear : logarithmic;
            }

            template<typename... Args>
            ALPAKA_FN_HOST_ACC std::pair<bool, uint32_t> getBinIdx(Args const&... args) const
            {
                auto val = getAttributeValue(args...);
                auto sVal = static_cast<ScalingType>(val);
                ScalingType y = symlogTransformKernel(sVal);

                uint32_t inRange = (y >= symlogPicRange.min) & (y < symlogPicRange.max);
                // uint32_t belowMin = (y < symlogPicRange.min);
                uint32_t aboveMax = (y >= symlogPicRange.max);

                ScalingType normalized = (y - symlogPicRange.min) * invYRange;

                uint32_t rangeIdx = math::min(static_cast<uint32_t>(normalized * nBins), nBins - 1);

                if(overflowEnabled)
                {
                    // underflow*0 + inRange*(rangeIdx+1) + aboveMax*(nBins-1)
                    // I dont compute underflow/belowMin since it is always zero
                    uint32_t binIdx = inRange * (rangeIdx + 1) + aboveMax * (nBins + 1);
                    return {true, binIdx};
                }
                else
                {
                    return {inRange, rangeIdx};
                }
            }
        };

        SymlogAxisKernel sAK;

        SymlogAxis(
            AxisSplitting<T_Attribute> const& axSplit,
            T_AttrFunctor const& attrFunctor,
            double linearThreshold,
            std::string const& label,
            std::array<double, numUnits> const& units)
            : axisSplit{axSplit}
            , label{label}
            , units{units}
            , linearThresholdSI{linearThreshold}
            , sAK{attrFunctor, axisSplit, linearThreshold, units}
        {
            if(linearThreshold <= 0)
            {
                throw std::domain_error("Linear threshold must be positive");
            }
            initBinEdges();
        }

        constexpr uint32_t getNBins() const
        {
            return sAK.nBins;
        }

        double getUnitConversion() const
        {
            return getConversionFactor(units);
        }

        SymlogAxisKernel getAxisKernel() const
        {
            return sAK;
        }

        /**
         * @return bin edges in SI units
         */
        std::vector<double> getBinEdgesSI() const
        {
            return binEdges;
        }
    };

    /**
     * @details Creates a symlog (symmetric log) axis
     * @tparam T_Attribute Type of the attribute
     * @param axisSplitting Range and number of bins
     * @param linearThreshold Threshold for linear region (in SI units)
     * @param functorDescription Functor and metadata
     */
    template<typename T_Attribute, typename T_FunctorDescription>
    HINLINE auto createSymlog(
        AxisSplitting<T_Attribute> const& axSplit,
        double linearThreshold,
        T_FunctorDescription const& functorDesc)
    {
        static_assert(
            std::is_same_v<typename T_FunctorDescription::QuantityType, T_Attribute>,
            "Access functor return type and range type should be the same");

        return SymlogAxis<T_Attribute, typename T_FunctorDescription::FunctorType>(
            axSplit,
            functorDesc.functor,
            linearThreshold,
            functorDesc.name,
            functorDesc.units);
    }

} // namespace picongpu::plugins::binning::axis
