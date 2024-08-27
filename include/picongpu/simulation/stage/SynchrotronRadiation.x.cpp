/* Copyright 2024 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Richard Pausch, Alexander Debus, Marco Garten,
 *                     Benjamin Worpitz, Alexander Grund, Sergei Bastrakov,
 *                     Filip Optolowicz
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

#include "picongpu/simulation/stage/SynchrotronRadiation.hpp"

#include "picongpu/simulation_defines.hpp"

#include "picongpu/param/synchrotron.param"
#include "picongpu/particles/creation/creation.hpp"
#include "picongpu/particles/filter/filter.hpp"
#include "picongpu/particles/synchrotron/AlgorithmSynchrotron.hpp"

#include <pmacc/meta/ForEach.hpp>
#include <pmacc/particles/traits/FilterByFlag.hpp>

#include <cstdint>

namespace picongpu::simulation::stage
{
    /** Call Synchrotron Algorithm of an electron species
     * called from: /include/picongpu/simulation/stage/SynchrotronRadiation.hpp
     *
     * Tests if species can radiate photons and calls the kernels to do that
     *
     * @tparam T_SpeciesType type or name as PMACC_CSTRING of particle species that is checked for Synchrotron
     */
    template<typename T_SpeciesType>
    struct CallSynchrotron
    {
        using SpeciesType = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_SpeciesType>;
        using FrameType = typename SpeciesType::FrameType;

        //! the following line only fetches the alias
        using FoundSynchrotronAlias = typename pmacc::traits::GetFlagType<FrameType, picongpu::synchrotron<>>::type;

        //! this now resolves the alias into the actual object type, a list of photons
        using DestinationSpecies = typename pmacc::traits::Resolve<FoundSynchrotronAlias>::type;

        /** Functor implementation
         *
         * @tparam T_CellDescription contains the number of blocks and blocksize
         *                           that is later passed to the kernel
         * @param cellDesc logical block information like dimension and cell sizes
         * @param currentStep The current time step
         * @param F1F2DeviceBuff GridBuffer on the device side containing F1 and F2 values for each particle
         */
        template<typename T_CellDescription>
        HINLINE void operator()(
            T_CellDescription cellDesc,
            const uint32_t currentStep,
            GridBuffer<float_X, 2>::DataBoxType F1F2DeviceBuff,
            std::shared_ptr<GridBuffer<int32_t, 1>> failedRequirementQ) const
        {
            DataConnector& dc = Environment<>::get().DataConnector();

            //! alias for pointer on source species
            auto srcSpeciesPtr = dc.get<SpeciesType>(FrameType::getName());
            //! alias for pointer on destination species
            auto photonsPtr = dc.get<DestinationSpecies>(DestinationSpecies::FrameType::getName());

            auto synchrotronFunctor = particles::synchrotron::AlgorithmSynchrotron<SpeciesType, DestinationSpecies>(
                currentStep,
                F1F2DeviceBuff,
                failedRequirementQ->getDeviceBuffer().getDataBox());

            particles::creation::createParticlesFromSpecies(*srcSpeciesPtr, *photonsPtr, synchrotronFunctor, cellDesc);
            photonsPtr->fillAllGaps();
        }
    };

    template<typename T_Number>
    T_Number SynchrotronRadiation::integrateAsExponential(
        T_Number const xLeft,
        T_Number const xMiddle,
        T_Number const xRight,
        T_Number const yLeft,
        T_Number const yMiddle)
    {
        if(xLeft == xMiddle)
        {
            return 0;
        }
        if(yLeft < 0 || yMiddle < 0)
        {
            std::cerr << "Caught exception when precomputing firstSynchrotronFunction" << std::endl;
            std::cerr << ", xLeft: " << xLeft << ", xMiddle: " << xMiddle << ", xRight: " << xRight << std::endl;
            std::cerr << "yLeft: " << yLeft << ", yMiddle: " << yMiddle << std::endl;
            throw std::runtime_error(
                "SynchrotronRadiation.hpp: integrateAsExponential(): yLeft and yMiddle must not be negative.");
        }
        if(yLeft == 0 || yMiddle == 0)
        {
            return 0;
        }
        if(yLeft == yMiddle)
        {
            return (xRight - xLeft) * yLeft;
        }

        //! fitting function: y = a * e^(b * x)
        T_Number const b = (math::log(yMiddle) - math::log(yLeft)) / (xMiddle - xLeft);
        T_Number const a = yLeft / math::exp(b * xLeft);

        T_Number const integral = (a / b) * (math::exp(b * xRight) - math::exp(b * xLeft));
        return integral;
    }

    float_64 SynchrotronRadiation::firstSynchrotronFunction(float_64 const zq)
    {
        //! for 'logEnd' and 'numberSamplePoints' from FirstSynchrotronFunctionParams
        using picongpu::particles::synchrotron::params::FirstSynchrotronFunctionParams;

        //! from zq to logEnd
        float_64 const log_start = std::log2(zq);
        float_64 const log_step = (FirstSynchrotronFunctionParams::logEnd - log_start)
            / (FirstSynchrotronFunctionParams::numberSamplePoints - 1);

        float_64 integral = 0.;

        float_64 xLeft;
        float_64 xRight = zq;

        for(unsigned int i = 0; i < FirstSynchrotronFunctionParams::numberSamplePoints - 1; ++i)
        {
            xLeft = xRight;
            xRight = math::pow(2., log_start + log_step * (i + 1));
            float_64 xMiddle = (xLeft + xRight) / 2.0;

            float_64 yLeft = std::cyl_bessel_k(5.0 / 3.0, xLeft);
            float_64 yMiddle = std::cyl_bessel_k(5.0 / 3.0, xMiddle);
            /** computes the integral over one interval: [xLeft, xRight] using the
             *  exponential approximation between: [xLeft, xMiddle]
             */
            integral += integrateAsExponential(xLeft, xMiddle, xRight, yLeft, yMiddle);
        }
        return zq * integral;
    }

    float_64 SynchrotronRadiation::secondSynchrotronFunction(float_64 const x)
    {
        return x * std::cyl_bessel_k(2. / 3, x);
    }

    void SynchrotronRadiation::setFailedRequirementQ(bool const value)
    {
        failedRequirementQ->getHostBuffer().getDataBox()(DataSpace<1>{0}) = static_cast<int32_t>(value);
        failedRequirementQ->hostToDevice();
    }

    bool SynchrotronRadiation::checkFailedRequirementQ()
    {
        failedRequirementQ->deviceToHost();
        return failedRequirementQ->getHostBuffer().getDataBox()(DataSpace<1>{0}) == true;
    }

    template<typename T_Accessor>
    void SynchrotronRadiation::setTableValuesF1F2(uint32_t const iZq, T_Accessor const accessor, float_X const value)
    {
        tableValuesF1F2->getHostBuffer().getDataBox()(DataSpace<2>{iZq, u32(accessor)}) = value;
    }

    SynchrotronRadiation::SynchrotronRadiation(MappingDesc const cellDescription) : cellDescription(cellDescription)
    {
        using pmacc::particles::traits::FilterByFlag;
        using SpeciesWithSynchrotron = typename FilterByFlag<VectorAllSpecies, picongpu::synchrotron<>>::type;

        auto const numSpeciesWithSynchrotronRadiation = pmacc::mp_size<SpeciesWithSynchrotron>::value;
        auto const enableSynchrotronRadiation = numSpeciesWithSynchrotronRadiation > 0;
        if(enableSynchrotronRadiation)
        {
            init();
        }
    }

    void SynchrotronRadiation::operator()(uint32_t const step)
    {
        using pmacc::particles::traits::FilterByFlag;
        using SpeciesWithSynchrotron = typename FilterByFlag<VectorAllSpecies, picongpu::synchrotron<>>::type;

        auto const numSpeciesWithSynchrotronRadiation = pmacc::mp_size<SpeciesWithSynchrotron>::value;
        auto const enableSynchrotronRadiation = numSpeciesWithSynchrotronRadiation > 0;
        if(enableSynchrotronRadiation)
        {
            //! call the synchrotron radiation for each particle species with the synchrotron attribute
            pmacc::meta::ForEach<SpeciesWithSynchrotron, CallSynchrotron<boost::mpl::_1>> synchrotronRadiation;

            synchrotronRadiation(
                cellDescription,
                step,
                tableValuesF1F2->getDeviceBuffer().getDataBox(),
                failedRequirementQ);

            //! check if the requirements are met
            if constexpr(particles::synchrotron::params::supressRequirementWarning == false)
            {
                if(checkFailedRequirementQ())
                {
                    if((failedRequirementPrinted) == false)
                    {
                        printf("Synchrotron Extension requirement1 or requirement2 failed; should be less than 0.1 "
                               "-> "
                               "reduce the timestep. \n\tCheck the requrement by specifying the predicted maxHeff "
                               "and "
                               "maxGamma in"
                               "\n\tpicongpu/lib/python/synchrotronRadiationExtension/synchrotronRequirements.py\n");
                        printf("This warning is printed only once per simulation. Next warnings are dots.\n");
                        failedRequirementPrinted = true;
                    }
                    printf(".");
                    //! reset the requirement flag
                    setFailedRequirementQ(false);
                }
            }
        }
    }

    void SynchrotronRadiation::init()
    {
        //! for "numberTableEntries", "minZqExponent" and "maxZqExponent" from InterpolationParams
        using picongpu::particles::synchrotron::params::InterpolationParams;
        auto data_space = DataSpace<2>{InterpolationParams::numberTableEntries, 2};
        auto grid_layout = GridLayout<2>{data_space};

        //! capture the space for table and variables
        tableValuesF1F2 = std::make_shared<GridBuffer<float_X, 2>>(grid_layout);
        failedRequirementQ = std::make_shared<GridBuffer<int32_t, 1>>(DataSpace<1>{1});
        //! set the values of variables to false
        setFailedRequirementQ(false);
        failedRequirementPrinted = false;

        constexpr float_64 minZqExp = InterpolationParams::minZqExponent;
        constexpr float_64 maxZqExp = InterpolationParams::maxZqExponent;
        constexpr float_64 tableEntries = InterpolationParams::numberTableEntries;

        //! precompute F1 and F2 on log scale
        for(uint32_t iZq = 0; iZq < tableEntries; iZq++)
        {
            /** inverse function for index retrieval:
             * index = (log2(zq) - minZqExp) / (maxZqExp - minZqExp) * (tableEntries-1);
             */
            float_64 zq = std::pow(2., minZqExp + (maxZqExp - minZqExp) * iZq / (tableEntries - 1));

            float_64 const F1 = firstSynchrotronFunction(zq);
            float_64 const F2 = secondSynchrotronFunction(zq);

            setTableValuesF1F2(iZq, Accessor::f1, static_cast<float_X>(F1));
            setTableValuesF1F2(iZq, Accessor::f2, static_cast<float_X>(F2));
        }
        //! move the data to the device
        tableValuesF1F2->hostToDevice();
    }
} // namespace picongpu::simulation::stage
