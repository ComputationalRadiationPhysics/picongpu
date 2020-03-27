/* Copyright 2013-2020 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch,
 *                     Klaus Steiniger, Felix Schmitt, Benjamin Worpitz,
 *                     Juncheng E, Sergei Bastrakov
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

#include "picongpu/particles/traits/SpeciesEligibleForSolver.hpp"
#include "picongpu/plugins/ISimulationPlugin.hpp"
#include "picongpu/plugins/common/stringHelpers.hpp"
#include "picongpu/plugins/xrayDiffraction/Implementation.hpp"
#include "picongpu/plugins/xrayDiffraction/ReciprocalSpace.hpp"

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/dimensions/DataSpaceOperations.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>


namespace picongpu
{
namespace plugins
{
namespace xrayDiffraction
{

    using namespace pmacc;

    namespace po = boost::program_options;

    /** X-ray diffraction plugin
     *
     * The plugin computesX-ray diffraction intensity based on particle
     * positions. The intensities are computed on a lattice in the reciprocal
     * space defined via command-line parameters. The results are aggregated for
     * the whole global simulation domain and written to files.
     *
     * The implementation is based on the GAPD code developed by Juncheng E
     * and the paper J.C. E, L. Wang, S. Chen, Y.Y. Zhang, S.N. Luo.
     * GAPD: a GPU-accelerated atom-based polychromatic diffraction simulation
     * code // Journal of Synchrotron Radiation. 25, 604-611 (2018).
     *
     * @tparam T_Species species type
     */
    template< typename T_Species >
    class XrayDiffraction : public ISimulationPlugin
    {
    public:

        //! Create X-ray diffraction plugin
        XrayDiffraction();

        /** Run the plugin
        *
        * @param currentStep current time iteration
        */
        void notify(uint32_t currentStep);

        /** Register command line options
         *
         * @param desc option descriptions
         */
        void pluginRegisterHelp( po::options_description & desc ) override;

        //! Get text name of a plugin
        std::string pluginGetName() const override;

        /** Set mapping description for kernels
         *
         * @param newCellDescription mapping for kernels
         */
        void setMappingDescription( MappingDesc * newCellDescription ) override;

        /** Restart from a checkpoint
         *
         * @param currentStep current time iteration
         * @param restartDirectory restart directory
         */
        void restart(
            uint32_t restartStep,
            std::string const restartDirectory
        ) override;

        /** Save data for checkpoint
         *
         * @param currentStep current time iteration
         * @param directory output directory
         */
        void checkpoint(
            uint32_t currentStep,
            std::string const directory
        ) override;

    private:
        
        //! Implementation type
        using Implementation = detail::Implementation< T_Species >;

        //! Pointer to implementation
        std::unique_ptr< Implementation > pImpl;

        //! Mapping description for kernels
        MappingDesc cellDescription;

        //! Notification period
        std::string notifyPeriod;

        //! Prefix for command-line parameters and output
        std::string prefix;

        //! Start of the reciprocal space
        float3_X qMin;

        //! End of the reciprocal space
        float3_X qMax;

        //! Number of scattering vectors
        DataSpace< 3 > numScatteringVectors;

        //! Load the plugin
        void pluginLoad();

        //! Unload the plugin
        void pluginUnload() override;

    };

    template< typename T_Species >
    XrayDiffraction< T_Species >::XrayDiffraction():
        prefix( T_Species::FrameType::getName() + std::string("_xrayDiffraction") ),
        cellDescription( SuperCellSize::toRT() )
    {
        Environment<>::get().PluginConnector().registerPlugin( this );
    }

    template< typename T_Species >
    void XrayDiffraction< T_Species >::notify( uint32_t currentStep )
    {
        pImpl->operator()(
            currentStep,
            cellDescription
        );
    }

    template< typename T_Species >
    void XrayDiffraction< T_Species >::pluginRegisterHelp(
        po::options_description & desc
    )
    {
        desc.add_options()(
            ( prefix + ".period" ).c_str(),
            po::value< std::string >( &notifyPeriod ),
            "enable plugin [for each n-th step]" )(
            ( prefix + ".qx_max" ).c_str(),
            po::value< float_X >( &qMax[ 0 ] )->default_value( 5._X ),
            "reciprocal space range qx_max (A^-1)" )(
            ( prefix + ".qy_max" ).c_str(),
            po::value< float_X >( &qMax[ 1 ] )->default_value( 5._X ),
            "reciprocal space range qy_max (A^-1)" )(
            ( prefix + ".qz_max" ).c_str(),
            po::value<float_X>( &qMax[ 2 ] )->default_value( 0._X ),
            "reciprocal space range qz_max (A^-1)" )(
            ( prefix + ".qx_min" ).c_str(),
            po::value< float_X >( &qMin[ 0 ] )->default_value( -5._X ),
            "reciprocal space range qx_min (A^-1)" )(
            ( prefix + ".qy_min" ).c_str(),
            po::value< float_X >( &qMin[ 1 ] )->default_value( -5._X ),
            "reciprocal space range qy_min (A^-1)" )(
            ( prefix + ".qz_min" ).c_str(),
            po::value< float_X >( &qMin[ 2 ] )->default_value( 0._X ),
            "reciprocal space range qz_min (A^-1)" )(
            ( prefix + ".n_qx" ).c_str(),
            po::value< int >( &numScatteringVectors[ 0 ] )->default_value( 100 ),
            "reciprocal space size in qx" )(
            ( prefix + ".n_qy" ).c_str(),
            po::value< int >( &numScatteringVectors[ 1 ] )->default_value( 100 ),
            "reciprocal space size in qy" )(
            ( prefix + ".n_qz" ).c_str(),
            po::value< int >( &numScatteringVectors[ 2 ] )->default_value( 1 ),
            "reciprocal space size in qz" );
    }

    template< typename T_Species >
    std::string XrayDiffraction< T_Species >::pluginGetName() const
    {
        return std::string{
            "X-ray diffraction: calculate diffraction scattering "
            "intensity of a species"
        };
    }

    template< typename T_Species >
    void XrayDiffraction< T_Species >::setMappingDescription(
        MappingDesc * newCellDescription
    )
    {
        cellDescription = *newCellDescription;
    }

    template< typename T_Species >
    void XrayDiffraction< T_Species >::restart(
        uint32_t restartStep,
        std::string const restartDirectory
    )
    {
        // No state to be read
    }

    template< typename T_Species >
    void XrayDiffraction< T_Species >::checkpoint(
        uint32_t currentStep,
        std::string const restartDirectory
    )
    {
        // No state to be saved
    }

    template< typename T_Species >
    void XrayDiffraction< T_Species >::pluginLoad()
    {
        if (!notifyPeriod.empty())
        {
            // Subtract 1 so that the last point is qMax
            auto numQIntervals = DataSpace< 3 >::create( 1 );
            for( auto dim = 0u; dim < 3u; dim++ )
                numQIntervals[ dim ] = std::max(
                    numScatteringVectors[ dim ] - 1,
                    1
                );
            auto qStep = ( qMax - qMin ) /
                precisionCast< float_X >( numQIntervals );
            auto reciprocalSpace = detail::ReciprocalSpace{
                qMin,
                qStep,
                numScatteringVectors
            };
            pImpl = memory::makeUnique< Implementation >(
                reciprocalSpace,
                T_Species::FrameType::getName()
            );
            Environment<>::get().PluginConnector().setNotificationPeriod(
                this,
                notifyPeriod
            );
        }
    }

    template< typename T_Species >
    void XrayDiffraction< T_Species >::pluginUnload()
    {
        if( !notifyPeriod.empty() )
            CUDA_CHECK( cuplaGetLastError() );
    }

} // namespace xrayDiffraction
} // namespace plugins

namespace particles
{
namespace traits
{

    /** Check if species fulfills requirements of the X-ray diffraction plugin
     *
     * @tparam T_Species species to check
     * @tparam T_UnspecifiedSpecies any species
     */
    template<
        typename T_Species,
        typename T_UnspecifiedSpecies
    >
    struct SpeciesEligibleForSolver<
        T_Species,
        plugins::xrayDiffraction::XrayDiffraction< T_UnspecifiedSpecies >
    >
    {
        using Frame = typename T_Species::FrameType;

        using RequiredIdentifiers = MakeSeq_t<
            localCellIdx,
            position<>,
            weighting
        >;

        using type = typename pmacc::traits::HasIdentifiers<
            Frame,
            RequiredIdentifiers
        >::type;
    };

} // namespace traits
} // namespace particles
} // namespace picongpu
