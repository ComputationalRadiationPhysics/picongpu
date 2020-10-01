#pragma once

#include "picongpu/simulation_defines.hpp"

#include <pmacc/mappings/simulation/SubGrid.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>

namespace picongpu
{
namespace fields
{
namespace laserProfiles
{
namespace generalisedlaguerreBeam
{
    template< typename T_Params >
    struct Unitless : public T_Params
    {
        using Params = T_Params;

        static constexpr float_X WAVE_LENGTH = float_X( Params::WAVE_LENGTH_SI / UNIT_LENGTH ); // unit: meter
        static constexpr float_X PULSE_LENGTH = float_X( Params::PULSE_LENGTH_SI / UNIT_TIME ); // unit: seconds (1 sigma)
        static constexpr float_X AMPLITUDE = float_X( Params::AMPLITUDE_SI / UNIT_EFIELD ); // unit: Volt /meter
        static constexpr float_X W0 = float_X( Params::W0_SI / UNIT_LENGTH ); // unit: meter
        static constexpr float_X FOCUS_POS = float_X( Params::FOCUS_POS_SI / UNIT_LENGTH ); // unit: meter
        static constexpr float_X INIT_TIME = float_X( ( Params::PULSE_INIT * Params::PULSE_LENGTH_SI ) / UNIT_TIME ); // unit: seconds (full initialization length)

        /* initialize the laser not in the first cell is equal to a negative shift
         * in time
         */
        static constexpr float_X laserTimeShift = Params::initPlaneY * CELL_HEIGHT / SPEED_OF_LIGHT;

        static constexpr float_64 f = SPEED_OF_LIGHT / WAVE_LENGTH;

    };
} // namespace generalisedlaguerreBeam

//---------Temporal---------------
//Until we find the best way to store the informations for defining the generalised Laguerre mode.We will use this publib struct
/*struct struct Temp_generalisedlaguerre{
    std::vector<int>    all_rad_modes;
    std::vector<int>    all_azi_modes;
    std::vector<double> all_amp_modes;
    int                     num_modes;
};
 it will be defined in LWFAk80..../laser.param*/


namespace acc
{


    template< typename T_Unitless >
    struct GeneralisedLaguerreBeam : public T_Unitless
    {
        using Unitless = T_Unitless;

        float3_X m_elong;
        float_X m_phase;
        typename FieldE::DataBoxType m_dataBoxE;
        DataSpace< simDim > m_offsetToTotalDomain;
        DataSpace< simDim > m_superCellToLocalOriginCellOffset;

        /** Simple iteration algorithm to implement Laguerre polynomials for GPUs.
         *
         *  @param n order of the Laguerre polynomial
         *  @param x coordinate at which the polynomial is evaluated
         *  @return ...
         */
        HDINLINE float_X generalisedLaguerre( const uint32_t n,const uint32_t a, const float_X x )
        {

            uint32_t currentN = 0;
            float_X laguerreN = 0;

            while (currentN <= n )
            {
                //Core statement of the algorithm
                laguerreN += coefficient(n,currentN,a) * pow(x,currentN) ;
                currentN++;
            }
            return laguerreN;
        }

        HDINLINE uint32_t nChoosek( const uint32_t n,  uint32_t k )//calculates nchoosek
        {
            if (k > n) return 0;
            if (k * 2 > n) k = n-k;
            if (k == 0) return 1;

            uint32_t result = n;
            for( uint32_t i = 2; i <= k; ++i )
            {
                result *= (n-i+1);
                result /= i;
            }
            return result;
        }

        HDINLINE float_X coefficient(const uint32_t n,const uint32_t m, const uint32_t a)//finds mth coeficiend for the n,a laguerre mode
        {
            uint32_t fac = 1.0;
            for (uint32_t i=2;i<=m;++i) fac*= i;
            return pow(-1.0,m)*nChoosek(n+a,n-m)/float_X(fac);
        }
        /** Device-Side Constructor
         *
         * @param superCellToLocalOriginCellOffset local offset in cells to current supercell
         * @param offsetToTotalDomain offset to origin of global (@todo: total) coordinate system (possibly after transform to centered origin)
         */
        HDINLINE GeneralisedLaguerreBeam(
            typename FieldE::DataBoxType const & dataBoxE,
            DataSpace< simDim > const & superCellToLocalOriginCellOffset,
            DataSpace< simDim > const & offsetToTotalDomain,
            float3_X const & elong,
            float_X const phase
        ) :
            m_elong( elong ),
            m_phase( phase ),
            m_dataBoxE( dataBoxE ),
            m_offsetToTotalDomain( offsetToTotalDomain ),
            m_superCellToLocalOriginCellOffset( superCellToLocalOriginCellOffset )
        {
        }

        /** device side manipulation for init plane (transversal)
         *
         * @tparam T_Args type of the arguments passed to the user manipulator functor
         *
         * @param cellIndexInSuperCell ND cell index in current supercell
         */
        template< typename T_Acc >
        HDINLINE
        void operator( )(
            T_Acc const &,
            DataSpace< simDim > const & cellIndexInSuperCell
        )
        {
            // coordinate system to global simulation as origin
            DataSpace< simDim > const localCell(
                cellIndexInSuperCell +
                m_superCellToLocalOriginCellOffset
            );

            // transform coordinate system to center of x-z plane of initialization
            constexpr uint8_t planeNormalDir = 1u;
            DataSpace< simDim > offsetToCenterOfPlane( m_offsetToTotalDomain );
            offsetToCenterOfPlane[ planeNormalDir ] = 0; // do not shift origin of plane normal
            floatD_X const pos = precisionCast< float_X >( localCell + offsetToCenterOfPlane ) * cellSize.shrink< simDim >();
            // @todo add half-cells via traits::FieldPosition< Solver::NumicalCellType, FieldE >()

            // transversal position only
            floatD_X planeNoNormal = floatD_X::create( 1.0_X );
            planeNoNormal[ planeNormalDir ] = 0.0_X;

            float_X const r2 = pmacc::math::abs2( pos * planeNoNormal );
            float_X const phi = picongpu::math::atan2(pos.x(),pos.z());
            // calculate focus position relative to the laser initialization plane
            float_X const focusPos = Unitless::FOCUS_POS - pos.y();

            // rayleigh length (in y-direction)
            float_X const y_R = float_X( PI ) * Unitless::W0 * Unitless::W0 / Unitless::WAVE_LENGTH;

            // inverse radius of curvature of the beam's  wavefronts
            float_X const R_y_inv = -focusPos / ( y_R * y_R + focusPos * focusPos );

            // initialize temporary variables
            float_X etrans( 0.0_X );
            float_X etrans_norm( 0.0_X );

            // We need to check at some point that Temp_generalisedlaguerre
            // has no problems : eg l,p,Amplitudes has same length (*)

            for( uint32_t num_mod = 0 ; num_mod < Unitless::num_modes ; ++num_mod )
                etrans_norm += typename Unitless::all_amp_modes_t{}[num_mod];



            // beam waist in the near field: w_y(y=0) == W0
            float_X const w_y = Unitless::W0 * math::sqrt( 1.0_X + ( focusPos / y_R )*( focusPos / y_R ) );
            //! the Gouy phase shift
            float_X const xi_y = math::atan( -focusPos / y_R );

            if( Unitless::Polarisation == Unitless::LINEAR_X || Unitless::Polarisation == Unitless::LINEAR_Z )
            {

                for( uint32_t num_mod = 0 ; num_mod < Unitless::num_modes ; ++num_mod )
                {

                    int32_t radial_mod = typename Unitless::all_rad_modes_t{}[num_mod];
                    int32_t azimut_mod = typename Unitless::all_azi_modes_t{}[num_mod];

                    etrans += typename Unitless::all_amp_modes_t{}[num_mod]
                    * generalisedLaguerre(radial_mod,float_X(azimut_mod), 2.0_X * r2 / w_y / w_y )
                    * pow(math::sqrt(2*r2)/w_y,azimut_mod)
                    * math::exp( -r2 / w_y / w_y )
                    * math::cos( 2.0_X * float_X( PI ) / Unitless::WAVE_LENGTH * focusPos - 2.0_X * float_X( PI ) / Unitless::WAVE_LENGTH * r2 / 2.0_X * R_y_inv + float_X(azimut_mod) * phi +  ( 2._X * float_X( radial_mod ) +float_X(azimut_mod) + 1._X ) * xi_y + m_phase )
                    * math::exp( -( r2 / 2.0_X * R_y_inv - focusPos - m_phase / 2.0_X / float_X( PI ) * Unitless::WAVE_LENGTH )
                                 *( r2 / 2.0_X * R_y_inv - focusPos - m_phase / 2.0_X / float_X( PI ) * Unitless::WAVE_LENGTH )
                                   / SPEED_OF_LIGHT / SPEED_OF_LIGHT / ( 2.0_X * Unitless::PULSE_LENGTH ) / ( 2.0_X * Unitless::PULSE_LENGTH ) );
                }

                m_elong *= etrans / etrans_norm;
                //printf("m_elong : %f \n",m_elong);//Debug
            }
            else if( Unitless::Polarisation == Unitless::CIRCULAR )
            {
                for( uint32_t num_mod = 0 ; num_mod < Unitless::num_modes ; ++num_mod )
                {

                    int32_t radial_mod = typename Unitless::all_rad_modes_t{}[num_mod];
                    int32_t azimut_mod = typename Unitless::all_azi_modes_t{}[num_mod];


                    etrans += typename Unitless::all_amp_modes_t{}[num_mod]
                    * generalisedLaguerre(radial_mod,float_X(azimut_mod), 2.0_X * r2 / w_y / w_y )
                    * pow(math::sqrt(2*r2)/w_y,float_X(azimut_mod))
                    * math::exp( -r2 / w_y / w_y )
                    * math::cos( 2.0_X * float_X( PI ) / Unitless::WAVE_LENGTH * focusPos - 2.0_X * float_X( PI ) / Unitless::WAVE_LENGTH * r2 / 2.0_X * R_y_inv + float_X(azimut_mod) * phi +  ( 2._X * float_X( radial_mod ) +float_X(azimut_mod) + 1._X ) * xi_y + m_phase )
                    * math::exp( -( r2 / 2.0_X * R_y_inv - focusPos - m_phase / 2.0_X / float_X( PI ) * Unitless::WAVE_LENGTH )
                                 *( r2 / 2.0_X * R_y_inv - focusPos - m_phase / 2.0_X / float_X( PI ) * Unitless::WAVE_LENGTH )
                                   / SPEED_OF_LIGHT / SPEED_OF_LIGHT / ( 2.0_X * Unitless::PULSE_LENGTH ) / ( 2.0_X * Unitless::PULSE_LENGTH ) );
                }

                m_elong.x() *= etrans / etrans_norm;
                m_phase += float_X( PI / 2.0 );
                etrans = 0.0_X;

                for( uint32_t num_mod = 0 ; num_mod < Unitless::num_modes ; ++num_mod )
                {

                    int32_t radial_mod = typename Unitless::all_rad_modes_t{}[num_mod];
                    int32_t azimut_mod = typename Unitless::all_azi_modes_t{}[num_mod];

                    etrans += typename Unitless::all_amp_modes_t{}[num_mod]
                    * generalisedLaguerre(radial_mod,float_X(azimut_mod), 2.0_X * r2 / w_y / w_y )
                    * pow(math::sqrt(2*r2)/w_y,float_X(azimut_mod))
                    * math::exp( -r2 / w_y / w_y )
                    * math::cos( 2.0_X * float_X( PI ) / Unitless::WAVE_LENGTH * focusPos - 2.0_X * float_X( PI ) / Unitless::WAVE_LENGTH * r2 / 2.0_X * R_y_inv + float_X(azimut_mod) * phi +  ( 2._X * float_X( radial_mod ) +float_X(azimut_mod) + 1._X ) * xi_y + m_phase )
                    * math::exp( -( r2 / 2.0_X * R_y_inv - focusPos - m_phase / 2.0_X / float_X( PI ) * Unitless::WAVE_LENGTH )
                                 *( r2 / 2.0_X * R_y_inv - focusPos - m_phase / 2.0_X / float_X( PI ) * Unitless::WAVE_LENGTH )
                                   / SPEED_OF_LIGHT / SPEED_OF_LIGHT / ( 2.0_X * Unitless::PULSE_LENGTH ) / ( 2.0_X * Unitless::PULSE_LENGTH ) );
                }
                m_elong.z() *= etrans / etrans_norm;
                // reminder: if you want to use phase below, substract pi/2
                // m_phase -= float_X( PI / 2.0 );
            }

            if( Unitless::initPlaneY != 0 ) // compile time if
            {
                /* If the laser is not initialized in the first cell we emit a
                 * negatively and positively propagating wave. Therefore we need to multiply the
                 * amplitude with a correction factor depending of the cell size in
                 * propagation direction.
                 * The negatively propagating wave is damped by the absorber.
                 *
                 * The `correctionFactor` assume that the wave is moving in y direction.
                 */
                auto const correctionFactor = ( SPEED_OF_LIGHT * DELTA_T ) / CELL_HEIGHT * 2._X;

                // jump over the guard of the electric field
                m_dataBoxE( localCell + SuperCellSize::toRT() * GuardSize::toRT() ) +=  correctionFactor * m_elong;
            }
            else
            {
                // jump over the guard of the electric field
                m_dataBoxE( localCell + SuperCellSize::toRT() * GuardSize::toRT() ) = m_elong;
            }
        }
    };
} // namespace acc

    template< typename T_Params >
    struct GeneralisedLaguerreBeam  : public  generalisedlaguerreBeam::Unitless< T_Params >
    {
        using Unitless = generalisedlaguerreBeam::Unitless< T_Params >;

        float3_X elong;
        float_X phase;
        typename FieldE::DataBoxType dataBoxE;
        DataSpace< simDim > offsetToTotalDomain;

        /** constructor
         *
         * @param currentStep current simulation time step
         */
        HINLINE GeneralisedLaguerreBeam( uint32_t currentStep )
        {
            // get data
            DataConnector & dc = Environment< >::get( ).DataConnector( );
            dataBoxE = dc.get< FieldE >(
                FieldE::getName(),
                true
            )->getDeviceDataBox();

            // get meta data for offsets
            SubGrid< simDim > const & subGrid = Environment< simDim >::get().SubGrid();
            // const DataSpace< simDim > totalCellOffset( subGrid.getGlobalDomain().offset );
            DataSpace< simDim > const globalCellOffset( subGrid.getLocalDomain().offset );
            DataSpace< simDim > const halfSimSize( subGrid.getGlobalDomain().size / 2 );

            // transform coordinate system to center of global simulation as origin [cells]
            offsetToTotalDomain = /* totalCellOffset + */ globalCellOffset - halfSimSize;

            // @todo reset origin of direction of moving window
            // offsetToTotalDomain.y() = 0

            float_64 const runTime = DELTA_T * currentStep - Unitless::laserTimeShift;

            // calculate focus position relative to the laser initialization plane
            float_X const focusPos = Unitless::FOCUS_POS - Unitless::initPlaneY * CELL_HEIGHT;

            elong = float3_X::create( 0.0_X );

            // This check is done here on HOST, since std::numeric_limits<float_X>::epsilon() does not compile on laserTransversal(), which is on DEVICE.
            float_X etrans_norm( 0.0_X );

           //(*) we should add a message here
            for( uint32_t num_mod = 0 ; num_mod < Unitless::num_modes ; ++num_mod )
                etrans_norm += typename  Unitless::all_amp_modes_t{}[num_mod];



            // a symmetric pulse will be initialized at position z=0 for
            // a time of PULSE_INIT * PULSE_LENGTH = INIT_TIME.
            // we shift the complete pulse for the half of this time to start with
            // the front of the laser pulse.
            constexpr float_64 mue = 0.5 * Unitless::INIT_TIME;

            // rayleigh length (in y-direction)
            constexpr float_64 y_R = PI * Unitless::W0 * Unitless::W0 / Unitless::WAVE_LENGTH;
            // gaussian beam waist in the nearfield: w_y(y=0) == W0
            float_64 const w_y = Unitless::W0 * math::sqrt( 1.0 + ( focusPos / y_R )*( focusPos / y_R ) );

            float_64 envelope = float_64( Unitless::AMPLITUDE );
            if( simDim == DIM2 )
                envelope *= math::sqrt( float_64( Unitless::W0 ) / w_y );
            else if( simDim == DIM3 )
                envelope *= float_64( Unitless::W0 ) / w_y;
            /* no 1D representation/implementation */

            if( Unitless::Polarisation == Unitless::LINEAR_X )
            {
                elong.x() = float_X( envelope );
            }
            else if( Unitless::Polarisation == Unitless::LINEAR_Z )
            {
                elong.z() = float_X( envelope );
            }
            else if( Unitless::Polarisation == Unitless::CIRCULAR )
            {
                elong.x() = float_X( envelope ) / math::sqrt( 2.0_X );
                elong.z() = float_X( envelope ) / math::sqrt( 2.0_X );
            }

            phase = 2.0_X * float_X( PI ) * float_X( Unitless::f ) * ( runTime - float_X( mue ) - focusPos / SPEED_OF_LIGHT ) + Unitless::LASER_PHASE;
        }

        /** create device manipulator functor
         *
         * @tparam T_WorkerCfg pmacc::mappings::threads::WorkerCfg, configuration of the worker
         * @tparam T_Acc alpaka accelerator type
         *
         * @param alpaka accelerator
         * @param localSupercellOffset (in supercells, without guards) to the
         *        origin of the local domain
         * @param configuration of the worker
         */
        template<
            typename T_WorkerCfg,
            typename T_Acc
        >
        HDINLINE acc::GeneralisedLaguerreBeam< Unitless >
        operator()(
            T_Acc const &,
            DataSpace< simDim > const & localSupercellOffset,
            T_WorkerCfg const &
        ) const
        {
            auto const superCellToLocalOriginCellOffset = localSupercellOffset * SuperCellSize::toRT();
            return acc::GeneralisedLaguerreBeam< Unitless >( dataBoxE, superCellToLocalOriginCellOffset, offsetToTotalDomain, elong, phase );
        }

        //! get the name of the laser profile
        static
        HINLINE std::string
        getName( )
        {
            return "Generalised Laguerre Beam";
        }

    };

} // namespace laserProfiles
} // namespace fields
} // namespace picongpu

