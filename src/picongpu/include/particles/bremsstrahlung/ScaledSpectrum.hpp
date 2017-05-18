/* Copyright 2016-2017 Heiko Burau
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

#include "particles/traits/GetAtomicNumbers.hpp"

#include "cuSTL/cursor/Cursor.hpp"
#include "cuSTL/cursor/navigator/PlusNavigator.hpp"
#include "cuSTL/cursor/tools/LinearInterp.hpp"
#include "cuSTL/cursor/BufferCursor.hpp"
#include "algorithms/math.hpp"
#include <boost/array.hpp>
/* `array_wrapper.hpp` must be included before `integrate.hpp` to avoid
 * the error
 * `boost/numeric/ublas/matrix.hpp(5977): error: namespace "boost::serialization" has no member "make_array"`
 * in boost 1.64.0
 * see boost issue https://svn.boost.org/trac/boost/ticket/12516
 */
#include <boost/serialization/array_wrapper.hpp>
#include <boost/numeric/odeint/integrate/integrate.hpp>
#include <boost/shared_ptr.hpp>
#include <limits>

namespace picongpu
{
namespace particles
{
namespace bremsstrahlung
{

namespace detail
{

/** Functor for the scaled differential cross section (dcs) which
 * equals to the electron energy loss times the cross section per unit energy.
 */
struct LookupTableFunctor
{
    typedef typename ::PMacc::result_of::Functor<
        ::PMacc::cursor::tools::LinearInterp<float_X>,
        ::PMacc::cursor::BufferCursor<float_X, DIM2> >::type LinInterpCursor;

    typedef float_X type;

    LinInterpCursor linInterpCursor;
    float_X lnEMin;
    float_X lnEMax;

    /** constructor
     *
     * @param linInterpCursor
     */
    HDINLINE LookupTableFunctor(LinInterpCursor linInterpCursor);
    /** scaled differential cross section
     *
     * @param Ekin kinectic energy of the incident electron
     * @param kappa energy loss normalized to Ekin
     */
    HDINLINE float_X operator()(const float_X Ekin, const float_X kappa) const;
};

} // namespace detail


/** Generates and holds the lookup tables for the scaled differential cross section
 * and the stopping power.
 *
 * scaled differential cross section = electron energy loss times cross section per unit energy
 *
 * stopping power = energy loss per unit length
 *
 * The lookup tables are generated from the screened Bethe-Heitler cross section. See e.g.:
 * Salvat, F., et al. "Monte Carlo simulation of bremsstrahlung emission by electrons."
 * Radiation Physics and Chemistry 75.10 (2006): 1201-1219.
 */
struct ScaledSpectrum
{
public:
    typedef detail::LookupTableFunctor LookupTableFunctor;
private:

    typedef boost::shared_ptr<PMacc::container::DeviceBuffer<float_X, DIM2> > MyBuf;
    MyBuf dBufScaledSpectrum;
    MyBuf dBufStoppingPower;

    /** differential cross section: cross section per unit energy
     *
     * This is the screened Bethe-Heitler cross section. See e.g.:
     * Salvat, F., et al. "Monte Carlo simulation of bremsstrahlung emission by electrons."
     * Radiation Physics and Chemistry 75.10 (2006): 1201-1219.
     *
     * @param Ekin kinetic electron energy
     * @param kappa energy loss normalized to Ekin
     * @param targetZ atomic number of the target material
     */
    float_64 dcs(const float_64 Ekin, const float_64 kappa, const float_64 targetZ) const;

    /** differential cross section times energy loss
     */
    struct StoppingPowerIntegrand
    {
        const float_64 Ekin;
        const float_64 targetZ;
        const ScaledSpectrum& scaledSpectrum;

        StoppingPowerIntegrand(const float_64 Ekin, const ScaledSpectrum& scaledSpectrum, const float_64 targetZ) :
            Ekin(Ekin), scaledSpectrum(scaledSpectrum), targetZ(targetZ) {}

        template<typename T_State, typename T_W>
        void operator()(const T_State &x, T_State &dxdW, T_W W) const
        {
            dxdW[0] = this->scaledSpectrum.dcs(this->Ekin, W / this->Ekin, this->targetZ) * W;
        }
    };

public:

    /** Generate lookup tables
     *
     * @param targetZ atomic number of the target material
     */
    void init(const float_64 targetZ);

    /** Return a functor representing the scaled differential cross section
     *
     * scaled differential cross section = electron energy loss times cross section per unit energy
     */
    LookupTableFunctor getScaledSpectrumFunctor() const;

    /** Return a functor representing the stopping power
     *
     * stopping power = energy loss per unit length
     */
    LookupTableFunctor getStoppingPowerFunctor() const;
};


/** Creates a `ScaledSpectrum` instance for a given electron species
 * and stores it in a map<atomic number, ScaledSpectrum> object.
 *
 * This functor is called from MySimulation::init() to generate lookup tables.
 */
template<typename T_ElectronSpecies>
struct FillScaledSpectrumMap
{
    typedef T_ElectronSpecies ElectronSpecies;

    typedef typename PMacc::particles::traits::ResolveAliasFromSpecies<
        ElectronSpecies,
        bremsstrahlungIons<>
    >::type IonSpecies;


    template<typename T_Map>
    void operator()(T_Map& map) const
    {
        const float_X targetZ = GetAtomicNumbers<IonSpecies>::type::numberOfProtons;

        if(map.count(targetZ) == 0)
        {
            ScaledSpectrum scaledSpectrum;
            scaledSpectrum.init(static_cast<float_64>(targetZ));
            map[targetZ] = scaledSpectrum;
        }
    }
};

} // namespace bremsstrahlung
} // namespace particles
} // namespace picongpu
