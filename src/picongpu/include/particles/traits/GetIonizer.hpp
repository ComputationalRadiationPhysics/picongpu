/* 
 * File:   GetIonizer.hpp
 * Author: noir
 *
 * Created on 5. November 2014, 11:39
 */

#pragma once

#include "simulation_defines.hpp"
#include "traits/GetFlagType.hpp"

namespace picongpu
{

template<typename T_Species>
struct GetIonizer
{
    typedef typename GetFlagType<typename T_Species::FrameType, particleIonizer<> >::type::ThisType type;
};

}// namespace picongpu

