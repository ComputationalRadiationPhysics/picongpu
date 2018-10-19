/* Copyright 2013-2018 Axel Huebl, Rene Widera, Benjamin Worpitz
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

#include <pmacc/verify.hpp>
#include <vector>   // std::vector
#include <string>   // std::string
#include <utility>  // std::pair
#include <iterator> // std::distance

#include <boost/regex.hpp>
#include <boost/lexical_cast.hpp>

namespace picongpu
{

class ParserGridDistribution
{
private:
    typedef std::vector<std::pair<uint32_t, uint32_t> > value_type;

public:
    ParserGridDistribution( const std::string s )
    {
        parseString( s );
    }

    uint32_t
    getOffset( const int gpuPos, const uint32_t maxCells ) const
    {
        value_type::const_iterator iter = parsedInput.begin();
        // go to last gpu of this block b{n}
        int i = iter->second - 1;
        int sum = 0;

        while( i < gpuPos )
        {
            // add last block
            sum += iter->first * iter->second;

            ++iter;
            // go to last gpu of this block b{n}
            i += iter->second;
        }

        // add part of this block that is before me
        sum += iter->first * ( gpuPos + iter->second - i - 1 );

        // check total number of cells
        uint32_t sumTotal = 0;
        for( iter = parsedInput.begin(); iter != parsedInput.end(); ++iter )
            sumTotal += iter->first * iter->second;

        PMACC_VERIFY( sumTotal == maxCells );

        return sum;
    }

    /** Get local Size of this dimension
     *
     *  \param[in] gpuPos as integer in the range [0, n-1] for this dimension
     *  \return uint32_t with local number of cells
     */
    uint32_t
    getLocalSize( const int gpuPos ) const
    {
        value_type::const_iterator iter = parsedInput.begin();
        // go to last gpu of this block b{n}
        int i = iter->second - 1;

        while( i < gpuPos )
        {
            ++iter;
            // go to last gpu of this block b{n}
            i += iter->second;
        }

        return iter->first;
    }

private:
    value_type parsedInput;

    /** Parses the input string to a vector of pairs
     *
     *  Parses the input string in the form a,b,c{n},d{m},e,f
     *  to a vector of pairs with base number (a,b,c,d,e,f) and multipliers
     *  (1,1,n,m,e,f)
     *
     *  \param[in] s as const std::string in the form a,b{n}
     *  \return std::vector<pair> with uint32_t (base, multiplier)
     */
    void
    parseString( const std::string s )
    {
        boost::regex regFind( "[0-9]+(\\{[0-9]+\\})*",
                              boost::regex_constants::perl );

        boost::sregex_token_iterator iter( s.begin( ), s.end( ),
                                           regFind, 0 );
        boost::sregex_token_iterator end;

        parsedInput.clear();
        parsedInput.reserve( std::distance( iter, end ) );

        for(; iter != end; ++iter )
        {
            std::string pM = *iter;

            // find multiplier n and base b of b{n}
            boost::regex regMultipl( "(.*\\{)|(\\})",
                                     boost::regex_constants::perl );
            std::string multipl = boost::regex_replace( pM, regMultipl, "" );
            boost::regex regBase( "\\{.*\\}",
                                  boost::regex_constants::perl );
            std::string base = boost::regex_replace( pM, regBase, "" );

            // no Multiplier {n} given
            if( multipl == *iter )
                multipl = "1";

            const std::pair<uint32_t, uint32_t> g(
                      boost::lexical_cast<uint32_t > ( base ),
                      boost::lexical_cast<uint32_t > ( multipl ) );
            parsedInput.push_back( g );
        }
    }

};

} // namespace picongpu
