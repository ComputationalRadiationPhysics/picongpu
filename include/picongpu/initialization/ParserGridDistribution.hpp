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
#include <iterator> // std::distance

#include <boost/regex.hpp>
#include <boost/lexical_cast.hpp>


namespace picongpu
{

class ParserGridDistribution
{
private:
    /** 1D sudomain extents
     *
     * Pair of extent and count entry in our grid distribution.
     *
     * For example, a single entry of the grid distribution a,b,c{n},d{m},e,f
     * is stored as entry (a,1) in SubdomainPair. Another as (b,1), another
     * n equally spaced subdomains as (c,n), another m subdomains of extent d
     * as (d,m), and so on.
     */
    struct SubdomainPair {
        // extent of the current subdomain
        uint32_t extent;
        // count of how often the subdomain shall be repeated
        uint32_t count;
    };
    using value_type = std::vector< SubdomainPair >;

public:
    ParserGridDistribution( std::string const s )
    {
        parsedInput = parse( s );
    }

    uint32_t
    getOffset( uint32_t const devicePos, uint32_t const maxCells ) const
    {
        value_type::const_iterator iter = parsedInput.begin();
        // go to last device of these n subdomains extent{n}
        uint32_t i = iter->count - 1u;
        uint32_t sum = 0u;

        while( i < devicePos )
        {
            // add last subdomain
            sum += iter->extent * iter->count;

            ++iter;
            // go to last device of these n subdomains extent{n}
            i += iter->count;
        }

        // add part of this subdomain that is before me
        sum += iter->extent * ( devicePos + iter->count - i - 1u );

        // check total number of cells
        uint32_t sumTotal = 0u;
        for( iter = parsedInput.begin(); iter != parsedInput.end(); ++iter )
            sumTotal += iter->extent * iter->count;

        PMACC_VERIFY( sumTotal == maxCells );

        return sum;
    }

    /** Get local Size of this dimension
     *
     *  \param[in] devicePos as unsigned integer in the range [0, n-1] for this dimension
     *  \return uint32_t with local number of cells
     */
    uint32_t
    getLocalSize( uint32_t const devicePos ) const
    {
        value_type::const_iterator iter = parsedInput.begin();
        // go to last device of these n subdomains extent{n}
        uint32_t i = iter->count - 1u;

        while( i < devicePos )
        {
            ++iter;
            // go to last device of these n subdomains extent{n}
            i += iter->count;
        }

        return iter->extent;
    }

    /** Verify the number of subdomains matches the devices
     *
     * Check that the number of subdomains in a dimension, after we
     * expanded all regexes, matches the number of devices for it.
     *
     * \param[in] numDevices number of devices for this dimension
     */
    void
    verifyDevices( uint32_t const numDevices ) const
    {
        uint32_t numSubdomains = 0u;
        for( SubdomainPair const & p : parsedInput )
            numSubdomains += p.count;

        PMACC_VERIFY( numSubdomains == numDevices );
    }

private:
    value_type parsedInput;

    /** Parses the input string to a vector of SubdomainPair(s)
     *
     * Parses the input string in the form a,b,c{n},d{m},e,f
     * to a vector of SubdomainPair with extent number (a,b,c,d,e,f) and
     * counts (1,1,n,m,e,f)
     *
     * \param[in] s as string in the form a,b{n}
     * \return std::vector<SubdomainPair> with 2x uint32_t (extent, count)
     */
    value_type
    parse( std::string const s ) const
    {
        boost::regex regFind( "[0-9]+(\\{[0-9]+\\})*",
                              boost::regex_constants::perl );

        boost::sregex_token_iterator iter( s.begin( ), s.end( ),
                                           regFind, 0 );
        boost::sregex_token_iterator end;

        value_type newInput;
        newInput.reserve( std::distance( iter, end ) );

        for(; iter != end; ++iter )
        {
            std::string pM = *iter;

            // find count n and extent b of b{n}
            boost::regex regCount(
                "(.*\\{)|(\\})",
                boost::regex_constants::perl
            );
            std::string count = boost::regex_replace( pM, regCount, "" );

            boost::regex regExtent(
                "\\{.*\\}",
                boost::regex_constants::perl
            );
            std::string extent = boost::regex_replace( pM, regExtent, "" );

            // no count {n} given (implies one)
            if( count == *iter )
                count = "1";

            const SubdomainPair g = {
                boost::lexical_cast< uint32_t > ( extent ),
                boost::lexical_cast< uint32_t > ( count )
            };
            newInput.emplace_back( g );
        }

        return newInput;
    }

};

} // namespace picongpu
