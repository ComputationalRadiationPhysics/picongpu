/**
 * Copyright 2013-2016 Rene Widera
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

#include "assert.hpp"

#include <string>
#include "zlib.h"

class ZipConnector
{
public:

    size_t compress(void* out, void* in, size_t sizeIn, int compressLevel)
    {
        int ret;

        z_stream strm;

        strm.zalloc = Z_NULL;
        strm.zfree = Z_NULL;
        strm.opaque = Z_NULL;
        ret = deflateInit(&strm, compressLevel);
        if (ret != Z_OK)
            return 0;

        strm.avail_in = sizeIn;
        strm.next_in = (Bytef*) in;

        strm.avail_out = sizeIn;
        strm.next_out = (Bytef*) out;

        ret = deflate(&strm, Z_FINISH);
        PMACC_ASSERT(ret != Z_STREAM_ERROR);

        size_t compressedBytes = strm.total_out;

        (void) deflateEnd(&strm);
        return compressedBytes;
    }

    size_t decompress(void* out, void* in, size_t sizeIn,size_t sizeOut)
    {
        int ret;

        z_stream strm;
        /* allocate deflate state */
        strm.zalloc = Z_NULL;
        strm.zfree = Z_NULL;
        strm.opaque = Z_NULL;
        ret = inflateInit(&strm);
        if (ret != Z_OK)
            return 0;

        strm.avail_in = sizeIn;

        strm.next_in = (Bytef*) in;

        strm.avail_out = sizeOut;
        strm.next_out = (Bytef*) out;
        ret = inflate(&strm, Z_FINISH);
        PMACC_ASSERT(ret != Z_STREAM_ERROR);

        size_t uncompressedBytes = strm.total_out;

        (void) inflateEnd(&strm);
        return uncompressedBytes;
    }

private:

};

