/*
  mallocMC: Memory Allocator for Many Core Architectures.
  http://www.icg.tugraz.at/project/mvp
  https://www.hzdr.de/crp

  Copyright (C) 2012 Institute for Computer Graphics and Vision,
                     Graz University of Technology
  Copyright (C) 2014-2024 Institute of Radiation Physics,
                          Helmholtz-Zentrum Dresden - Rossendorf

  Author(s):  Markus Steinberger - steinberger ( at ) icg.tugraz.at
              Bernhard Kainz - kainz ( at ) icg.tugraz.at
              Michael Kenzel - kenzel ( at ) icg.tugraz.at
              Rene Widera - r.widera ( at ) hzdr.de
              Axel Huebl - a.huebl ( at ) hzdr.de
              Carlchristian Eckert - c.eckert ( at ) hzdr.de
              Julian Lenz - j.lenz ( at ) hzdr.de

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.
*/

#pragma once

/** the mallocMC version: major API changes should be reflected here */
#define MALLOCMC_VERSION_MAJOR 3
#define MALLOCMC_VERSION_MINOR 0
#define MALLOCMC_VERSION_PATCH 0

/** the mallocMC flavor is used to differentiate the releases of the
 *  Computational Radiation Physics group (crp) from other releases
 *  This should be useful to avoid versioning conflicts */
#define MALLOCMC_FLAVOR "crp"
