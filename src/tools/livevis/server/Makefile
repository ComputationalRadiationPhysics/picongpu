# Copyright 2013-2016 Benjamin Schneider, Rene Widera, Axel Huebl
#
# This file is part of PIConGPU.
#
# PIConGPU is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PIConGPU is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PIConGPU.
# If not, see <http://www.gnu.org/licenses/>.

CC=g++
INC=-I$(RIVLIB_ROOT)/include
LIB=-L$(RIVLIB_ROOT)/lib
CFLAGS=-c -Wall -std=c++0x $(INC)
LDFLAGS=-lpthread $(LIB) -lrivlib -lvislibsys64 -lvislibbase64
SOURCES=main.cpp src/Visualization.cpp src/VisualizationServer.cpp src/net/tcp_acceptor.cpp src/net/tcp_connector.cpp src/net/tcp_stream.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=server

all: $(SOURCES) $(EXECUTABLE)
	rm -rf *.o src/*.o *~
	
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -rf *.o $(EXECUTABLE) src/*.o *~
