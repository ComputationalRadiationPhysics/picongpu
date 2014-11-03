CC=g++
INC=-I$(PNGWRITER_ROOT)/include -I$(RIVLIB_ROOT)/include
LIB=-L$(PNGWRITER_ROOT)/lib  -L$(VISLIB_ROOT)/lib -L$(THELIB_ROOT)/lib -L$(RIVLIB_ROOT)/lib
CFLAGS=-c -Wall -std=c++0x $(INC) -DNO_FREETYPE
LDFLAGS=-lpthread -lpngwriter $(LIB) -lpng -lfreetype -lrivlib -lvislibsys64 -lvislibbase64 -lthelib
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
