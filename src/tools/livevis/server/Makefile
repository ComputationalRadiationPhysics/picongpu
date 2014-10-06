CC=g++
INC=-I/home/widera/lib/pngwriter/include -I/home/schnei49/libs/rivlib/trunk/rivlib/include
LIB=-L/home/widera/lib/pngwriter/lib  -L/home/schnei49/libs/vislib/lib -L/home/schnei49/libs/thelib/thelib++/build -L/home/schnei49/libs/rivlib/trunk/build/rivlib
CFLAGS=-c -Wall -std=c++0x $(INC) -DNO_FREETYPE
LDFLAGS=-lpthread -lpngwriter $(LIB) -lpng -lfreetype -lrivlib -lvislibsys64 -lvislibbase64 -lthelib
SOURCES=main.cpp src/Visualization.cpp src/VisualizationServer.cpp src/net/tcp_acceptor.cpp src/net/tcp_connector.cpp src/net/tcp_stream.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=server

all: $(SOURCES) $(EXECUTABLE)
	rm *.o src/*.o *~
	
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm *.o $(EXECUTABLE) src/*.o *~
