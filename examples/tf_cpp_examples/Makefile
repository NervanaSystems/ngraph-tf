# A simple makefile that builds the hello_tf example
TF_INCLUDE=
CXXFLAGS=-std=c++11 -Wall -pedantic \
	-I include \
	-I ../ngraph-tf/build/tensorflow/ \
	-I ../ngraph-tf/build/tensorflow/bazel-genfiles \
	-I ../ngraph-tf/src \
	-I ../ngraph-tf/logging

ODIR=obj
INCDIR = include
LIBDIR=lib
BIN = hello_tf

LIBS=-lngraph_bridge \
	-ltensorflow_framework \
	-ltensorflow_cc

SRC=$(wildcard *.cpp)
OBJ=$(SRC:%.cpp=%.o)

all: $(OBJ)
	$(CXX) -Llib  $(LIBS) -o $(BIN) $^

%.o: %.c
	$(CXX)  $@ -c $<

.PHONY: clean

clean:
	rm -f *.o
	rm $(BIN)
	
#all: $(OBJ)
	#$(CXX) -L$(LIBDIR) ${LIBS) -o $(BIN) $^
	#$(CXX) ${LIBS) -o $(BIN) $^
