CXX = mpic++
FLAGS = -std=c++20 -O3 #-Wall
LIBS = -llapack -lboost_serialization -lboost_mpi -larmadillo
SRC = ${wildcard *.cpp}
OBJ = ${patsubst %.cpp, build/%.o, $(SRC)}
INCLUDE = -I../include/
TARGET = tests.out

$(TARGET): $(OBJ)
	$(CXX) $(FLAGS) -o $(TARGET) $(OBJ) $(LIBS)

build/%.o : %.cpp
	mkdir -p ${dir $@}
	$(CXX) $(FLAGS) -o $@ $< -c $(INCLUDE)

clean:
	rm -rf $(TARGET) build