CC = mpicxx
CXX = mpicxx
CXXFLAGS = -g -O3
LDLIBS = -lscalapack-openmpi -lblacs-openmpi -lblacsCinit-openmpi -llapack -lblas
TARGET = main
ARGS = 10.0 0.1 0.12 210000000000.0 7850 24 1.0 15000 0.5

default: $(TARGET)

compile: $(TARGET)

all: $(TARGET)


task1: ./$(TARGET)
	./$(TARGET) ${ARGS} 1

task2: ./$(TARGET)
	./$(TARGET) ${ARGS} 2

task3: ./$(TARGET)
	./$(TARGET) ${ARGS} 3

task4: $(TARGET)
	mpiexec -np 2 ./$(TARGET) ${ARGS} 4

task5: $(TARGET)
	mpiexec -np 4 ./$(TARGET) ${ARGS} 5


.PHONY: clean

clean:
	rm -f $(TARGET) *.o
