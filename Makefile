all:
	g++ gol.cpp -o gol -g -lOpenCL -lglut -lGLEW -lGLU -lGL -fopenmp

clean:
	rm -rf gol
