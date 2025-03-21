echo "Compiling g.C with gcc: gcc -I. -O -c g.C"
gcc -I. -O -c g.C
echo "Creating static library: ar cr libgzstream.a g.o"
ar cr libgzstream.a g.o

echo "Compiling main with hipcc: hipcc -I. -O -c main.C"
hipcc -I. -O -c main.C
echo "Linking main with hipcc: hipcc -v  main.o ./libgzstream.a  -o t"
HIPCC_VERBOSE=7 hipcc -v  main.o ./libgzstream.a  -o t
echo "Running main: ./t"
./t
