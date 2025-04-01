set -eu

HIPCC=@CMAKE_BINARY_DIR@/bin/hipcc
HIPCC=/space/pvelesko/install/llvm/18.0/bin/clang++

echo '#include "gzstream.w.h"
void open( const char* name, int open_mode) {
}' > gzstream.w.C

echo 'void open( const char* name, int open_mode);' > gzstream.w.h

echo '#include "gzstream.w.h"
#include <stdlib.h>

int main( int argc, char*argv[]) {
  open( "c", 0);
  return 0;
}' > test.w.C

# Compile the library
${HIPCC} -c gzstream.w.C -o gzstream.w.o
ar cr libgzstream.w.a gzstream.w.o

# Compile main with -c
${HIPCC} -c test.w.C -o test.w.o

# Link via -L. -lgzstream.w
echo "${HIPCC} test.w.o -o test -L. -lgzstream.w"
${HIPCC} test.w.o -o test -L. -lgzstream.w
