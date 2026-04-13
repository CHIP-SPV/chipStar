/* Pure C helper used by the regression test for issue #1001.
   'class' is a reserved keyword in C++ / HIP but a valid identifier in C.
   If this file is compiled with -x hip the compiler will reject it. */
int get_value(void) {
  int class = 42;
  return class;
}
