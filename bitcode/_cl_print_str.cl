
void __attribute__((used)) _cl_print_str(__generic const char *S) {
  unsigned Pos = 0;
  char C;
  while ((C = S[Pos]) != 0) {
    printf("%c", C);
    ++Pos;
  }
}
