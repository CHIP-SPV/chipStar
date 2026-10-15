// The constant-offset shuffles HipShuffleShiftPass emits, for devices that
// keep the indexed shuffle.

int __chip_sg_shift_ok(uint idx, int d) { return 0; }

#define SHIFT(S, T)                                                            \
  T __attribute__((overloadable)) sub_group_shuffle(T var, uint idx);          \
  T __chip_sg_shift_##S(T v, uint idx, int d) {                                \
    return sub_group_shuffle(v, idx);                                          \
  }
SHIFT(i, int)
SHIFT(j, uint)
SHIFT(l, long)
SHIFT(m, ulong)
SHIFT(f, float)
