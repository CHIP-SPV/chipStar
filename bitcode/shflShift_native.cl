// The constant-offset shuffles HipShuffleShiftPass emits, for devices where a
// shuffle down/up compiles to register moves and an indexed shuffle to a
// gather.

// Nonzero if every active lane reads its own lane or its lane + d, wrapping
// at the full sub-group.
int __chip_sg_shift_ok(uint idx, int d) {
  int l = get_sub_group_local_id(), n = get_sub_group_size();
  int e = l + d < 0 ? l + d + n : l + d >= n ? l + d - n : l + d;
  return n == get_max_sub_group_size() && d != 0 && d > -n && d < n &&
         sub_group_non_uniform_all(idx == l || idx == e);
}

// Returns sub_group_shuffle(v, idx) for an idx that is the lane, or the lane
// plus d wrapped at the max sub-group size, which HipShuffleShiftPass proves
// for max sizes 8 to 64.
#define SHIFT(S, T)                                                            \
  T __attribute__((overloadable)) sub_group_shuffle(T var, uint idx);          \
  T __chip_sg_shift_##S(T v, uint idx, int d) {                                \
    uint m = get_max_sub_group_size();                                         \
    if (m < 8 || m > 64 || (m & (m - 1)))                                      \
      return sub_group_shuffle(v, idx);                                        \
    T s = d > 0 ? intel_sub_group_shuffle_down(v, v, d)                        \
                : intel_sub_group_shuffle_up(v, v, -d);                        \
    return idx == get_sub_group_local_id() ? v : s;                            \
  }
SHIFT(i, int)
SHIFT(j, uint)
SHIFT(l, long)
SHIFT(m, ulong)
SHIFT(f, float)
