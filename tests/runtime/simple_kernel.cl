__kernel void simple_kernel(__global char *ptr1, __global char *ptr2, int n) {
    int id = get_global_id(0);
    if (id < n) {
        ptr2[id] = ptr1[id];
    }
}
