__kernel void myKernel() {
    float val = 0.0f;
    for (int i = 0; i < 1000000; i++) {
        for (int j = 0; j < 10000; j++) {
            val += sqrt(val + i + j);
        }
    }

//     if (get_global_id(0) == 0 && get_local_id(0) == 0) {
//         printf("complete\n");
//     }

}