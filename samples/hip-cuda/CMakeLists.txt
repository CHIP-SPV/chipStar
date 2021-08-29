set(SAMPLES
    BinomialOption
    BitonicSort
    DCT
    dwtHaar1D
    FastWalshTransform
    FloydWarshall
    Histogram
#   still broken:
#    PrefixSum
    RecursiveGaussian
    SimpleConvolution
)

foreach (SAMPLE ${SAMPLES})
  add_subdirectory(${SAMPLE})
endforeach()
