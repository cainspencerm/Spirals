# Spirals
The purpose of the Spirals script is to create a computer-generated image. The image maps to GPUs with CUDA support.


### Examples
The following are examples of images created with the Python script. Their configuration details can be found in the [Examples](https://github.com/cainspencerm/Spirals/tree/master/Examples "Examples") folder with other examples.

#### Candy Cane
![CandyCane.png](https://raw.githubusercontent.com/cainspencerm/Spirals/master/Examples/CandyCane.png "CandyCane.png")

#### Swirls
![Swirls.png](https://raw.githubusercontent.com/cainspencerm/Spirals/master/Examples/Swirls.png "Swirls.png")


### Dependencies
In order to run the Python script, the following dependencies need to be installed:
- [numba](https://pypi.org/project/numba/ "numba") v0.51.2
- [pillow](https://pypi.org/project/Pillow/ "pillow") v8.1.0
- [numpy](https://pypi.org/project/numpy/ "numpy") v1.19.2


### Optimizations
When ran sequentially on an Intel i7-9700K, each image takes roughly 23.5877081 seconds.

##### 1. Parallelize the code.
The first step to optimizing the CGI algorithm was to reconstruct the algorithm for parallel use on the GPU. This was done by using the `@cuda.jit` decorator, which requires kernel invocation. Before the algorithm executes, the code determines how to map a pixel array onto the GPU hardware. In the algorithm, loops to calculate each pixel are replaced with `i, j = cuda.grid(2)`. If `i` and `j` are within the bounds of the image, the function computes the pixel color of the index. No loops. The speedup of parallelization is around 1,072, taking roughly 22.1124 milliseconds.

##### 2. Minimize host-to-device data transfer.
According to the `nvprof` CLI tool (in [Optimizations.txt](https://github.com/cainspencerm/Spirals/blob/master/Optimizations.txt "Optimizations.txt")), `[CUDA memcpy HtoD]` accounts for roughly `36.09%` of the execution. To bypass this, the pixel array can be created and initialized on the device without ever being added in CPU-side memory. `numba.cuda` provides a function `device_array` to do so. The speedup of minimizing host-to-device transfer is around 1.38, taking roughly 15.9563 milliseconds.

##### 3. Minimize device-side slow memory access.
The next optimization makes use of shared memory, memory that a block of threads shares. Since the memory is block specific, read and write speeds are faster. Therefore, memory access operations can be computed on shared memory. Following operations, one slow access to the global pixel array completes parallelized computation. The speedup of shared memory usage is around 1.068, taking roughly 14.9425 milliseconds.

The total speedup from the CPU algorithm to the optimized GPU algorithm is roughly 1,578.