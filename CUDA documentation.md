# **Notes from CUDA Programming Guide**


<span style="font-size:16px">

## **Introduction**

GPU is specialised for compute-intensive, highly parallel computation - exactly what graphics rendering is about.<br>

![](./Pictures/CPU-GPU%20trend.png)

More Transistors are used for Comptue(data processing) rather caching and flow control.<br>
More focussed on throughput rather latency.


![](./Pictures/GPU%20vs%20CPU.png)

It is also highly scalable programming model.<br> There are three key abstraction:
* A hierarchy of thread groups.
* shared memories.
* barrier sychronization.

These abstraction provide fine grained <b>fine grained data and thread parallelism</b> nested within <b>coarse-grained data and task parallelism</b>

![](./Pictures/scalable%20programming%20model.png)

GPU are build around an array of <b>Streaming Multiprocessors</b>. So more SM's less time to execute CUDA code.

## **CUDA Programming Model**

* **CUDA** stands for **Compute Unified Device Architecture**

* **Programming model**: It is the programmer's view of how the code gets executed.

### Kernels
* C - like functions that will be called N times in parallel by N-different CUDA threads.<br>

```C++
__global__ void function([arguments]){
  // body of the funtion
}

/* 
function call
 <<<...>>> - excution configuration syntax.
 */


function<<<...>>>([arguments])
```

* Each Thread has a unique **thread ID** which can be accessed using built-in **threadIdx** variable.<br>
A simple **VectorAddition** function in C++.

```C++
#include <iostream>

int* vectorAdd(int* A, int* B, int n){
  int *C = new int[n];
  for(int i = 0; i < n; i++){
    C[i] = A[i] + B[i];
  }
  return C;
}

int main(){
  int n = 10;
  int *A = new int[n], *B = new int[n];
  srand(time(NULL));
  for(int i = 0; i < n; i++){
    A[i] = rand();
    B[i] = rand();
  }

  int* C = vectorAdd(A, B, n);
  return 0;
}

```

> In CUDA C++

```C++
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

__global__
void vectorAdd(int* A, int* B, int *C int n){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= n){
    return;
  }

  C[tid] = A[tid] + B[tid];

}

int main(){
  int n = 10;
  int *A = new int[n], *B = new int[n];
  int *C = new int[n];
  srand(time(NULL));
  for(int i = 0; i < n; i++){
    A[i] = rand();
    B[i] = rand();
  }

  int *d_A, *d_B, *d_C;
  //perform Memory copy to GPU.
  cudaMalloc((void**)&d_A, n * sizeof(int));
  cudaMemcpy(d_A, A, n * sizeof(int),cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_B, n * sizeof(int));
  cudaMemcpy(d_B, B, n * sizeof(int),cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_C, n * sizeof(int));
  cudaMemcpy(d_C, C, n * sizeof(int),cudaMemcpyHostToDevice);

  vectorAdd<<<1, n>>>(d_A, d_B, d_C, n);

  cudaMemcpy(C, d_C, n * sizeof(int),cudaMemcpyDeviceToHost);

  return 0;
}

```

* The above code output the same as the serial code as long as **n < 1024**.

### **Thread Hierarchy**

**threadIdx** has three components.
1. threadIdx.x
2. threadIdx.y
3. threadIdx.z

* **Block**: Threads are organized into a 3-dimensional structure called **Block**.

* Each thread can be identified in one, two or three dimension in a **thread block**.
  * **one-dimensional block**:  threadIdx.x
  * **two-dimensional block**:  threadIdx.x + threadIdx.y * blockDim.x
  * **three-dimensional block**: threadIdx.x + threadIdx.y * blockIdx.x + threadIdx.z * blockDim.x * blockDim.y

    * The above equation applies for a single block. It gets complicated when there are multiple blocks.

* **Grid**: Blocks are organized into 3-dimensional structure called **grid**.

![](./Pictures/Block%20structure.png)

* The number and dimension of threads in block and blocks in grid are represented using **_int_** or **_dim3_** type inside <<<...>>>.

```C++
// Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N],
float C[N][N])
{
 int i = blockIdx.x * blockDim.x + threadIdx.x;
 int j = blockIdx.y * blockDim.y + threadIdx.y;
 if (i < N && j < N)
 C[i][j] = A[i][j] + B[i][j];
}
int main()
{
 ...
 // Kernel invocation
 dim3 threadsPerBlock(16, 16);
 dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
 MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
 ...
}
```

* Thus each threads act on a single matrix element.
* Thread blocks are scheduled independently. This ensures **_scalability_**<br>
* Threads inside a block cooperate and share data through _shared memory_ and synchronize using _barrier synchronization_ using **__synchthreads()** intrinsic function. 

### **Memory Hierarchy**

* Per Threads **Private local memory**.
* Per Block **shared memory**.
* **Global Memory** visible to all threads.
* **read-only** device memory accessible by all threads<br>
  * constant memory
  * Texture memory<br>

The global, texture and constant memory spaces are optimized for different memory usages.

![](./Pictures/Memory%20hierarchy.png)

The global, constant and texture memory are persistent across.

### Heterogenous Programming

The Kernel function call is a non-blocking call meaning that the **main thread** continues to execute and the it can be blocked by **cudaDeviceSynchronize()**.

![](./Pictures/Kernel%20call.png)

### **Compute Capability**

It is a version number also called as **SM version**.<br>
version determines the features supported by the GPU hardware and is used by applications at runtime to determine which hardware features and instructions are available on the present **GPU**.

It is represented as **X.Y** where
* X - major version
* Y - minor version<br>
X - determines the core architecture.


## **Programming Model**

### **Compilation with NVCC**:

* **Kernels** are written using the **CUDA instruction set architecture** called **PTX(Parallel Thread eXecution)**. But it is effective to use a high language like C/C++/python/fortran. In both cases it must be compiled into a binary.

* **nvcc** or nvidia CUDA C compiler is a compiler driver compiles C or PTX to native instruction set that hardware supports.

### **Compilation workflow**: 

### Offline Compilation:

* Source file = Host code + Device code.
* Device code is compiled into an assembly form (PTX) and binary form(cubin object).
* host code is modified where **<<<>>>** is replaced with necessary **CUDA C runtime function call to load and launch from PTX code or cubin object**
* The host code is left a C code that is left to be compiled using another tool or object that is compiled by nvcc by invoking the host compiler.
* Application can either link to the complied host code or ignore the modified host code and use the CUDA driver API to load and execute PTX code or cubin object.

### Just in Time Compilation:

* **PTX** code is loaded by the application at runtime and is complied to **binary** by the **device driver**. This process is called **Just-in-time compilation**.<br>

* This allows latest instruction or feature to be used even before the new features are available.<br>

* The device driver caches the compiled binary code so as to decrease the application running time and it is invalidated when device driver is upgraded.<br>

* This affects the application load time.

### Binary Compatibility:

* Binary code is architecture specific. cubin object is generated using the compiler option **-code** that specifies the targeted architecture. e.g. **-code=sm_35** produces binary code for devices with **compute capability 3.5**.<br>

* A cubin object generated for **X.y** will one execute on devices of compute capability **X.z** where **z>=y**.

* Similarly **PTX compatibility** can be add using **"-arch="** flag. PTX produces for some specific compute capability can always be complied to binary code of greater or equal compute capability.

* Other flags include -m64 or -m32 for 64-bit or 32 bit device code compilation.

* -gencode arch=__,code=__ is used to specify the nature PTX and binary.

### **CUDA C Runtime**
* The **runtime** is implemented in **cudart** library.
* There are various API calls provide by the runtime to modify the device specific properties like memory.

### Initialization:
* There is no explicit intialization function for runtime in **CUDA**. it is initialized when the runtime function is called.

* The **CUDA runtime** creates a a CUDA context for each device in the system. This context is _primary context_ for this device and is shared across all the host threads.
* During context creation, the device code is either **just-in-time** compiled and loaded into device memory
* This all happens underneath the hood.
* This is part of the **CUDA programming model**.
* **_cudaDeviceReset()_** - destroys all the primary contexts of the device the host currently operates on.
* the next runtime function call will create a primary context.

### Device Memory
* According to the **CUDA programming model**, the host and device have separate memory.
* **Kernel** operate on **device memory**. so the runtime provides functions to operate on device memory like _allocate, deallocate and copy_ as well data transfer across PCIe bus from host to device and vice versa.

checkpoint 34

* Allocation Methods on device memory:
  * linear arrays
  * CUDA arrays.

> Linear arrays: C style array referenced by a pointer, it lives in a 40 bit address space.<br>

> CUDA arrays: opaque memory layouts optimized for texture fetching.

### Shared Memory

* Shared Memory is allocated statically using **\_\_shared\_\_** memory space specifier.

* Shared Memory can be allocated and accessed only within the block.

### Page Lock Host Memory

* The runtime provides facilities to use page-locked host memory, so that the OS doesn't page the host memory out.
* It can be done using **_cudaHostAlloc()_** and **_cudaFreeHost()_**.

* **_cudaHostRegister()_** page locks a range of memory allocated by malloc().

* But it is a scarce resource, too much use degrades the system performance, reducing memory for the host processes.

Advantages:

* Concurrent copy from host to device with kernel execution.

* Pinned memory can be mapped on to device's address space, eliminating the need to copy.
* Bandwidth is higher, even higher when allocated as **write-combining**.

### Portable Memory

* Page-locked memory is faster when the device was current when the block was allocated.

* To make page-locked memory portable **cudaHostAllocPortable** to **cudaHostAlloc()** or page-locked by passing the flag **cudaHostRegisterPortable** to **cudaHostRegister()**

### Write-Combining Memory

* Page-locked host memory is allocated as cacheable. It can optionally be allocated as _write-combining_ instead by passing flag **cudaHostAllocWriteCombined** to **cudaHostAlloc()**.

* It frees up the host's L1 and L2 cache resources, making more cache available to the rest of the application.

* it is not **snooped** during transfers across PCIe bus, which improve transfer performance by upto 40%.

* But write-combining memory has poor performance in terms of host code.

### Mapped Memory

* A block of page-locked host memory can also be mapped into the address space of the device using **cudaHostAllocMapped** to **cudaHostAlloc()** or by passing flag **cudaHostRegisterMapped** to **cudaHostRegister()**.

* Such block will have two address
  * one is host returned by **cudaHostAlloc**
  * other is device which can retrieved by **cudaHostGetDevicePointer()**

Advantages:

* No need to allocate block in device and copy data between this block adn block in host memory, data transfers are implicity performed.

* No need for streams to overlap with kernel launch, kernel-originated data transfers automatically overlap with kernel execution.

* To enable the feature, flag **cudaDeviceMapHost** must be passed to **cudaSetDeviceFlags()** before any call performed else **cudaGetdevicePointer()** will return error.

* It is not available on all cuda capable devices which can queried using **canMapHostMemory** from **cudaGetDeviceProperties()**.

### **Asynchronous Concurrent Execution**

* Concurrent tasks supported by CUDA.
  * Computation on the host.
  * Computation on the device.
  * Memory transfers from host to device.
  * Memory transfers from device to host.
  * Memory transfers within memory of a given device.
  * Memory transfers among devices.

the level of concurrency depends of feature set and compute capability.

### Concurrent Execution between Host and Device

* Concurrent host execution can take place through asynchronous library functions that return control to the host thread before the device completes the task.

* Asynchronous calls are queued up and executed by the CUDA driver when resources are available.

* This relieves the host.

* Asynchronous operations
  * Kernel launches.
  * Memory copies within single device's memory.
  * Memory copies from host to device with 64Kb or less.
  * Memory copies performed by functions that are suffixed with **Async**.
  * Memory set function calls.

* Can disable by setting **CUDA\_LAUNCH\_BLOCKING** environment variable.

* Async memory copies will also be synchronous if they involve host memory that is not page-locked.

### Concurrent Kernel Execution

* Multiple kernels concurrently.

* Can query if it exists by using **concurrentKernels** device property.

* **Maximum number of kernel launches** that a device can execute concurrently depends on it **compute capability**

* A kernel from one context cannot execute concurrently with another kernel in a different context.

* Kernels taht use many textures or large amount of local memory are less likely to execute concurrently with other kernels.

### Overlap of Data Transfer and Kernel Execution

* Memory copy can concurrently be done with kernel execution.

* If the device property **asyncEngineCount** > 0, then the feature is supported.

* If host memory is involved, then it must be page locked.

* intra-device copies can be done using standard memory copy functions with destination and source address residing on the same device.

### Concurrent Data Transfers

* concurrent copies to and from the device. feature can be checked using **asyncEngineCount** device property.

### Streams

* The above concurrent operations are managed using **streams**.
* **Streams** are sequence of commands that are executed **inorder**, but multiple streams can execute commands **out of order** or **concurrently**.


  ### Stream Creation and Destruction

    * Streams are created using **cudaStreamCreate(cudaStream_t)**

    * Streams object are created using **cudaStream_t** to access the stream.

    ```c++
        cudaStream_t stream[2];
        for(int i = 0; i < 2; i++){
          cudaStreamCreate(&stream[i]);
        }

        float* hostPtr;
        cudaMalloc(&hostPtr, 2 * size);

        for(int i = 0; i < 2; i++){
          
          cudaMemcpyAsync(inputDevPtr + i * size, hostPtr + i * size, size, cudaMemcpyHostToDevice, stream[i]);

          MyKernel<<<GridSize, BlockSize, 0, stream[i]>>>(outputDevPtr + i * size, inputDevPtr + i * size, size);

          cudaMemcpyAsync(hostPtr + i * size, outputDevPtr + i * size, size, cudaMemcpyDeviceToHost, stream[i]);

        }

        for(int i = 0; i < 2; i++){
          cudaStreamDestory(stream[i]);
        }

    ```

    ### Default Stream
    * If no streams are specified they are added to default stream.

    * The default stream is a special stream called the **NULL stream** and **all host thread share the same stream** if compiled with **--default-stream legacy** and also this is the default option if **--default-stream** flag is not used.

    * If compiled using **--default-stream per-thread** the default stream is a regular stream and **each host thread has its own default stream**.

    ### Explicit Synchronization

    > **cudaDeviceSynchronize()**: waits until all preceding commands in all streams of all host threads have completed.

    > **cudaStreamSynchronize()**: takes a stream as a parameter and waits until all preceding commands in the given stream

    > **cudaStreamWaitEvent()**: takes a stream as a parameter and makes all the commands added to the stream after the call and dealys their execution until the given event has completed.

    > **cudaStreamQuery()**: provides application with a way to know whether all preceding commands in a stream have completed.

    ### Implicit Synchronization
    
    * Two commands in different streams cannot execute concurrently when the following happens

      * A page-locked host memory allocation.
      * A device memory allocation.
      * A device memory set.
      * A memory copy between two address to the same device memory.
      * Any CUDA command to the NULL stream.

  ### Callbacks

  * Callbacks are host functions that execute when the preceding commands in the streams are completed.

  * Callbacks can be added using **cudaStreamAddCallback()** 
