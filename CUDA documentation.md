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

**CUDA** stands for **Compute Unified Device Architecture**

* **Programming model**: It is the programmer's view of how the code gets executed.

### Kernels
C - like functions that will be called N times in parallel by N-different CUDA threads.<br>

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

Each Thread has a unique **thread ID** which can be accessed using built-in **threadIdx** variable.<br>
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

The above code output the same as the serial code as long as **n < 1024**.

### **Thread Hierarchy**

**threadIdx** has three components.
1. threadIdx.x
2. threadIdx.y
3. threadIdx.z

**Block**: Threads are organized into a 3-dimensional structure called **Block**.

Each thread can be identified in one, two or three dimension in a **thread block**.
* **one-dimensional block**:  threadIdx.x
* **two-dimensional block**:  threadIdx.x + threadIdx.y * blockDim.x
* **three-dimensional block**: threadIdx.x + threadIdx.y * blockIdx.x + threadIdx.z * blockDim.x * blockDim.y

The above equation applies for a single block. It gets complicated when there are multiple blocks.

**Grid**: Blocks are organized into 3-dimensional structure called **grid**.

![](./Pictures/Block%20structure.png)

The number and dimension of threads in block and blocks in grid are represented using **_int_** or **_dim3_** type inside <<<...>>>.

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

Thus each threads act on each matrix element.<br>Thread blocks are scheduled independently. This ensures **_scalability_**<br>
Threads inside a block cooperate and share data through _shared memory_ and synchronize using _barrier synchronization_ using **__synchthreads()** intrinsic function. 

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

**Kernels** are written using the **CUDA instruction set architecture** called **PTX(Parallel Thread eXecution)**. But it is effective to use a high language like C/C++/python/fortran. In both cases it must be compiled into a binary.

**nvcc** or nvidia CUDA C compiler is a compiler driver compiles C or PTX to native instruction set that hardware supports.

### **Compilation workflow**: 

### Offline Compilation:

* Source file = Host code + Device code.
* Device code is compiled into an assembly form (PTX) and binary form(cubin object).
* host code is modified where **<<<>>>** is replaced with necessary **CUDA C runtime function call to load and launch from PTX code or cubin object**
* The host code is left a C code that is left to be compiled using another tool or object that is compiled by nvcc by invoking the host compiler.
* Application can either link to the complied host code or ignore the modified host code and use the CUDA driver API to load and execute PTX code or cubin object.

### Just in Time Compilation:

**PTX** code is loaded by the application at runtime and is complied to **binary** by the **device driver**. This process is called **Just-in-time compilation**.<br>

This allows latest instruction or feature to be used even before the new features are available.<br>

The device driver caches the compiled binary code so as to decrease the application running time and it is invalidated when device driver is upgraded.<br>

This affects the application load time.

### Binary Compatibility:

Binary code is architecture specific. cubin object is generated using the compiler option **-code** that specifies the targeted architecture. e.g. **-code=sm_35** produces binary code for devices with **compute capability 3.5**.<br>

A cubin object generated for **X.y** will one execute on devices of compute capability **X.z** where **z>=y**.

Similarly **PTX compatibility** can be add using **"-arch="** flag. PTX produces for some specific compute capability can always be complied to binary code of greater or equal compute capability.



</span>
 