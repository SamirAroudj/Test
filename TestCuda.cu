#include <cuda_runtime_api.h>
#include <cub/cub.cuh>
#include <iostream>
#include <limits>
#include <vector>
#include "TestCuda.h"

using namespace std;

using Real = float;
using uint32 = unsigned int;

constexpr uint32 uint32Max = std::numeric_limits<uint32>::max();
constexpr Real EPSILON = 1.0f * 10e-6f;

#define CUDA_CHECK(result)\
{\
    check((result), __FILE__, __LINE__);\
}

#define CUDA_CHECK_LAST_ERROR() CUDA_CHECK(cudaGetLastError())

void check(
    cudaError_t errorCode,
    const char *sourceFile = nullptr,
    const uint32 line = uint32Max);

void checkLastCudaErrorSynchronized();

const Real *getHostData(
    Real &maximum,
    Real &sum,
    const uint32 pixelCount);

void testVersion();

TestCuda::TestCuda()
{
    testVersion();

    constexpr uint32 width = 256;
    constexpr uint32 height = 256;
    constexpr uint32 pixelCount = (width * height);
    Real *testData = nullptr;
    Real *deviceParams = nullptr;
    cudaStream_t streamIdx = nullptr;

    // test constexpr if - requires at least C++17
    constexpr uint32 cppTest = 42;
    if constexpr (cppTest == 42)
        cout << "Using C++ standard 17 or newer.\n" << endl;

    CUDA_CHECK(cudaMalloc((void **) &testData, sizeof(Real) * pixelCount));
    CUDA_CHECK(cudaMalloc((void **) &deviceParams, sizeof(Real) * 2));

    // create & upload test data
    Real hostMax;
    Real hostSum;
    const Real *hostData = getHostData(hostMax, hostSum, pixelCount);
    CUDA_CHECK(cudaMemcpy(testData, hostData, pixelCount * sizeof(Real), cudaMemcpyHostToDevice));


    // sum reduction
    {
        Real *deviceSum = (deviceParams + 1);
        size_t tempStorageSize = 0;
        void *tempStorage = nullptr;

        // get sum: 1) how much memory is required?
        cub::DeviceReduce::Sum<Real *, Real *>(tempStorage, tempStorageSize,
        testData, deviceSum, pixelCount, streamIdx);
        checkLastCudaErrorSynchronized();

        // get sum 2) allocate necessary memory
        cout << "Required temp storage: " << tempStorageSize << " bytes." << endl;
        cudaMalloc(&tempStorage, tempStorageSize);
        checkLastCudaErrorSynchronized();

        // get sum 3) do the actual work
        cout << "Finding sum on device." << endl;
        cub::DeviceReduce::Sum<Real *, Real *>(tempStorage, tempStorageSize,
            testData, deviceSum, pixelCount, streamIdx);
        checkLastCudaErrorSynchronized();

        // free memory
        CUDA_CHECK(cudaFree(tempStorage));
        checkLastCudaErrorSynchronized();
    }

    // max reduction
    {
        Real *deviceMax = deviceParams;
        size_t tempStorageSize = 0;
        void *tempStorage = nullptr;

        // get maximum: 1) how much memory is required?
        cub::DeviceReduce::Max<Real *, Real *>(tempStorage, tempStorageSize,
        testData, deviceMax, pixelCount, streamIdx);
        checkLastCudaErrorSynchronized();

        // get maximum 2) allocate necessary memory
        cout << "Required temp storage: " << tempStorageSize << " bytes." << endl;
        cudaMalloc(&tempStorage, tempStorageSize);
        checkLastCudaErrorSynchronized();

        // get maximum 3) do the actual work
        cout << "Finding maximum on device." << endl;
        cub::DeviceReduce::Max<Real *, Real *>(tempStorage, tempStorageSize,
        testData, deviceMax, pixelCount, streamIdx);
        checkLastCudaErrorSynchronized();

        // free memory
        CUDA_CHECK(cudaFree(tempStorage));
        checkLastCudaErrorSynchronized();
    }

    // download results
    cout << "Downloading results from device to host." << endl;
    Real result[2];
    CUDA_CHECK(cudaMemcpy(&result, deviceParams, 2 * sizeof(Real), cudaMemcpyDeviceToHost));
    Real &deviceMax = result[0];
    Real &deviceSum = result[1];

    // free memory
    cout << "Freeing device results memory." << endl;
    CUDA_CHECK(cudaFree(deviceParams));
    deviceParams = nullptr;

    cout << "Freeing device test data memory." << endl;
    CUDA_CHECK(cudaFree(testData));
    testData = nullptr;
    checkLastCudaErrorSynchronized();

    cout << "Device Sum: " << deviceSum << "\n";
    cout << "Host sum: " << hostSum << "\n";
    cout << "Device max: " << deviceMax << "\n";
    cout << "Host max: " << hostMax << "\n";

    const Real tmp = (std::max(std::abs(hostSum), std::abs(deviceSum)));
    if (std::abs(hostSum - deviceSum) / tmp > EPSILON)
    {
        cerr << "Invalid device sum!" << endl;
        throw std::exception();
    }

    if (std::abs(hostMax - deviceMax) > EPSILON)
    {
        cerr << "Invalid device maximum!" << endl;
        throw std::exception();
    }

    cout << endl;
}

void checkLastCudaErrorSynchronized()
{
    #ifdef TEST_DEBUG
        cudaDeviceSynchronize();
        CUDA_CHECK_LAST_ERROR();
    #endif // TEST_DEBUG
}

void check(
    cudaError_t errorCode,
    const char *sourceFile,
    const unsigned int line)
{
    if (cudaSuccess == errorCode)
        return;

    {
        cerr << "Failed Cuda function!\n";

        // where did it happen?
        if (sourceFile)
            cerr << "Source file: " << sourceFile << "\n";
        if (uint32Max != line)
            cerr << "Line: " << line << "\n";

        // cuda error name & description
        const string errorName(cudaGetErrorName(errorCode));
        const string errorDescription(cudaGetErrorString(errorCode));

        cerr << "Failed Cuda function!\n";
        cerr << errorName << ":\n";
        cerr << errorDescription << "\n";

        // end the output
        cerr << flush;

        // end the program
        throw std::exception();
    }
}

const Real *getHostData(
    Real &maximum,
    Real &sum,
    const uint32 pixelCount)
{
    static vector<Real> hostVector(pixelCount);
    maximum = std::numeric_limits<Real>::lowest();
    sum = Real(0.0f);

    for (uint32 i = 0; i < pixelCount; ++i)
    {
        const Real ele = 42.0f + (20 * i) - 0.5f * ((i + 3) * (i + 3));
        hostVector[i] = ele;
        if (ele > maximum)
            maximum = ele;
        sum += ele;
    }

    const Real *hostData = hostVector.data();
    return hostData;
}

void testVersion()
{
    int driverVersion;
    CUDA_CHECK(cudaDriverGetVersion(&driverVersion));
    cout << "Cuda driver version: " << driverVersion << endl;

    int runtimeVersion;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVersion));
    cout << "Cuda runtime version: " << runtimeVersion << endl;
}
