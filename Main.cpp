#include <iostream>
#ifdef BASE_CUDA
    #include "TestCuda.h"
#endif //BASE_CUDA

#include "Exercises.h"
#include "TestEigen.h"


int main(
    [[maybe_unused]] int argc,
    [[maybe_unused]] char **argv)
{
    int dummy = 42;
    ++dummy;

    Exercises exercises;
    //exercises.testJaccard();
    //exercises.testReservoirSampling();
    exercises.testPrecisionRecallCurve();
    exercises.testClosestPoints();
    TestEigen testEigen;

    #ifdef BASE_CUDA
        TestCuda testCuda;
    #endif //BASE_CUDA

    std::cout << "The end." << std::endl;
    return 0;
}
