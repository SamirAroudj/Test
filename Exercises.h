#pragma once

#include <vector>


class Exercises
{
public:
    Exercises();

    void testClosestPoints(
        const std::size_t pointCount = 42,
        const std::size_t k = 10,
        const int seed = 0);

    void testJaccard();

    void testReservoirSampling(
        const int minX = 22,
        const int maxX = 54321,
        const std::size_t sizeOfX = 42,
        const std::size_t sampleSize = 8);

    void testPrecisionRecallCurve(
        const std::size_t thresholdCount = 11u,
        const std::size_t estimateCount = 52u,
        const std::size_t seed = 0u);

private:
};

