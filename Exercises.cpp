#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <set>
#include <vector>
#include "Exercises.h"

using namespace std;

Exercises::Exercises()
{

}

class X
{
public:
    X(
        const int min = 1,
        const int max = 420,
        const int n = -1,
        const size_t seed = 0u);

    bool getNext(int &x);
    int getNextI() const;
    bool isEmpty() const;

private:
    std::uniform_int_distribution<> mDistribution;
    std::mt19937 mGenerator;
    int mCurrentCount;
    int mMaxCount;
};

X::X(const int min, const int max, const int n, const std::size_t seed) :
    mDistribution(min, max),
    mGenerator(seed),
    mCurrentCount(0),
    mMaxCount(n)
{

}

int X::getNextI() const
{
    return mCurrentCount;
}

bool X::getNext(int &x)
{
    if (isEmpty())
        return false;

    x = mDistribution(mGenerator);
    ++mCurrentCount;
    return true;
}

bool X::isEmpty() const
{
    return (mMaxCount > 0 && mCurrentCount >= mMaxCount);
}

void reservoirSample(
    vector<int> &sample,
    X &x,
    const size_t sampleSize,
    const size_t seed = 0u)
{
    std::mt19937 generator(seed);
    int xi;

    // fill sample with first sampleSize inputs
    sample.clear();
    sample.reserve(sampleSize);
    while (!x.isEmpty() && sample.size() < sampleSize)
    {
        const int i = x.getNextI();
        const bool okay = x.getNext(xi);
        assert(okay);

        cout << "Received initial x[" << i << "] = " << xi << endl;
        sample.push_back(xi);
    }

    // draw samples and select them randomly
    while (!x.isEmpty())
    {
        const int i = x.getNextI();
        const bool okay = x.getNext(xi);
        assert(okay);

        cout << "Received x[" << i << "] = " << xi << endl;
        std::uniform_int_distribution selectionDist(1, i);
        const auto j = selectionDist(generator);
        std::cout << "Selection: " << j << "/" << i << std::endl;
        if (static_cast<size_t>(j) <= sampleSize)
            sample[j - 1u] = xi;
    }
}


void Exercises::testReservoirSampling(
    const int minX,
    const int maxX,
    const size_t sizeOfX,
    const size_t sampleSize)
{
    //std::random_device randomDevice;
    //const auto seed = randomDevice();
    size_t seedForSampling = 0u;
    size_t seedForX = 1u;
    X x(minX, maxX, sizeOfX, seedForX);

    std::vector<int> sample;
    sample.reserve(99999);

    reservoirSample(sample, x, sampleSize, seedForSampling);
    cout << "Sampled " << sampleSize << " elements from input x:\n";
    for (size_t i = 0; i < sampleSize; ++i)
        cout << "S[" << (i + 1) << "] = " << sample[i] << endl;

    cout << "Finished reservoir sampling computations." << endl;
}

class Point
{
public:
    Point();
    Point(
        const float x,
        const float y);

    float getDistanceTo(
        const Point &p) const;

private:
    float mX;
    float mY;
};

Point::Point() :
    mX(-std::numeric_limits<float>::max()),
    mY(-std::numeric_limits<float>::max())
{

}

Point::Point(const float x, const float y) :
    mX(x),
    mY(y)
{

}

float Point::getDistanceTo(
    const Point &p) const
{
    const auto dX = (p.mX - mX);
    const auto dY = (p.mY - mY);
    return std::sqrt(dX * dX + dY * dY);
}

void Exercises::testClosestPoints(
    const size_t pointCount,
    const size_t k,
    const int seed)
{
    std::uniform_real_distribution dist(-1.0f, 1.0f);
    std::vector<Point> points(pointCount);
    std::mt19937 generator(seed);

    for (size_t pIdx = 0; pIdx < pointCount; ++pIdx)
    {
        const auto x = dist(generator);
        const auto y = dist(generator);
        points[pIdx] = Point(x, y);
    }

    const Point target(0.42f, 0.42f);
    std::vector<Point> closestPoints(k);
    for (size_t pIdx = 0; pIdx < pointCount; ++pIdx)
    {
        //const auto distance = points[pIdx].getDistanceTo(target);
    }
}

void Exercises::testJaccard()
{
    const size_t seed = 0u;
    const size_t pointCount = 42;
    std::mt19937 generator(seed);
    std::uniform_int_distribution dist(0, 100);

    std::vector<int> pointsA(pointCount);
    std::vector<int> pointsB(pointCount);
    for (size_t i = 0; i < pointCount; ++i)
    {
        pointsA[i] = dist(generator);
        pointsB[i] = dist(generator);
    }

    std::set<int> unionAB;
    std::set<int> intersectionAB;

    std::sort(pointsA.begin(), pointsA.end());
    std::sort(pointsB.begin(), pointsB.end());

    while (false)
    {

    }
}

void Exercises::testPrecisionRecallCurves(
    const std::size_t thresholdCount,
    const std::size_t estimateCount,
    const std::size_t seed)
{
    // initialize them using random values
    std::vector<bool> groundTruths(estimateCount);
    std::vector<float> predictions(estimateCount);
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> predictionDist(0.0f, 1.0f);
    for (size_t i = 0; i < estimateCount; ++i)
    {
        const auto randPrediction = predictionDist(generator);
        const auto randGT = (predictionDist(generator) >= 0.5f);
        predictions[i] = randPrediction;
        groundTruths[i] = randGT;
    }


    // compute curves
    const float binSize = 1.0f / (thresholdCount - 1.0f);
    std::vector<float> precisionCurve(thresholdCount);
    std::vector<float> recallCurve(thresholdCount);
    std::vector<size_t> truePositives(thresholdCount, 0u);
    std::vector<size_t> falsePositives(thresholdCount, 0u);
    std::vector<size_t> falseNegatives(thresholdCount, 0u);

    for (size_t i = 0u; i < estimateCount; ++i)
    {
        const auto prediction = predictions[i];
        const auto gt = groundTruths[i];

        const auto k = std::ceil(prediction / binSize);
        if (prediction > 0.0f)
        {
            const auto j = k - 1u;
            if (gt)
                ++truePositives[j];
            else
                ++falsePositives[j];
        }

        if (gt)
            ++falseNegatives[k];

        const auto precision = truePositives / static_cast<float>(truePositives + falsePositives);
        const auto recall = truePositives / static_cast<float>(truePositives + falseNegatives);
    }
}
