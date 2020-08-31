#include <iostream>
#include "TestCuda.h"
#include <iostream>

int main(
    int argc,
    char **argv)
{
    int dummy = 42;
    ++dummy;

    TestCuda testCuda;

    std::cout << "The end." << std::endl;
    return 0;
}
