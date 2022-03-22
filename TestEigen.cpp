#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <iostream>
#include "TestEigen.h"

using namespace Eigen;
using namespace std;

TestEigen::TestEigen()
{
    auto m = Matrix4f::Identity();
    //m *= 2.0f;
    cout << "m:\n" << m << endl;
    cout << m.rows() << "x" << m.cols() << endl;
    cout << "m(1, 1): " << m(1, 1) << endl;
    Matrix4f m3 = m.array() + 3.0f;
    m3(2, 2) = 2.0f;
    cout << m3 << endl;
    cout << m3.maxCoeff() << endl;
    cout << m3.rowwise().maxCoeff() << endl;

    Matrix3f m42;
    m42.Zero();
    m42 << 42.0f, 0.0f, 0.0f,
           0.00f, 1.0f, 0.0f,
           0.00f, 0.0f, 1.0f;
    cout << "m42:\n" << m42 << endl;


    Vector4f v(0.0f, 0.0f, 1.0f, 1.0f);
    cout << v.data() << endl;
    cout << v.data()[0] << endl;
    cout << v.size() << endl;
    v.y() = 4.0f;
    v(0) = 2.0f;
    v.coeffRef(3) = 3.5f;
    cout << v.x() << endl;
    cout << v.y() << endl;
    cout << v.z() << endl;
    cout << v.w() << endl;
    cout << v.minCoeff() << endl;

    cout << "42 v: " << Vector3f::Constant(42.0f) << endl;
    cout << "Unit(1), UnitX()\n" << Vector3f::Unit(1) << "\n" << Vector3f::UnitX() << endl;

    auto v2 = m * v;
    cout << "v2:\n" << v2 << endl;
    cout << "v2(3): " << v2(3) << endl;
    cout << "v2.norm: " << v2.norm() << endl;
    cout << "v2.squaredNorm: " << v2.squaredNorm() << endl;
    cout << "v2.normalized:\n" << v2.normalized() << endl;

    // the end of playing with Eigen
    cout << "\n\n\n\n" << endl;
    cout << "Finished TestEigen" << endl;
}
