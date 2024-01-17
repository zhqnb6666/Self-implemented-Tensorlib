#include "tensor.h"
#include "tensor.cpp"
#include <iostream>
#include "tensor_math.h"
#include "tensor_math.cpp"
#include "tensor_bool.h"
using namespace std;

void test_math(){
    std::cout<< "Math test:"<<std::endl;


    // std::cout<< "3d test: " << std::endl;
    Tensor<int> add1({3, 2}, {1, 2, 3, 4, 5, 6}); // 2D matrix
    Tensor<int> add2({3, 2}, {7, 8, 9, 10, 11, 12}); // 2D matrix
    //addition test
    cout << "add test" << endl;
    Tensor<int> add3 = add1 + add2;
    cout << "add3: \n"; 
    add3.print();

    Tensor<int> add4 = add(add1, add2);
    cout << "add4: \n"; 
    add4.print();

    Tensor<int> add5 = add1.add(add2);
    cout << "add5: \n"; 
    add5.print();

    Tensor<int> add6 = add(add1, 1);
    cout << "add6: \n"; 
    add6.print();

    Tensor<int> add7 = add1.add((int)1.0);
    cout << "add7: \n";
    add7.print();

    Tensor<int> add8 = add1 + 1;
    cout << "add8: \n";
    add8.print();



    // //substraction test
    cout << "sub test" << endl;
    Tensor<int> sub1({3, 2}, {1, 2, 3, 4, 5, 6}); // 2D matrix
    Tensor<int> sub2({3, 2}, {7, 8, 9, 10, 11, 12}); // 2D matrix

    Tensor<int> sub3 = sub1 - sub2;
    cout << "sub3: \n"; 
    sub3.print();

    Tensor<int> sub4 = subtract(sub1, sub2); // 假设`subtract`是相应的减法函数
    cout << "sub4: \n"; 
    sub4.print();

    Tensor<int> sub5 = sub1.subtract(sub2); // 假设`Tensor`类有一个`subtract`成员函数
    cout << "sub5: \n"; 
    sub5.print();

    Tensor<int> sub6 = subtract(sub1, 1); // 这里假设`subtract`函数可以接受一个标量作为第二个参数
    cout << "sub6: \n"; 
    sub6.print();

    Tensor<int> sub7 = sub1.subtract((int)1.0); // 强制类型转换以匹配函数的期望参数类型
    cout << "sub7: \n";
    sub7.print();

    Tensor<int> sub8 = sub1 - 1; // 强制类型转换以匹配函数的期望参数类型
    cout << "sub8: \n";
    sub8.print();



    //multiply test
    cout << "mul test" << endl;
    Tensor<int> mul1({3, 2}, {1, 2, 3, 4, 5, 6}); // 2D matrix
    Tensor<int> mul2({2, 3}, {7, 8, 9, 10, 11, 12}); // 2D matrix

    Tensor<int> mul3 = mul1 * mul2;
    // cout << "mul3: \n"; 
    mul3.print();

    Tensor<int> mul4 = multiply(mul1, mul2); // 假设`multiply`是相应的乘法函数
    cout << "mul4: \n"; 
    mul4.print();

    Tensor<int> mul5 = mul1.multiply(mul2); // 假设`Tensor`类有一个`multiply`成员函数
    cout << "mul5: \n"; 
    mul5.print();

    Tensor<int> mul6 = multiply(mul1, 1); // 这里假设`multiply`函数可以接受一个标量作为第二个参数
    cout << "mul6: \n"; 
    mul6.print();

    Tensor<int> mul7 = mul1.multiply((int)1.0); // 强制类型转换以匹配函数的期望参数类型
    cout << "mul7: \n";
    mul7.print();

    Tensor<int> mul8 = mul1 * 2; // 强制类型转换以匹配函数的期望参数类型
    cout << "mul8: \n";
    mul8.print();



    //div test
    cout << "div test" << endl;
    Tensor<double> div1({3, 2}, {1, 2, 3, 4, 5, 6}); // 2D matrix
    Tensor<double> div2({3, 2}, {7, 8, 9, 10, 11, 12}); // 2D matrix

    Tensor<double> div3 = div1 / div2;
    cout << "div3: \n"; 
    div3.print();

    Tensor<double> div4 = divide(div1, div2); // 假设`divide`是相应的除法函数
    cout << "div4: \n"; 
    div4.print();

    Tensor<double> div5 = div1.divide(div2); // 假设`Tensor`类有一个`divide`成员函数
    cout << "div5: \n"; 
    div5.print();

    Tensor<double> div6 = divide(div1, (double)2.0); // 这里假设`divide`函数可以接受一个标量作为第二个参数
    cout << "div6: \n"; 
    div6.print();

    Tensor<double> div7 = div1.divide((double)2.0); // 强制类型转换以匹配函数的期望参数类型
    cout << "div7: \n";
    div7.print();

    Tensor<double> div8 = div1 / 2.0; // 强制类型转换以匹配函数的期望参数类型
    cout << "div8: \n";
    div8.print();



    //log test
    cout << "log test" << endl;
    Tensor<double> log1({3, 2}, {1, 2, 3, 4, 5, 6}); // 2D matrix
    Tensor<double> log2({3, 2}, {7, 8, 9, 10, 11, 12}); // 2D matrix

    Tensor<double> log3 = log(log1);
    cout << "log3: log(log1)\n"; 
    log3.print();

    Tensor<double> log4 = log1.log();
    cout << "log4: log1.log()\n"; 
    log4.print();

    //project introduction examples for addition
    // official example
    // std::vector<double> testArray1 =  {0.1, 1.2, 2.2, 3.1, 4.9, 5.2};
    // std::vector<double> testArray2 =  {0.2, 1.3, 2.3, 3.2, 4.8, 5.1};
    // Tensor<double> t1({3,2},testArray1);
    // Tensor<double> t2({3,2},testArray2);
    // std::cout << t1 + t2 << std::endl << add(t1, 1.0) << std::endl; //是否应该考虑不同数据类型之间的相加
    // Output
    // [[ 0.3000, 2.5000],
    // [ 4.5000, 6.3000],
    // [ 9.7000, 10.3000]]
    // [[ 1.1000, 2.2000],
    // [ 3.2000, 4.1000],
    // [ 5.9000, 6.2000]]

    // //similiar sub test
    // std::cout << t1 - t2 << std::endl << subtract(t1,t2) << std::endl;


    //sum mean max min test 成员函数还未实现
    cout << "sum mean max min test" << endl;
    Tensor<int> test1({3, 2}, {1, 2, 3, 4, 5, 6}); // 2D matrix
    Tensor<int> test2({3, 2}, {7, 8, 9, 10, 11, 12}); // 2D matrix
    cout << "sum " << endl;
    std::cout << sum(test1, 1) << std::endl << test1.sum(1) << std::endl;

    cout << "mean" << endl;
    std::cout << mean(test1, 1) << std::endl << test1.mean(1) << std::endl;

    cout << "max" << endl;
    std::cout << max(test1, 1) << std::endl << test1.max(1) << std::endl;

    cout << "min" << endl;
    std::cout << min(test1, 1) << std::endl << test1.min(1) << std::endl;

    //equal test
    // Tensor<bool> t3 = eq(t1, t2);// This compares t1 and t2 element-wise.
    // Tensor<bool> t4= t1.eq(t2);// Another way to compare t1 and t2 element-wise.
    // Tensor<bool> t5= t1 == t2;

    //comparision test
    // equal official Example 缺少==以及成员函数
    std::vector<double> equalArray1 =  {0.1, 1.2, 2.2, 3.1, 4.9, 5.2};
    std::vector<double> equalArray2 =  {0.2, 1.3, 2.2, 3.2, 4.8, 5.2};
    Tensor<double> eq1 = Tensor<double>({3,2},equalArray1);
    Tensor<double> eq2 = Tensor<double>({3,2},equalArray2);
    cout << "eq ne gt... test" << endl;

    cout << "eq" << endl;
    std::cout << eq(eq1,eq2) << std::endl;
    cout << eq1.eq(eq2) << endl;
    cout << (eq1 == eq2) << endl;

    cout << "not eq" << endl;
    std::cout << ne(eq1,eq2) << std::endl;
    cout << eq1.ne(eq2) << endl;
    cout << (eq1 != eq2) << endl;

    cout << "greater than" << endl;
    cout << gt(eq1,eq2) << endl;// 0 0 0 0 1 0
    cout << eq1.gt(eq2) << endl;
    cout << (eq1 > eq2) << endl;

    cout << "greater than or equal" << endl;
    cout << ge(eq1,eq2) << endl; // 0 0 1 0 1 1
    cout << eq1.ge(eq2) << endl;
    cout << (eq1 >= eq2) << endl;

    cout << "less than " << endl;
    cout << lt(eq1,eq2) << endl; // 1 1 0 1 0 0
    cout << eq1.lt(eq2) << endl;
    cout << (eq1 < eq2) << endl;

    cout << "less than or equal " << endl;
    cout << le(eq1,eq2) << endl; // 1 1 1 1 0 1
    cout << eq1.le(eq2) << endl;
    cout << (eq1 <= eq2) << endl;
}

int main() {
    test_math();
    return 0;
}
