#pragma once
#include "types.hpp"
#include <initializer_list>
#include <utility>
namespace sfr {
enum class Penalty { Norm, Quadratic, Huber };
struct Group {int start=0,size=0;double weight=0,sigma=0,cap=0;Penalty type=Penalty::Norm;};
// CSR operator. The transpose is an exact scatter of the same stored entries.
class Operator {
public:
    explicit Operator(int n):columns(n),row_ptr{0}{}
    int columns;
    std::vector<int> row_ptr,col;
    std::vector<double> value,tau;
    std::vector<Group> groups;
    int rows()const{return static_cast<int>(row_ptr.size())-1;}
    void row(const std::vector<std::pair<int,double>>& entries);
    void group(int start,int size,double weight,Penalty type,double cap=0);
    void apply(const std::vector<double>&x,std::vector<double>&y)const;
    void transpose(const std::vector<double>&y,std::vector<double>&x)const;
    void prepare_steps(int n_pixels);
    void dual_prox(std::vector<double>&y)const;
    void fused_dual(const std::vector<double>&xb,std::vector<double>&y,std::vector<double>&kty)const;
    double energy(const std::vector<double>&kx)const;
    double adjoint_test()const;
};
} // namespace sfr
