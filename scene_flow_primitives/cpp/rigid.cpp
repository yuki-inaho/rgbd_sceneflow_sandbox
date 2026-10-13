#include "types.hpp"
#include <numeric>

namespace sfr {
namespace {
using M3=std::array<double,9>;
V3 multiply(const M3&R,const V3&x){V3 y{};for(int i=0;i<3;++i)for(int j=0;j<3;++j)y[i]+=R[3*i+j]*x[j];return y;}
struct Pose {M3 R{1,0,0,0,1,0,0,0,1};V3 t{};};
// Horn's unit-quaternion least-squares solution. Scale is fixed to one.
Pose weighted_pose(const std::vector<V3>&x,const std::vector<V3>&y,const std::vector<double>&w){
    double total=0;V3 cx{},cy{};
    for(std::size_t i=0;i<x.size();++i){total+=w[i];for(int c=0;c<3;++c){cx[c]+=w[i]*x[i][c];cy[c]+=w[i]*y[i][c];}}
    Pose pose;if(total<=1e-12)return pose;
    for(int c=0;c<3;++c){cx[c]/=total;cy[c]/=total;}
    M3 S{};
    for(std::size_t i=0;i<x.size();++i)for(int a=0;a<3;++a)for(int b=0;b<3;++b)
        S[3*a+b]+=w[i]*(x[i][a]-cx[a])*(y[i][b]-cy[b]);
    const double tr=S[0]+S[4]+S[8];
    double A[4][4]={{tr,S[5]-S[7],S[6]-S[2],S[1]-S[3]},
        {S[5]-S[7],S[0]-S[4]-S[8],S[1]+S[3],S[2]+S[6]},
        {S[6]-S[2],S[1]+S[3],-S[0]+S[4]-S[8],S[5]+S[7]},
        {S[1]-S[3],S[2]+S[6],S[5]+S[7],-S[0]-S[4]+S[8]}};
    double V[4][4]{};for(int i=0;i<4;++i)V[i][i]=1;
    // Symmetric Jacobi, largest ALGEBRAIC eigenvalue (not largest magnitude).
    for(int sweep=0;sweep<64;++sweep){int p=0,q=1;double largest=0;
        for(int i=0;i<4;++i)for(int j=i+1;j<4;++j)if(std::abs(A[i][j])>largest){p=i;q=j;largest=std::abs(A[i][j]);}
        if(largest<1e-14*std::max(1.0,std::abs(tr)))break;
        const double angle=.5*std::atan2(2*A[p][q],A[q][q]-A[p][p]);
        const double c=std::cos(angle),s=std::sin(angle);
        const double app=A[p][p],aqq=A[q][q],apq=A[p][q];
        A[p][p]=c*c*app-2*s*c*apq+s*s*aqq;
        A[q][q]=s*s*app+2*s*c*apq+c*c*aqq;A[p][q]=A[q][p]=0;
        for(int k=0;k<4;++k)if(k!=p&&k!=q){const double a=A[k][p],b=A[k][q];
            A[k][p]=A[p][k]=c*a-s*b;A[k][q]=A[q][k]=s*a+c*b;}
        for(int k=0;k<4;++k){const double a=V[k][p],b=V[k][q];V[k][p]=c*a-s*b;V[k][q]=s*a+c*b;}
    }
    int best=0;for(int i=1;i<4;++i)if(A[i][i]>A[best][best])best=i;
    const double a=V[0][best],b=V[1][best],c=V[2][best],d=V[3][best];
    pose.R={a*a+b*b-c*c-d*d,2*(b*c-a*d),2*(b*d+a*c),
            2*(b*c+a*d),a*a-b*b+c*c-d*d,2*(c*d-a*b),
            2*(b*d-a*c),2*(c*d+a*b),a*a-b*b-c*c+d*d};
    const auto rx=multiply(pose.R,cx);for(int j=0;j<3;++j)pose.t[j]=cy[j]-rx[j];return pose;
}
}

void weighted_alignment(int n,const double*x,const double*y,const double*w,double*out){
    std::vector<V3>a(n),b(n);std::vector<double>weights(w,w+n);
    for(int i=0;i<n;++i)for(int j=0;j<3;++j){a[i][j]=x[3*i+j];b[i][j]=y[3*i+j];}
    const auto pose=weighted_pose(a,b,weights);
    std::copy(pose.R.begin(),pose.R.end(),out);std::copy(pose.t.begin(),pose.t.end(),out+9);
}

RigidPrior fit_rigid_prior(const Input&in,const Config&cfg,const Matches&m){
    const int n=in.n();RigidPrior out;out.scene.assign(3*n,0);out.weight.assign(n,0);out.component.assign(n,-1);
    if(!cfg.rigid_prior)return out;
    std::vector<int> labels(n,-1);std::vector<std::vector<int>> components;
    for(int seed=0;seed<n;++seed){if(labels[seed]>=0||!valid_depth(in.depth0[seed]))continue;
        const int label=static_cast<int>(components.size());std::vector<int> pixels{seed};labels[seed]=label;
        for(std::size_t k=0;k<pixels.size();++k){const int p=pixels[k];
            for(int r:{p-1,p+1,p-in.w,p+in.w}){
                if(r<0||r>=n||((r==p-1||r==p+1)&&r/in.w!=p/in.w)||labels[r]>=0||!valid_depth(in.depth0[r]))continue;
                if(std::abs(in.depth0[r]-in.depth0[p])>cfg.edge_threshold)continue;
                labels[r]=label;pixels.push_back(r);
            }
        }
        components.push_back(std::move(pixels));
    }
    out.components=static_cast<int>(components.size());
    for(int p=0;p<n;++p)out.component[p]=labels[p];
    for(const auto&pixels:components){
        std::vector<V3>x,y;std::vector<double>base,w;
        for(int p:pixels)if(m.active[p]>0){const auto&c=m.candidates[p][0];
            V3 a=in.camera.ray(p%in.w,p/in.w),b=c.ray;
            const double z=sample(in.depth1,c);
            for(int j=0;j<3;++j){a[j]*=in.depth0[p];b[j]*=z;}
            x.push_back(a);y.push_back(b);base.push_back(1/(1+c.photo));
        }
        if(static_cast<int>(x.size())<cfg.rigid_min_points||x.size()<.1*pixels.size())continue;
        // Reject almost collinear/point-like support via a covariance minor sum.
        V3 center{};for(const auto&a:x)for(int j=0;j<3;++j)center[j]+=a[j]/x.size();M3 scatter{};
        for(const auto&a:x)for(int j=0;j<3;++j)for(int k=0;k<3;++k)scatter[3*j+k]+=(a[j]-center[j])*(a[k]-center[k])/x.size();
        const double minors=scatter[0]*scatter[4]-scatter[1]*scatter[3]+scatter[0]*scatter[8]-scatter[2]*scatter[6]+scatter[4]*scatter[8]-scatter[5]*scatter[7];
        if(minors<1e-10)continue;
        w=base;Pose pose;
        std::vector<double> residual(x.size());
        for(int it=0;it<8;++it){pose=weighted_pose(x,y,w);
            for(std::size_t k=0;k<x.size();++k){V3 r=multiply(pose.R,x[k]);for(int j=0;j<3;++j)r[j]+=pose.t[j]-y[k][j];
                residual[k]=norm(r);w[k]=base[k]*std::min(1.,cfg.rigid_huber/std::max(1e-12,residual[k]));}
        }
        auto copy=residual;std::nth_element(copy.begin(),copy.begin()+copy.size()/2,copy.end());const double median=copy[copy.size()/2];
        if(!std::isfinite(median)||median>1.5*cfg.rigid_huber)continue;
        const double confidence=std::min(1.,cfg.rigid_huber*cfg.rigid_huber/std::max(1e-12,median*median));
        ++out.accepted;
        for(int p:pixels){V3 a=in.camera.ray(p%in.w,p/in.w);for(double&v:a)v*=in.depth0[p];
            V3 b=multiply(pose.R,a);for(int j=0;j<3;++j)out.scene[3*p+j]=b[j]+pose.t[j]-a[j];out.weight[p]=confidence;}
    }
    return out;
}
} // namespace sfr
