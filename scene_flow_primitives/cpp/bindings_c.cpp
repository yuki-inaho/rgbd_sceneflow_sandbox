#include "types.hpp"
#include "solver.hpp"
#include <memory>
#include <string>
#if defined(_WIN32)
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif
namespace {
thread_local std::string error;
struct Handle { sfr::Result result;std::vector<double> stats; };
sfr::Config config(const double* p,int size){
    if(size!=46)throw std::invalid_argument("configuration ABI expects 46 values");
    for(int j=0;j<size;++j)if(!std::isfinite(p[j]))throw std::invalid_argument("non-finite configuration value");
    // Refuse fractional integer fields and non-Boolean flag values at the ABI boundary.
    for(int j:{20,21,22,23,24,37,38,39,40})
        if(std::floor(p[j])!=p[j]||std::abs(p[j])>1000000)throw std::invalid_argument("invalid integer configuration");
    for(int j:{25,26,27,28,29,30,31,32,33,34,35,36,44})
        if(p[j]!=0&&p[j]!=1)throw std::invalid_argument("flags must be zero or one");
    sfr::Config c;int i=0;
#define SET(field) c.field=p[i++]
    SET(scale);SET(sigma_depth);SET(sigma_flow);SET(sigma_image);SET(sigma_geometry);
    SET(max_motion);SET(trust_sigma);SET(min_depth);SET(max_depth);SET(edge_threshold);
    SET(depth_alpha1);SET(depth_alpha0);SET(flow_alpha1);SET(flow_alpha0);
    SET(image_weight);SET(flow_weight);SET(flow_truncation);SET(photo_reject);SET(huber_k);SET(ridge);
    SET(iterations);SET(outer_iterations);SET(candidate_count);SET(max_search);SET(patch_radius);
    SET(edge_aware);SET(use_tgv);SET(enforce_bound);
    SET(centered_photo);SET(reciprocal_check);SET(diverse_candidates);
    SET(fast_match);SET(affine_photo);SET(rigid_prior);SET(fused_solver);SET(warm_start);SET(keep_identity);
    SET(later_iterations);SET(coarse_stride);SET(seed_count);SET(rigid_min_points);
    SET(rigid_sigma);SET(rigid_huber);SET(step_balance);SET(geometry_huber);SET(geometry_k);
#undef SET
    // ABI v3: formerly reserved slots now have named boolean semantics.
    for(;i<size;++i)if(p[i]!=0)throw std::invalid_argument("nonzero reserved configuration slot");
    c.validate();return c;
}
}
API const char* sfr_error(){return error.c_str();}
API void* sfr_solve(int h,int w,const double* rgb0,const double* rgb1,
    const double* d0,const double* d1,const double* f,const double* intr,const double* cfg,int size){
    try{
        error.clear();const auto c=config(cfg,size);
        if(h<2||w<2)throw std::invalid_argument("H and W must be >=2");
        if(!rgb0||!rgb1||!d0||!d1||!f||!intr)throw std::invalid_argument("null input pointer");
        for(int i=0;i<4;++i)if(!std::isfinite(intr[i]))throw std::invalid_argument("non-finite intrinsics");
        if(intr[0]<=0||intr[1]<=0)throw std::invalid_argument("focal lengths must be positive");
        const int n=h*w;
        for(int i=0;i<3*n;++i)for(double v:{rgb0[i],rgb1[i]})
            if(!std::isfinite(v)||v<0||v>1)throw std::invalid_argument("RGB values must be finite in [0,1]");
        for(int i=0;i<n;++i)for(double z:{d0[i],d1[i]})
            if(sfr::valid_depth(z)&&(z<c.min_depth||z>c.max_depth))throw std::invalid_argument("depth outside configured range");
        sfr::Input in;in.h=h;in.w=w;in.camera={intr[0],intr[1],intr[2],intr[3]};
        in.rgb0={rgb0,rgb0+3*n};in.rgb1={rgb1,rgb1+3*n};
        in.photo_channels=1;
        for(int j=0;j<n;++j)if(rgb0[3*j]!=rgb0[3*j+1]||rgb0[3*j]!=rgb0[3*j+2]||
            rgb1[3*j]!=rgb1[3*j+1]||rgb1[3*j]!=rgb1[3*j+2]){in.photo_channels=3;break;}
        in.depth0={d0,d0+n};in.depth1={d1,d1+n};in.flow={f,f+2*n};
        auto ptr=std::make_unique<Handle>();ptr->result=sfr::solve(in,c);const auto&r=ptr->result;
        ptr->stats={r.elapsed_seconds,r.matching_seconds,r.adjoint_error,r.fixed_point_primal,
            r.fixed_point_dual,r.operator_norm_bound,static_cast<double>(r.candidate_changes),
            static_cast<double>(r.candidate_total),static_cast<double>(r.proposed),static_cast<double>(r.rejected_geometry),
            r.prior_seconds,r.optimization_seconds,static_cast<double>(r.rigid_components),
            static_cast<double>(r.rigid_accepted),static_cast<double>(r.inner_iterations)};
        return ptr.release();
    }catch(const std::exception&e){error=e.what();return nullptr;}
    catch(...){error="unknown C++ exception";return nullptr;}
}
API const double* sfr_array(const void*handle,int field,int*size){
    if(!handle||!size)return nullptr;const auto&h=*static_cast<const Handle*>(handle);
    const auto&r=h.result;const std::vector<double>*v=nullptr;
    switch(field){case 0:v=&r.depth0;break;case 1:v=&r.depth1;break;case 2:v=&r.scene_flow;break;
        case 3:v=&r.flow;break;case 4:v=&r.match_flow;break;case 5:v=&r.initial_scene_flow;break;
        case 6:v=&r.support;break;case 7:v=&r.geometry_residual;break;case 8:v=&r.history;break;
        case 9:v=&h.stats;break;case 10:v=&r.rigid_scene;break;case 11:v=&r.rigid_weight;break;case 12:v=&r.components;break;default:return nullptr;}
    *size=static_cast<int>(v->size());return v->data();
}
API void sfr_free(void*handle){delete static_cast<Handle*>(handle);}
API double sfr_huber_prox(double a,double y,double eta,double k,double tau){return sfr::huber_prox(a,y,eta,k,tau);}
API double sfr_ray_distance(const double*a,const double*b,double l0,double u0,double l1,double u1){
    return sfr::ray_segment_distance({a[0],a[1],a[2]},{b[0],b[1],b[2]},l0,u0,l1,u1);
}

API double sfr_regularizer_energy(int h,int w,int channels,const double*u,const double*aux,
    const double*guide,double threshold,double a1,double a0,int cut){
    try{return sfr::regularizer_energy(h,w,channels,u,aux,guide,threshold,a1,a0,cut!=0);}
    catch(const std::exception&e){error=e.what();return std::numeric_limits<double>::quiet_NaN();}
}

// Diagnostic entry points exercise the same kernels as the estimator.
API int sfr_weighted_alignment(int n,const double*x,const double*y,const double*w,double*out){
    try{if(n<3||!x||!y||!w||!out)throw std::invalid_argument("invalid alignment inputs");
        double sum=0;for(int i=0;i<n;++i){if(!std::isfinite(w[i])||w[i]<0)throw std::invalid_argument("invalid alignment weight");sum+=w[i];
            for(int c=0;c<3;++c)if(!std::isfinite(x[3*i+c])||!std::isfinite(y[3*i+c]))throw std::invalid_argument("nonfinite alignment coordinate");}
        if(sum<=0)throw std::invalid_argument("zero total alignment weight");
        sfr::weighted_alignment(n,x,y,w,out);return 0;
    }catch(const std::exception&e){error=e.what();return -1;}
}
API int sfr_probe_dual(int kind,const double*input,double weight,double sigma,double cap,double*out){
    if(!input||!out||weight<=0||sigma<=0||(kind==2&&cap<=0))return -1;
    sfr::Operator op(3);for(int i=0;i<3;++i)op.row({{i,1}});
    op.group(0,3,weight,kind==2?sfr::Penalty::Huber:(kind==1?sfr::Penalty::Quadratic:sfr::Penalty::Norm),cap);
    op.groups[0].sigma=sigma;std::vector<double>y(input,input+3);op.dual_prox(y);std::copy(y.begin(),y.end(),out);return 0;
}
