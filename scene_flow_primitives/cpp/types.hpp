#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace sfr {
using V3 = std::array<double, 3>;
inline double dot(const V3& a, const V3& b) {
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}
inline double norm(const V3& a) { return std::sqrt(dot(a,a)); }
inline double huber(double x, double k) {
    x=std::abs(x); return x<=k ? 0.5*x*x : k*(x-0.5*k);
}
inline double huber_prox(double a, double y, double eta, double k, double tau) {
    const double r=a-y;
    const double threshold=k*(eta+tau/eta);
    const double v=std::abs(r)<=threshold ? r/(1.0+tau/(eta*eta))
        : r-std::copysign(tau*k/eta,r);
    return y+v;
}
struct Camera {
    double fx=140, fy=140, cx=63.5, cy=47.5;
    V3 ray(double x, double y) const { return {(x-cx)/fx,(y-cy)/fy,1.0}; }
};
struct Config {
    double scale=0.01;
    double sigma_depth=0.003, sigma_flow=0.7, sigma_image=0.025;
    double sigma_geometry=0.0008;
    double max_motion=0.05, trust_sigma=4.0;
    double min_depth=0.10, max_depth=5.0;
    double edge_threshold=0.06;
    double depth_alpha1=1.2, depth_alpha0=4.0;
    double flow_alpha1=2.0, flow_alpha0=6.0;
    double image_weight=1.0, flow_weight=0.025, flow_truncation=9.0;
    double photo_reject=3.0, huber_k=3.0, ridge=1e-6;
    int iterations=400, outer_iterations=3, candidate_count=24;
    int max_search=18, patch_radius=2;
    bool edge_aware=true, use_tgv=true, enforce_bound=true;
    bool centered_photo=true, reciprocal_check=true, diverse_candidates=true;
    // v0.3: independently switchable changes for matched-input ablations.
    bool fast_match=true, affine_photo=false, rigid_prior=true;
    bool fused_solver=true, warm_start=true, keep_identity=true;
    int later_iterations=150, coarse_stride=2, seed_count=3, rigid_min_points=24;
    double rigid_sigma=0.0015, rigid_huber=0.006, step_balance=1.0;
    bool geometry_huber=true;
    double geometry_k=3.0;
    void validate() const;
};
struct Input {
    int h=0,w=0,photo_channels=3;
    Camera camera;
    std::vector<double> rgb0,rgb1,depth0,depth1,flow;
    int n() const { return h*w; }
};
struct Candidate {
    double x=0,y=0,cost=0,photo=0;
    std::array<int,4> index{};
    std::array<double,4> weight{};
    V3 ray{};
};
struct Matches {
    std::vector<std::vector<Candidate>> candidates;
    std::vector<int> selected;
    std::vector<double> active;
    long long proposed=0, rejected_geometry=0;
};
struct RigidPrior {
    std::vector<double> scene, weight, component;
    int components=0, accepted=0;
};
void weighted_alignment(int n,const double*x,const double*y,const double*w,double*out);
RigidPrior fit_rigid_prior(const Input&, const Config&, const Matches&);
struct Result {
    int h=0,w=0;
    std::vector<double> depth0,depth1,scene_flow,flow,match_flow,initial_scene_flow;
    std::vector<double> support,geometry_residual,history;
    std::vector<double> rigid_scene,rigid_weight,components;
    // history rows: outer iteration, inner iteration, current objective, best objective
    double elapsed_seconds=0, matching_seconds=0, adjoint_error=0;
    double fixed_point_primal=0, fixed_point_dual=0, operator_norm_bound=0.95;
    int candidate_changes=0, candidate_total=0;
    double prior_seconds=0, optimization_seconds=0;
    int rigid_components=0, rigid_accepted=0, inner_iterations=0;
    long long proposed=0,rejected_geometry=0;
};
inline bool valid_depth(double x) { return std::isfinite(x) && x>0; }
inline bool stencil(double x,double y,int h,int w,
                    std::array<int,4>& idx,std::array<double,4>& a) {
    if(!std::isfinite(x)||!std::isfinite(y)||x<0||y<0||x>w-1||y>h-1) return false;
    // At the closed image boundary use the last interior cell with endpoint weights.
    const int ix=std::min(w-2,static_cast<int>(std::floor(x)));
    const int iy=std::min(h-2,static_cast<int>(std::floor(y)));
    const double dx=x-ix,dy=y-iy;
    idx={iy*w+ix,iy*w+ix+1,(iy+1)*w+ix,(iy+1)*w+ix+1};
    a={(1-dx)*(1-dy),dx*(1-dy),(1-dx)*dy,dx*dy};
    return true;
}
inline double sample(const std::vector<double>& a,const Candidate& c,int offset=0) {
    double v=0;for(int k=0;k<4;++k)if(c.weight[k]>1e-14)v+=c.weight[k]*a[offset+c.index[k]];return v;
}
// Exact distance between two closed ray segments, both measured from the camera.
inline double ray_segment_distance(const V3& r0,const V3& r1,
    double l0,double u0,double l1,double u1) {
    const double A=dot(r0,r0), B=dot(r0,r1), C=dot(r1,r1);
    double best=std::numeric_limits<double>::infinity();
    auto update=[&](double a,double b) {
        V3 v{};for(int c=0;c<3;++c)v[c]=b*r1[c]-a*r0[c];
        best=std::min(best,dot(v,v));
    };
    for(double a:{l0,u0})update(a,std::clamp(a*B/C,l1,u1));
    for(double b:{l1,u1})update(std::clamp(b*B/A,l0,u0),b);
    return std::sqrt(std::max(0.0,best));
}
Matches make_matches(const Input&, const Config&);
Result solve(const Input&, const Config&);
double regularizer_energy(int h,int w,int channels,const double* u,const double* aux,
    const double* guide,double edge_threshold,double alpha1,double alpha0,bool cut);
} // namespace sfr
