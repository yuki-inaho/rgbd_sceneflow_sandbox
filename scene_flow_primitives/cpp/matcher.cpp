#include "types.hpp"
#include <numeric>

namespace sfr {
void Config::validate() const {
    for(double v:{scale,sigma_depth,sigma_flow,sigma_image,sigma_geometry,max_motion,
                  trust_sigma,min_depth,max_depth,edge_threshold,huber_k,photo_reject,ridge})
        if(!std::isfinite(v)||v<=0)throw std::invalid_argument("scales and bounds must be finite and positive");
    for(double v:{depth_alpha1,depth_alpha0,flow_alpha1,flow_alpha0,image_weight,
                  flow_weight,flow_truncation})
        if(!std::isfinite(v)||v<0)throw std::invalid_argument("weights must be finite and nonnegative");
    if(later_iterations<1||coarse_stride<1||coarse_stride>4||seed_count<1||seed_count>8||rigid_min_points<6)
        throw std::invalid_argument("invalid v3 search or rigid settings");
    for(double v:{rigid_sigma,rigid_huber,step_balance,geometry_k})
        if(!std::isfinite(v)||v<=0)throw std::invalid_argument("invalid v3 scales");
    if(max_motion>=min_depth)throw std::invalid_argument("max_motion must be smaller than min_depth to keep projections in front of camera");
    if(min_depth>=max_depth)throw std::invalid_argument("min_depth must be smaller than max_depth");
    if(iterations<1||outer_iterations<1||candidate_count<1||max_search<1||patch_radius<0||patch_radius>8)
        throw std::invalid_argument("invalid iterations/search settings; patch_radius must be in [0,8]");
}

static bool safe_target(const Input& in,const Config& cfg,Candidate& c) {
    if(!stencil(c.x,c.y,in.h,in.w,c.index,c.weight))return false;
    double lo=1e100,hi=-1e100;
    // Zero-weight vertices do not belong to the interpolation support.
    for(int j=0;j<4;++j){if(c.weight[j]<=1e-14)continue;const double z=in.depth1[c.index[j]];
        if(!valid_depth(z))return false;lo=std::min(lo,z);hi=std::max(hi,z);}
    return hi-lo<=cfg.edge_threshold;
}

static double photo_cost(const Input& in,const Config& cfg,int p,const Candidate& center) {
    const int px=p%in.w,py=p/in.w;
    const double z0=in.depth0[p],z1=sample(in.depth1,center);
    // Bounded stack storage avoids a heap allocation for every candidate.
    std::array<std::array<double,6>,289> patch;
    int count=0;const int channels=in.photo_channels;
    std::array<double,6> mean{};
    for(int dy=-cfg.patch_radius;dy<=cfg.patch_radius;++dy)
      for(int dx=-cfg.patch_radius;dx<=cfg.patch_radius;++dx){
        const int x=px+dx,y=py+dy;
        if(x<0||y<0||x>=in.w||y>=in.h)continue;
        const int pp=y*in.w+x;
        if(!valid_depth(in.depth0[pp])||std::abs(in.depth0[pp]-z0)>cfg.edge_threshold)continue;
        Candidate c;c.x=center.x+dx;c.y=center.y+dy;
        if(!safe_target(in,cfg,c)||std::abs(sample(in.depth1,c)-z1)>cfg.edge_threshold)continue;
        std::array<double,6> pair{};
        for(int ch=0;ch<channels;++ch){pair[ch]=in.rgb0[3*pp+ch];
            for(int k=0;k<4;++k)if(c.weight[k]>1e-14)pair[3+ch]+=c.weight[k]*in.rgb1[3*c.index[k]+ch];}
        patch[count++]=pair;for(int ch=0;ch<6;++ch)mean[ch]+=pair[ch];
      }
    // At least three spatial samples for a centred descriptor (except explicit radius zero).
    if(count<(cfg.centered_photo&&cfg.patch_radius>0?3:1))
        return std::numeric_limits<double>::infinity();
    for(double&v:mean)v/=count;
    std::array<double,3> gain{1.,1.,1.};
    if(cfg.affine_photo&&cfg.centered_photo&&cfg.patch_radius>0){
        for(int ch=0;ch<channels;++ch){double variance=0,covariance=0;
            for(int j=0;j<count;++j){const auto&a=patch[j];
                variance+=(a[ch]-mean[ch])*(a[ch]-mean[ch]);
                covariance+=(a[ch]-mean[ch])*(a[3+ch]-mean[3+ch]);}
            if(variance>1e-10)gain[ch]=std::clamp(covariance/variance,.5,2.);
        }
    }
    double loss=0;
    for(int j=0;j<count;++j)for(int ch=0;ch<channels;++ch){
        const auto&a=patch[j];
        const double shift=cfg.centered_photo&&cfg.patch_radius>0?mean[3+ch]-gain[ch]*mean[ch]:0.;
        loss+=huber((a[3+ch]-gain[ch]*a[ch]-shift)/cfg.sigma_image,2.0);
    }
    return loss/(channels*count);
}

Matches make_matches(const Input& in,const Config& cfg) {
    Matches out;const int n=in.n();
    out.candidates.resize(n);out.selected.assign(n,0);out.active.assign(n,0.0);
    const double margin=cfg.trust_sigma*cfg.sigma_depth;
    for(int p=0;p<n;++p){
        if(!valid_depth(in.depth0[p]))continue;
        const int px=p%in.w,py=p/in.w;
        const V3 r0=in.camera.ray(px,py);
        const double l0=std::max(cfg.min_depth,in.depth0[p]-margin);
        const double u0=std::min(cfg.max_depth,in.depth0[p]+margin);
        if(l0>u0)continue;
        std::vector<Candidate> pool;
        pool.reserve(1024);
        auto add=[&](double x,double y){
            Candidate c;c.x=x;c.y=y;
            if(!safe_target(in,cfg,c))return;
            c.ray=in.camera.ray(x,y);++out.proposed;
            if(cfg.enforce_bound){
                double l1=0,u1=0;
                for(int k=0;k<4;++k)if(c.weight[k]>1e-14){
                    l1+=c.weight[k]*std::max(cfg.min_depth,in.depth1[c.index[k]]-margin);
                    u1+=c.weight[k]*std::min(cfg.max_depth,in.depth1[c.index[k]]+margin);
                }
                if(ray_segment_distance(r0,c.ray,l0,u0,l1,u1)>cfg.max_motion+1e-12){
                    ++out.rejected_geometry;return;
                }
            }
            c.photo=cfg.image_weight>0?photo_cost(in,cfg,p,c):0.;
            if(!std::isfinite(c.photo))return;
            const double ex=x-px-in.flow[2*p],ey=y-py-in.flow[2*p+1];
            const double ef=std::isfinite(ex)&&std::isfinite(ey)
                ? std::min(0.5*(ex*ex+ey*ey)/(cfg.sigma_flow*cfg.sigma_flow),cfg.flow_truncation)
                : cfg.flow_truncation;
            c.cost=cfg.image_weight*c.photo+cfg.flow_weight*ef;
            pool.push_back(c);
        };
        const double denominator=l0-cfg.max_motion;
        const int rx=cfg.enforce_bound?std::min(cfg.max_search,static_cast<int>(std::ceil(in.camera.fx*
            cfg.max_motion*std::sqrt(1+r0[0]*r0[0])/denominator))+1):cfg.max_search;
        const int ry=cfg.enforce_bound?std::min(cfg.max_search,static_cast<int>(std::ceil(in.camera.fy*
            cfg.max_motion*std::sqrt(1+r0[1]*r0[1])/denominator))+1):cfg.max_search;
        const int stride=cfg.fast_match?cfg.coarse_stride:1;
        for(int dy=-ry;dy<=ry;dy+=stride)for(int dx=-rx;dx<=rx;dx+=stride)add(px+dx,py+dy);
        if(cfg.fast_match)add(px,py);
        add(px+in.flow[2*p],py+in.flow[2*p+1]);
        if(pool.empty())continue;
        auto less=[](const Candidate&a,const Candidate&b){return a.cost<b.cost;};
        std::stable_sort(pool.begin(),pool.end(),less);
        std::vector<Candidate> seeds;
        for(const auto&c:pool){bool close=false;
            if(cfg.diverse_candidates)for(const auto&s:seeds)if(std::hypot(c.x-s.x,c.y-s.y)<1.5){close=true;break;}
            if(!close)seeds.push_back(c);
            if(seeds.size()>=static_cast<std::size_t>(cfg.fast_match?cfg.seed_count:(cfg.diverse_candidates?5:3)))break;
        }
        // Fixed finite observation-only proposals. The ground truth never enters matching.
        for(const auto& seed:seeds){
            if(!cfg.fast_match){
                for(int dy=-4;dy<=4;++dy)for(int dx=-4;dx<=4;++dx)
                    add(seed.x+dx*0.125,seed.y+dy*0.125);
            }else{
                Candidate current=seed;
                for(double step:{.5,.25,.125}){
                    const Candidate anchor=current;
                    for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx){
                        if(dx==0&&dy==0)continue;
                        const auto before=pool.size();add(anchor.x+dx*step,anchor.y+dy*step);
                        if(pool.size()>before&&pool.back().cost<current.cost)current=pool.back();
                    }
                }
            }
        }
        std::stable_sort(pool.begin(),pool.end(),less);
        auto& kept=out.candidates[p];
        auto keep=[&](double distance,int limit){
            for(const auto&c:pool){bool duplicate=false;
                for(const auto&old:kept)if(std::hypot(c.x-old.x,c.y-old.y)<distance){duplicate=true;break;}
                if(!duplicate)kept.push_back(c);
                if(static_cast<int>(kept.size())>=limit)break;
            }
        };
        if(cfg.diverse_candidates)keep(.65,std::max(1,cfg.candidate_count/2));
        if(static_cast<int>(kept.size())<cfg.candidate_count)keep(.04,cfg.candidate_count);
        if(cfg.keep_identity){
            // Identity is a feasible competing explanation for a moving shadow.
            // Retaining it is NOT a prior that the scene is static.
            for(const auto&c:pool)if(c.x==px&&c.y==py){
                bool found=false;for(const auto&k:kept)if(k.x==px&&k.y==py){found=true;break;}
                if(!found){if(kept.size()>=static_cast<std::size_t>(cfg.candidate_count))kept.pop_back();kept.push_back(c);}
                break;
            }
        }
        if(!kept.empty()&&kept.front().photo<=cfg.photo_reject)out.active[p]=1.0;
    }
    if(cfg.reciprocal_check){
        Input reverse=in;std::swap(reverse.rgb0,reverse.rgb1);std::swap(reverse.depth0,reverse.depth1);
        reverse.flow.assign(2*n,std::numeric_limits<double>::quiet_NaN());
        Config rcfg=cfg;rcfg.reciprocal_check=false;rcfg.flow_weight=0;
        Matches back=make_matches(reverse,rcfg);
        for(int p=0;p<n;++p)if(out.active[p]>0){
            const auto&c=out.candidates[p][0];double bx=0,by=0,weight=0;
            for(int k=0;k<4;++k){const int j=c.index[k];if(c.weight[k]<=1e-14)continue;
                if(back.active[j]<=0)continue;
                const auto&b=back.candidates[j][0];
                bx+=c.weight[k]*(b.x-j%in.w);by+=c.weight[k]*(b.y-j/in.w);weight+=c.weight[k];}
            if(weight<.99||std::hypot(c.x+bx-p%in.w,c.y+by-p/in.w)>1.5)out.active[p]=0.;
        }
        out.proposed+=back.proposed;out.rejected_geometry+=back.rejected_geometry;
    }
    return out;
}
} // namespace sfr
