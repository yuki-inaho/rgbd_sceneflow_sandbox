#include "solver.hpp"
#include <chrono>
#include <numeric>
#include <random>

namespace sfr {
void Operator::row(const std::vector<std::pair<int,double>>& entries){
    for(const auto& [i,v]:entries)if(v!=0){col.push_back(i);value.push_back(v);}
    row_ptr.push_back(static_cast<int>(col.size()));
}
void Operator::group(int start,int size,double weight,Penalty type,double cap){
    groups.push_back({start,size,weight,0,cap,type});
}
void Operator::apply(const std::vector<double>&x,std::vector<double>&y)const{
    y.assign(rows(),0);
    for(int r=0;r<rows();++r){double s=0;
        for(int j=row_ptr[r];j<row_ptr[r+1];++j)s+=value[j]*x[col[j]];y[r]=s;}
}
void Operator::transpose(const std::vector<double>&y,std::vector<double>&x)const{
    x.assign(columns,0);
    for(int r=0;r<rows();++r)for(int j=row_ptr[r];j<row_ptr[r+1];++j)x[col[j]]+=value[j]*y[r];
}
void Operator::prepare_steps(int n){
    std::vector<double> rowsum(rows(),0),colsum(columns,0);
    for(int r=0;r<rows();++r)for(int j=row_ptr[r];j<row_ptr[r+1];++j){
        const double a=std::abs(value[j]);rowsum[r]+=a;colsum[col[j]]+=a;}
    tau.resize(columns);
    for(int i=0;i<columns;++i)tau[i]=0.95/std::max(1.0,colsum[i]);
    // Equal steps in each scene-vector block make its metric prox a Euclidean ball projection.
    for(int p=0;p<n;++p){const int b=2*n+3*p;
        const double t=std::min({tau[b],tau[b+1],tau[b+2]});
        for(int c=0;c<3;++c)tau[b+c]=t;}
    for(auto&g:groups){double maximum=1.0;
        for(int k=0;k<g.size;++k)maximum=std::max(maximum,rowsum[g.start+k]);g.sigma=0.95/maximum;}
}
void Operator::dual_prox(std::vector<double>&y)const{
    for(const auto&g:groups){
        if(g.type!=Penalty::Norm){
            const double den=1+g.sigma/g.weight;
            for(int k=0;k<g.size;++k)y[g.start+k]/=den;
            if(g.type==Penalty::Huber){double sq=0;for(int k=0;k<g.size;++k)sq+=y[g.start+k]*y[g.start+k];
                const double factor=std::max(1.,std::sqrt(sq)/g.cap);for(int k=0;k<g.size;++k)y[g.start+k]/=factor;}
        }else{
            double l2=0;for(int k=0;k<g.size;++k)l2+=y[g.start+k]*y[g.start+k];
            const double d=std::max(1.0,std::sqrt(l2)/g.weight);
            for(int k=0;k<g.size;++k)y[g.start+k]/=d;
        }
    }
}
// Identical PDHG dual update, but no intermediate Kx buffer or repeated group pass.
void Operator::fused_dual(const std::vector<double>&xb,std::vector<double>&y,std::vector<double>&kty)const{
    kty.assign(columns,0);
    for(const auto&g:groups){double l2=0;
        for(int k=0;k<g.size;++k){const int r=g.start+k;double s=0;
            for(int j=row_ptr[r];j<row_ptr[r+1];++j)s+=value[j]*xb[col[j]];
            y[r]+=g.sigma*s;l2+=y[r]*y[r];}
        double d=g.type!=Penalty::Norm?1+g.sigma/g.weight:std::max(1.,std::sqrt(l2)/g.weight);
        if(g.type==Penalty::Huber)d=std::max(d,std::sqrt(l2)/g.cap);
        for(int k=0;k<g.size;++k){const int r=g.start+k;y[r]/=d;
            for(int j=row_ptr[r];j<row_ptr[r+1];++j)kty[col[j]]+=value[j]*y[r];}
    }
}
double Operator::energy(const std::vector<double>&kx)const{
    double e=0;for(const auto&g:groups){double l2=0;
        for(int k=0;k<g.size;++k)l2+=kx[g.start+k]*kx[g.start+k];
        e+=g.weight*(g.type==Penalty::Quadratic ? 0.5*l2 : (g.type==Penalty::Huber ? huber(std::sqrt(l2),g.cap/g.weight) : std::sqrt(l2)));}return e;
}
double Operator::adjoint_test()const{
    std::mt19937 rng(437);std::normal_distribution<double> normal;
    std::vector<double>x(columns),y(rows()),kx,kty;
    for(double&v:x)v=normal(rng);for(double&v:y)v=normal(rng);
    apply(x,kx);transpose(y,kty);
    const double a=std::inner_product(kx.begin(),kx.end(),y.begin(),0.0);
    const double b=std::inner_product(x.begin(),x.end(),kty.begin(),0.0);
    return std::abs(a-b)/std::max({1.0,std::abs(a),std::abs(b)});
}

static void add_regularizer(Operator&op,const Input&in,const Config&cfg,
    const std::vector<double>&guide,int u_base,int channels,int w_base,double a1,double a0){
    if(a1==0&&a0==0)return;
    const int n=in.n();
    for(int p=0;p<n;++p)for(int axis=0;axis<2;++axis){
        if((axis==0&&p%in.w==in.w-1)||(axis==1&&p/in.w==in.h-1))continue;
        const int r=p+(axis==0?1:in.w);
        // Missing-depth links are conservatively cut. Main experiments use dense depth.
        if(cfg.edge_aware&&(!valid_depth(guide[p])||!valid_depth(guide[r])||
            std::abs(guide[p]-guide[r])>cfg.edge_threshold))continue;
        if(a1>0){const int start=op.rows();
            for(int c=0;c<channels;++c){
                std::vector<std::pair<int,double>> row{{u_base+channels*r+c,1},{u_base+channels*p+c,-1}};
                if(cfg.use_tgv){row.emplace_back(w_base+(p*2+axis)*channels+c,-0.5);
                    row.emplace_back(w_base+(r*2+axis)*channels+c,-0.5);}
                op.row(row);
            }
            op.group(start,channels,a1,Penalty::Norm);
        }
        if(cfg.use_tgv&&a0>0){const int start=op.rows();
            for(int c=0;c<2*channels;++c)
                op.row({{w_base+2*channels*r+c,1},{w_base+2*channels*p+c,-1}});
            op.group(start,2*channels,a0,Penalty::Norm);
        }
    }
}

double regularizer_energy(int h,int w,int channels,const double* u,const double* aux,
    const double* guide,double edge_threshold,double alpha1,double alpha0,bool cut){
    if(h<1||w<1||channels<1)throw std::invalid_argument("invalid regularizer dimensions");
    Input in;in.h=h;in.w=w;const int n=h*w;
    Config cfg;cfg.edge_threshold=edge_threshold;cfg.edge_aware=cut;
    std::vector<double> depth(guide,guide+n),x(u,u+n*channels),kx;
    x.insert(x.end(),aux,aux+2*n*channels);
    Operator op(3*n*channels);
    add_regularizer(op,in,cfg,depth,0,channels,n*channels,alpha1,alpha0);
    op.apply(x,kx);return op.energy(kx);
}

static Operator make_operator(const Input&in,const Config&cfg,const Matches&m){
    const int n=in.n();Operator op(15*n);
    // Layout: z0[N], z1[N], s[3N], w0[2N], w1[2N], ws[6N].
    add_regularizer(op,in,cfg,in.depth0,0,1,5*n,cfg.depth_alpha1,cfg.depth_alpha0);
    add_regularizer(op,in,cfg,in.depth1,n,1,7*n,cfg.depth_alpha1,cfg.depth_alpha0);
    add_regularizer(op,in,cfg,in.depth0,2*n,3,9*n,cfg.flow_alpha1,cfg.flow_alpha0);
    const double eta=cfg.sigma_geometry/cfg.scale;
    for(int p=0;p<n;++p)if(m.active[p]>0){
        const auto&c=m.candidates[p][m.selected[p]];
        const V3 r0=in.camera.ray(p%in.w,p/in.w);const int start=op.rows();
        for(int ch=0;ch<3;++ch){
            std::vector<std::pair<int,double>> row{{p,-r0[ch]},{2*n+3*p+ch,-1}};
            for(int k=0;k<4;++k)row.emplace_back(n+c.index[k],c.weight[k]*c.ray[ch]);
            op.row(row);
        }
        op.group(start,3,m.active[p]/(eta*eta),cfg.geometry_huber?Penalty::Huber:Penalty::Quadratic,m.active[p]*cfg.geometry_k/eta);
    }
    op.prepare_steps(n);
    for(double&t:op.tau)t*=cfg.step_balance;
    for(auto&g:op.groups)g.sigma/=cfg.step_balance;
    return op;
}

static void primal_prox(std::vector<double>&x,const Operator&op,const Input&in,const Config&cfg,const RigidPrior&prior){
    const int n=in.n();const double eta=cfg.sigma_depth/cfg.scale;
    const double margin=cfg.trust_sigma*eta;
    for(int t=0;t<2;++t){const auto&data=t==0?in.depth0:in.depth1;
        for(int p=0;p<n;++p){const int i=t*n+p;
            double lo=cfg.min_depth/cfg.scale,hi=cfg.max_depth/cfg.scale;
            if(valid_depth(data[p])){
                const double y=data[p]/cfg.scale;
                x[i]=huber_prox(x[i],y,eta,cfg.huber_k,op.tau[i]);
                lo=std::max(lo,y-margin);hi=std::min(hi,y+margin);
            }
            x[i]=std::clamp(x[i],lo,hi);
        }
    }
    // Fixed observation-derived rigid prediction is a SOFT quadratic centre.
    const double eta_r=cfg.rigid_sigma/cfg.scale;
    for(int p=0;p<n;++p)if(prior.weight[p]>0){const int b=2*n+3*p;const double v=op.tau[b]*prior.weight[p]/(eta_r*eta_r);
        for(int ch=0;ch<3;++ch)x[b+ch]=(x[b+ch]+v*prior.scene[3*p+ch]/cfg.scale)/(1+v);}
    if(cfg.enforce_bound){const double radius=cfg.max_motion/cfg.scale;
        for(int p=0;p<n;++p){const int i=2*n+3*p;const double len=std::sqrt(x[i]*x[i]+x[i+1]*x[i+1]+x[i+2]*x[i+2]);
            const double factor=std::max(1.0,len/radius);for(int c=0;c<3;++c)x[i+c]/=factor;}}
    for(int i=5*n;i<15*n;++i)x[i]/=1+op.tau[i]*cfg.ridge;
}

static double primal_energy(const std::vector<double>&x,const Input&in,const Config&cfg,const RigidPrior&prior){
    const int n=in.n();double e=0;const double eta=cfg.sigma_depth/cfg.scale;
    for(int t=0;t<2;++t){const auto&d=t==0?in.depth0:in.depth1;
        for(int p=0;p<n;++p)if(valid_depth(d[p]))e+=huber((x[t*n+p]-d[p]/cfg.scale)/eta,cfg.huber_k);}
    const double eta_r=cfg.rigid_sigma/cfg.scale;
    for(int p=0;p<n;++p)if(prior.weight[p]>0)for(int ch=0;ch<3;++ch){const double v=x[2*n+3*p+ch]-prior.scene[3*p+ch]/cfg.scale;
        e+=.5*prior.weight[p]*v*v/(eta_r*eta_r);}
    for(int i=5*n;i<15*n;++i)e+=0.5*cfg.ridge*x[i]*x[i];return e;
}
static double match_energy(const Matches&m){
    double e=0;for(std::size_t p=0;p<m.active.size();++p)if(m.active[p]>0)
        e+=m.active[p]*m.candidates[p][m.selected[p]].cost;return e;
}
static V3 lift(const std::vector<double>&x,const Input&in,int p,const Candidate&c){
    const int n=in.n();double z=0;for(int k=0;k<4;++k)z+=c.weight[k]*x[n+c.index[k]];
    V3 s{},r=in.camera.ray(p%in.w,p/in.w);
    for(int ch=0;ch<3;++ch)s[ch]=z*c.ray[ch]-x[p]*r[ch];return s;
}
static int update_matches(Matches&m,const std::vector<double>&x,const Input&in,const Config&cfg){
    const double eta=cfg.sigma_geometry/cfg.scale;int changed=0;
    for(int p=0;p<in.n();++p)if(m.active[p]>0){
        int best=m.selected[p];double bestcost=std::numeric_limits<double>::infinity();
        for(int k=0;k<static_cast<int>(m.candidates[p].size());++k){
            const auto&c=m.candidates[p][k];const V3 lifted=lift(x,in,p,c);double sq=0;
            for(int ch=0;ch<3;++ch){const double r=lifted[ch]-x[2*in.n()+3*p+ch];sq+=r*r;}
            const double cost=c.cost+(cfg.geometry_huber?huber(std::sqrt(sq)/eta,cfg.geometry_k):0.5*sq/(eta*eta));
            if(cost<bestcost){bestcost=cost;best=k;}
        }
        if(best!=m.selected[p]){m.selected[p]=best;++changed;}
    }
    return changed;
}

static void convex_block(std::vector<double>&x,const Input&in,const Config&cfg,
                         const Matches&m,int outer,Result&result,const RigidPrior&prior,std::vector<double>&warm){
    auto op=make_operator(in,cfg,m);
    result.adjoint_error=std::max(result.adjoint_error,op.adjoint_test());
    std::vector<double> y(op.rows(),0);
    if(cfg.warm_start&&warm.size()==y.size())y=warm;
    std::vector<double> xb=x,kx,kty,next,best=x,besty=y;
    const double matching=match_energy(m);
    auto objective=[&](const std::vector<double>&state){
        op.apply(state,kx);return op.energy(kx)+primal_energy(state,in,cfg,prior)+matching;};
    double best_energy=objective(x);
    result.history.insert(result.history.end(),{static_cast<double>(outer),0,best_energy,best_energy});
    const int iterations=outer==0?cfg.iterations:cfg.later_iterations;
    result.inner_iterations+=iterations;
    for(int it=1;it<=iterations;++it){
      if(cfg.fused_solver)op.fused_dual(xb,y,kty);
      else{
        op.apply(xb,kx);
        for(const auto&g:op.groups)for(int k=0;k<g.size;++k)y[g.start+k]+=g.sigma*kx[g.start+k];
        op.dual_prox(y);op.transpose(y,kty);
      }
        next=x;for(int i=0;i<op.columns;++i)next[i]-=op.tau[i]*kty[i];
        primal_prox(next,op,in,cfg,prior);
        for(int i=0;i<op.columns;++i)xb[i]=2*next[i]-x[i];x.swap(next);
        if(it%25==0||it==iterations){
            const double energy=objective(x);
            if(!std::isfinite(energy))throw std::runtime_error("non-finite primal objective");
            if(energy<best_energy){best_energy=energy;best=x;besty=y;}
            result.history.insert(result.history.end(),{static_cast<double>(outer),static_cast<double>(it),energy,best_energy});
        }
    }
    // Preserve monotone outer-block acceptance, not a false claim that every CP iterate decreases energy.
    x=best;y=besty;warm=y;
    op.transpose(y,kty);next=x;
    for(int i=0;i<op.columns;++i)next[i]-=op.tau[i]*kty[i];primal_prox(next,op,in,cfg,prior);
    result.fixed_point_primal=0;
    for(int i=0;i<op.columns;++i)result.fixed_point_primal=std::max(result.fixed_point_primal,std::abs(x[i]-next[i])/op.tau[i]);
    op.apply(x,kx);auto ynext=y;
    for(const auto&g:op.groups)for(int k=0;k<g.size;++k)ynext[g.start+k]+=g.sigma*kx[g.start+k];
    op.dual_prox(ynext);result.fixed_point_dual=0;
    for(const auto&g:op.groups)for(int k=0;k<g.size;++k)
        result.fixed_point_dual=std::max(result.fixed_point_dual,std::abs(y[g.start+k]-ynext[g.start+k])/g.sigma);
}

Result solve(const Input&in,const Config&cfg){
    cfg.validate();const auto start=std::chrono::steady_clock::now();
    const int n=in.n();Result r;r.h=in.h;r.w=in.w;
    Matches m=make_matches(in,cfg);
    r.matching_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    const auto prior_start=std::chrono::steady_clock::now();
    const RigidPrior prior=fit_rigid_prior(in,cfg,m);
    r.rigid_scene=prior.scene;r.rigid_weight=prior.weight;r.components=prior.component;
    r.rigid_components=prior.components;r.rigid_accepted=prior.accepted;
    r.prior_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-prior_start).count();
    const auto optimization_start=std::chrono::steady_clock::now();
    std::vector<double>x(15*n,0),warm;
    for(int p=0;p<n;++p){x[p]=(valid_depth(in.depth0[p])?in.depth0[p]:std::clamp(1.0,cfg.min_depth,cfg.max_depth))/cfg.scale;
        x[n+p]=(valid_depth(in.depth1[p])?in.depth1[p]:std::clamp(1.0,cfg.min_depth,cfg.max_depth))/cfg.scale;}
    r.initial_scene_flow.assign(3*n,0);
    for(int p=0;p<n;++p){
        r.candidate_total+=static_cast<int>(m.candidates[p].size());
        if(m.active[p]>0){V3 s=lift(x,in,p,m.candidates[p][0]);
            for(int c=0;c<3;++c)r.initial_scene_flow[3*p+c]=s[c]*cfg.scale;
            const double fac=cfg.enforce_bound?std::max(1.0,norm(s)*cfg.scale/cfg.max_motion):1.0;
            for(int c=0;c<3;++c)x[2*n+3*p+c]=s[c]/fac;}
    }
    for(int outer=0;outer<cfg.outer_iterations;++outer){
        if(outer>0)r.candidate_changes+=update_matches(m,x,in,cfg);
        convex_block(x,in,cfg,m,outer,r,prior,warm);
    }
    r.depth0.resize(n);r.depth1.resize(n);r.scene_flow.resize(3*n);r.flow.resize(2*n);
    r.match_flow.assign(2*n,std::numeric_limits<double>::quiet_NaN());
    r.geometry_residual.assign(n,std::numeric_limits<double>::quiet_NaN());r.support=m.active;
    for(int p=0;p<n;++p){
        r.depth0[p]=x[p]*cfg.scale;r.depth1[p]=x[n+p]*cfg.scale;
        V3 position=in.camera.ray(p%in.w,p/in.w);
        for(int c=0;c<3;++c){r.scene_flow[3*p+c]=x[2*n+3*p+c]*cfg.scale;
            position[c]=position[c]*r.depth0[p]+r.scene_flow[3*p+c];}
        if(position[2]>0){r.flow[2*p]=in.camera.fx*position[0]/position[2]+in.camera.cx-p%in.w;
            r.flow[2*p+1]=in.camera.fy*position[1]/position[2]+in.camera.cy-p/in.w;
        }else{r.flow[2*p]=r.flow[2*p+1]=std::numeric_limits<double>::quiet_NaN();}
        if(m.active[p]>0){const auto&c=m.candidates[p][m.selected[p]];
            r.match_flow[2*p]=c.x-p%in.w;r.match_flow[2*p+1]=c.y-p/in.w;
            V3 e=lift(x,in,p,c);for(int ch=0;ch<3;++ch)e[ch]-=x[2*n+3*p+ch];
            r.geometry_residual[p]=norm(e)*cfg.scale;}
    }
    r.optimization_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-optimization_start).count();
    r.proposed=m.proposed;r.rejected_geometry=m.rejected_geometry;
    r.elapsed_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    return r;
}
} // namespace sfr
