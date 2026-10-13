// Analytic, material-coordinate ray tracer. No ground-truth data enters the solver.
// Object record (17 doubles): type, halfsize[3], center[3], local-to-world R[9], material seed.
// Types: 0 bounded plane (local z=0), 1 ellipsoid (sphere if sizes equal),
//        2 oriented box, 3 capped cylinder (local y axis, x radius, y halfheight).
#include "types.hpp"
#include <cstdint>
#if defined(_WIN32)
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif
namespace {
using sfr::V3;
V3 plus(const V3&a,const V3&b){return {a[0]+b[0],a[1]+b[1],a[2]+b[2]};}
V3 minus(const V3&a,const V3&b){return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};}
V3 mul(const V3&a,double t){return {a[0]*t,a[1]*t,a[2]*t};}
V3 unit(const V3&a){return mul(a,1/std::max(1e-30,sfr::norm(a)));}
struct Object { int type; V3 s,c; std::array<double,9> R; int seed;
 V3 local(const V3&v) const {V3 r{};for(int j=0;j<3;++j)for(int k=0;k<3;++k)r[j]+=v[k]*R[3*k+j];return r;}
 V3 world(const V3&v) const {V3 r{};for(int j=0;j<3;++j)for(int k=0;k<3;++k)r[j]+=v[k]*R[3*j+k];return r;}
};
struct Hit {double t=std::numeric_limits<double>::infinity();int id=-1;V3 local{},normal{},world{};};
bool intersect(const Object&b,const V3&O,const V3&D,double&best,V3&normal,V3&point){
 const V3 o=b.local(minus(O,b.c)),d=b.local(D);
 double t=std::numeric_limits<double>::infinity(); V3 n{};
 auto accept=[&](double v,const V3&nn){if(v>1e-7&&v<t){t=v;n=nn;}};
 if(b.type==0){
  if(std::abs(d[2])>1e-15){double v=-o[2]/d[2];auto p=plus(o,mul(d,v));
   if(std::abs(p[0])<=b.s[0]&&std::abs(p[1])<=b.s[1]) accept(v,{0,0,d[2]>0?-1.:1.});}
 } else if(b.type==1){
  double A=0,B=0,C=-1;for(int k=0;k<3;++k){double ss=b.s[k]*b.s[k];A+=d[k]*d[k]/ss;B+=o[k]*d[k]/ss;C+=o[k]*o[k]/ss;}
  const double det=B*B-A*C;
  if(det>=0&&A>0)for(double v:{(-B-std::sqrt(det))/A,(-B+std::sqrt(det))/A}){
   auto p=plus(o,mul(d,v));V3 nn{};for(int k=0;k<3;++k)nn[k]=p[k]/(b.s[k]*b.s[k]);accept(v,unit(nn));}
 } else if(b.type==2){
  double near=-std::numeric_limits<double>::infinity(),far=std::numeric_limits<double>::infinity();V3 nn{},nf{};bool valid=true;
  for(int k=0;k<3;++k){if(std::abs(d[k])<1e-15){if(std::abs(o[k])>b.s[k])valid=false;continue;}
   double a=(-b.s[k]-o[k])/d[k],z=(b.s[k]-o[k])/d[k];V3 na{},nz{};na[k]=-1;nz[k]=1;
   if(a>z){std::swap(a,z);std::swap(na,nz);}if(a>near){near=a;nn=na;}if(z<far){far=z;nf=nz;}}
  if(valid&&near<=far){accept(near,nn);accept(far,nf);}
 } else if(b.type==3){
  const double A=d[0]*d[0]+d[2]*d[2],B=o[0]*d[0]+o[2]*d[2],C=o[0]*o[0]+o[2]*o[2]-b.s[0]*b.s[0];
  const double det=B*B-A*C;
  if(A>1e-15&&det>=0)for(double v:{(-B-std::sqrt(det))/A,(-B+std::sqrt(det))/A}){
   auto p=plus(o,mul(d,v));if(std::abs(p[1])<=b.s[1])accept(v,unit({p[0],0,p[2]}));}
  if(std::abs(d[1])>1e-15)for(double sy:{-1.,1.}){double v=(sy*b.s[1]-o[1])/d[1];auto p=plus(o,mul(d,v));
   if(p[0]*p[0]+p[2]*p[2]<=b.s[0]*b.s[0])accept(v,{0,sy,0});}
 }
 if(t<best){best=t;normal=unit(b.world(n));point=plus(o,mul(d,t));return true;}return false;
}
Hit trace(const std::vector<Object>&objects,const V3&O,const V3&D){Hit hit;
 for(int i=0;i<int(objects.size());++i){V3 n,p;double t=hit.t;if(intersect(objects[i],O,D,t,n,p)){hit.t=t;hit.id=i;hit.normal=n;hit.local=p;}}
 if(hit.id>=0)hit.world=plus(O,mul(D,hit.t));return hit;
}
double hash(int seed,int k){uint32_t x=uint32_t(seed)*2654435761u+uint32_t(k)*2246822519u;x^=x>>16;x*=2246822519u;x^=x>>13;return double(x)/4294967296.;}
double albedo(const V3&p,int seed,double contrast){
 double sum=0;
 for(int j=0;j<14;++j){V3 f{};double frequency=j<5?45.:j<10?105.:185.;for(int k=0;k<3;++k)f[k]=(2*hash(seed,4*j+k)-1)*frequency;
  sum+=std::sin(sfr::dot(p,f)+6.283185307179586*hash(seed,4*j+3));}
 // A smooth, non-tiled multi-frequency material: no camera-coordinate texture swimming.
 return std::clamp(.57+contrast*.105*sum/std::sqrt(3.5),.07,.96);
}
double shade(const Hit&hit,const std::vector<Object>&objects,double contrast,bool shadows){
 const V3 lights[3]={{-.58,-.74,.15},{-.48,-.74,.15},{-.53,-.64,.15}};
 double diffuse=0;
 for(const V3&l:lights){V3 delta=minus(l,hit.world);double distance=sfr::norm(delta);V3 direction=mul(delta,1/distance);
  double visibility=1.;if(shadows){auto s=trace(objects,plus(hit.world,mul(hit.normal,1e-5)),direction);if(s.id>=0&&s.t<distance)visibility=0.;}
  diffuse+=visibility*std::max(0.,sfr::dot(hit.normal,direction))/3.;}
 const double fill=.08*std::max(0.,sfr::dot(hit.normal,unit({.7,-.1,-1.})));
 return std::clamp(albedo(hit.local,objects[hit.id].seed,contrast*(objects[hit.id].type==0?.28:1.))*(.43+.54*diffuse+fill),0.,1.);
}
}
// Output Nx12: z, gray, object_id, local xyz, world normal xyz, world xyz.
API int sfr_render(int count,const double*pixels,const double*intr,int object_count,
 const double*records,double contrast,int shadows,int with_shading,double*out){
 try {
  if(count<0||object_count<1||!pixels||!intr||!records||!out||intr[0]<=0||intr[1]<=0)return -1;
  std::vector<Object> objects;
  for(int i=0;i<object_count;++i){const double*a=records+17*i;Object b;b.type=int(a[0]);
   if(b.type<0||b.type>3)return -2;
   for(int k=0;k<3;++k){b.s[k]=a[1+k];b.c[k]=a[4+k];if(b.s[k]<=0)return -3;}
   for(int k=0;k<9;++k)b.R[k]=a[7+k];b.seed=int(a[16]);objects.push_back(b);}
  for(int i=0;i<count;++i){V3 ray{(pixels[2*i]-intr[2])/intr[0],(pixels[2*i+1]-intr[3])/intr[1],1};
   Hit h=trace(objects,{0,0,0},ray);double*v=out+12*i;v[0]=h.t;v[1]=h.id>=0&&with_shading?shade(h,objects,contrast,shadows!=0):0;v[2]=h.id;
   for(int k=0;k<3;++k){v[3+k]=h.local[k];v[6+k]=h.normal[k];v[9+k]=h.world[k];}}
  return 0;
 }catch(...){return -10;}
}
