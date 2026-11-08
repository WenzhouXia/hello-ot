#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/warp/warp_merge_sort.cuh>
#include <limits>
#include <type_traits>

// CN: 移植 hello_ot 的 initialization warp queue；只替换分数生成，不复用未检查的范数展开。
// EN: Port hello_ot's initialization warp queue, replacing score generation without reusing unchecked norm expansion.
struct Candidate {double value;int32_t index;};
__device__ __forceinline__ bool better(Candidate a,Candidate b) {
    return a.value>b.value || (a.value==b.value && a.index>=0 && (b.index<0 || a.index<b.index));
}
struct Compare {
    __device__ __forceinline__ bool operator()(const Candidate& a,const Candidate& b)const{return better(a,b);}
};
template<typename Sort>
__device__ __forceinline__ void merge_queue(Candidate (&thread_queue)[2],int& count,Candidate* queue,Candidate* items,typename Sort::TempStorage& storage) {
    int lane=threadIdx.x%32;
    Candidate local[3]={queue[lane],thread_queue[0],thread_queue[1]};
    Sort(storage).Sort(local,Compare{});
    #pragma unroll
    for(int j=0;j<3;++j)items[lane*3+j]=local[j];
    __syncwarp();queue[lane]=items[lane];
    thread_queue[0]={-INFINITY,-1};thread_queue[1]={-INFINITY,-1};count=0;__syncwarp();
}
template<int KIND> __device__ float accumulate(float a,float x,float y) {
    float z=x-y;
    if constexpr(KIND==0)return a+fabsf(z);
    else if constexpr(KIND==1)return fmaxf(a,fabsf(z));
    else return a+z*z;
}
template<int W,int T,int KIND,int MODE,int FIXED=0,bool STATS=false>
__global__ void queue_kernel(const float* x,const float* y,const double* g,const float* matrix,const double* nx,const double* ny,int* flag,int q,int n,int d,double* values,int64_t* indices,int k,double* statistics) {
    constexpr int F=FIXED?FIXED:32,C=T/32;
    __shared__ float sq[MODE?1:F*(W+1)],sd[MODE?1:F*(T+1)];
    int lane=threadIdx.x%32,warp=threadIdx.x/32,row=blockIdx.x*W+warp,dim=FIXED?FIXED:d;
    Candidate best={-INFINITY,-1};
    double fixed_sum=0,fixed_max=0;
    bool ambiguous=false;float qnorm=MODE==2 && row<q?(float)nx[row]:0;
    for(int c0=0;c0<n;c0+=T) {
        float a[C]={};
        if constexpr(MODE==0) {
            for(int base=0;base<dim;base+=F) {
                for(int p=threadIdx.x;p<W*F;p+=W*32){int r=p/F,k=p%F,rr=blockIdx.x*W+r;sq[k*(W+1)+r]=rr<q && base+k<dim?x[(int64_t)rr*dim+base+k]:0;}
                for(int p=threadIdx.x;p<T*F;p+=W*32){int c=p/F,k=p%F;sd[k*(T+1)+c]=c0+c<n && base+k<dim?y[(int64_t)(c0+c)*dim+base+k]:0;}
                __syncthreads();float partial[C]={};
                if(row<q)for(int k=0;k<min(F,dim-base);++k) {
                    float v=sq[k*(W+1)+warp];
                    #pragma unroll
                    for(int j=0;j<C;++j)partial[j]=accumulate<KIND>(partial[j],v,sd[k*(T+1)+lane+j*32]);
                }
                #pragma unroll
                for(int j=0;j<C;++j)a[j]=KIND==1?fmaxf(a[j],partial[j]):a[j]+partial[j];
                __syncthreads();
            }
        }
        #pragma unroll
        for(int j=0;j<C;++j) {
            int col=c0+lane+j*32;
            if(row<q && col<n) {
                float cost=(MODE==5 || MODE==7)?0.f:(MODE?matrix[(int64_t)row*n+col]:a[j]);
                if constexpr(MODE==2) {
                    float scale=qnorm+(float)ny[col];float squared=fmaxf(scale-2*cost,0);
                    ambiguous=ambiguous || squared<=8*d*1.1920928955078125e-7f*scale;
                    cost=KIND==2?sqrtf(squared):squared;
                } else if constexpr(MODE==0 && KIND==2)cost=sqrtf(cost);
                double score=g[col]-(double)cost;
                if constexpr(MODE==3 || MODE==6) {
                    // CN: mode6接收一般raw偏置；mode3保留平方距离相消检查。
                    // EN: Mode 6 accepts general raw biases; mode 3 retains squared-distance cancellation checks.
                    score=(static_cast<double>(cost)+(MODE==6?nx[row]:-nx[row]))+g[col];
                    if constexpr(MODE==3) {
                        double scale=nx[row]+ny[col],squared=scale-static_cast<double>(cost);
                        ambiguous=ambiguous || squared<=8*d*1.1920928955078125e-7*scale;
                    }
                }
                if constexpr(MODE==5)score=g[col]-reinterpret_cast<const double*>(matrix)[(int64_t)row*n+col];
                if constexpr(MODE==7)score=(reinterpret_cast<const double*>(matrix)[(int64_t)row*n+col]+nx[row])+g[col];
                if constexpr(STATS){double c=MODE==5?reinterpret_cast<const double*>(matrix)[(int64_t)row*n+col]:static_cast<double>(cost);fixed_sum+=c*c;fixed_max=fmax(fixed_max,fabs(c));}
                Candidate candidate={score,col};
                if(better(candidate,best))best=candidate;
            }

        }
    }
    for(int offset=16;offset>0;offset/=2){
        Candidate other={__shfl_down_sync(0xffffffffu,best.value,offset),__shfl_down_sync(0xffffffffu,best.index,offset)};
        if(lane+offset<32 && better(other,best))best=other;
    }
    if constexpr(STATS){
        for(int offset=16;offset>0;offset/=2){fixed_sum+=__shfl_down_sync(0xffffffffu,fixed_sum,offset);fixed_max=fmax(fixed_max,__shfl_down_sync(0xffffffffu,fixed_max,offset));}
        if(row<q && lane==0){atomicAdd(statistics,fixed_sum);auto address=reinterpret_cast<unsigned long long*>(statistics+1);auto old=*address;
            while(__longlong_as_double(old)<fixed_max){auto assumed=old;old=atomicCAS(address,assumed,__double_as_longlong(fixed_max));if(old==assumed)break;}}
    }
    if constexpr(MODE==2 || MODE==3)if(__any_sync(0xffffffffu,ambiguous) && lane==0)atomicOr(flag,1);
    if(row<q && lane==0){values[row]=-best.value;indices[row]=best.index;}
}

template<int W,int T,int KIND,int MODE,bool STATS=false>
void launch(torch::Tensor x,torch::Tensor y,torch::Tensor g,torch::Tensor matrix,torch::Tensor nx,torch::Tensor ny,torch::Tensor flag,torch::Tensor values,torch::Tensor indices,int k,torch::Tensor statistics={}) {
    int q=x.size(0),n=y.size(0),d=x.size(1);auto s=at::cuda::getCurrentCUDAStream();
#define CALL(D) queue_kernel<W,T,KIND,MODE,D,STATS><<<(q+W-1)/W,W*32,0,s>>>(reinterpret_cast<const float*>(x.data_ptr()),reinterpret_cast<const float*>(y.data_ptr()),g.data_ptr<double>(),(MODE==5 || MODE==7)?reinterpret_cast<const float*>(matrix.data_ptr<double>()):(MODE?matrix.data_ptr<float>():nullptr),nx.data_ptr<double>(),ny.data_ptr<double>(),flag.data_ptr<int>(),q,n,d,values.data_ptr<double>(),indices.data_ptr<int64_t>(),k,STATS?statistics.data_ptr<double>():nullptr)
    if constexpr(MODE){CALL(0);}else if(d==2){CALL(2);}else if(d==3){CALL(3);}else{CALL(0);}
#undef CALL
}
std::vector<torch::Tensor> topk(torch::Tensor x,torch::Tensor y,torch::Tensor g,int kind,int config,torch::Tensor matrix,int mode,torch::Tensor nx,torch::Tensor ny,torch::Tensor flag,int k,bool collect_statistics=false) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && y.is_contiguous() && ((x.scalar_type()==torch::kFloat32 && y.scalar_type()==torch::kFloat32) || (mode==7 && x.scalar_type()==torch::kFloat64 && y.scalar_type()==torch::kFloat64)),"Expected contiguous CUDA FP32 features");
    TORCH_CHECK(g.scalar_type()==torch::kFloat64 && g.is_contiguous(),"Expected contiguous FP64 potentials");
    TORCH_CHECK(y.size(0)>0 && k==1 && config>=0 && config<4 && kind>=0 && kind<4,"Invalid queue configuration");
    TORCH_CHECK((mode==0 || mode==1 || mode==2 || mode==3 || mode==5 || mode==6 || mode==7) && (mode!=2 || kind>=2) && (mode!=3 || kind==3),"Invalid distance mode");
    c10::cuda::CUDAGuard guard(x.device());
    auto values=torch::empty({x.size(0),k},x.options().dtype(torch::kFloat64));
    auto indices=torch::empty({x.size(0),k},x.options().dtype(torch::kInt64));
    auto statistics=collect_statistics?torch::zeros({2},x.options().dtype(torch::kFloat64)):torch::Tensor();
#define STATS_GEOMETRY(C,M) if(config==0){launch<4,64,C,M,true>(x,y,g,matrix,nx,ny,flag,values,indices,k,statistics);}else if(config==1){launch<4,128,C,M,true>(x,y,g,matrix,nx,ny,flag,values,indices,k,statistics);}else if(config==2){launch<8,64,C,M,true>(x,y,g,matrix,nx,ny,flag,values,indices,k,statistics);}else{launch<8,128,C,M,true>(x,y,g,matrix,nx,ny,flag,values,indices,k,statistics);}
    if(collect_statistics){
        TORCH_CHECK(kind<3 && (mode==0 || mode==1 || mode==5),"Statistics require norm costs");
        if(mode==5){STATS_GEOMETRY(0,5);}else if(mode==1){STATS_GEOMETRY(0,1);}
        else if(kind==0){STATS_GEOMETRY(0,0);}else if(kind==1){STATS_GEOMETRY(1,0);}else{STATS_GEOMETRY(2,0);}
        C10_CUDA_KERNEL_LAUNCH_CHECK();return {values,indices,statistics};
    }
#undef STATS_GEOMETRY
#define GEOMETRY(C,M) if(config==0){launch<4,64,C,M>(x,y,g,matrix,nx,ny,flag,values,indices,k);}else if(config==1){launch<4,128,C,M>(x,y,g,matrix,nx,ny,flag,values,indices,k);}else if(config==2){launch<8,64,C,M>(x,y,g,matrix,nx,ny,flag,values,indices,k);}else{launch<8,128,C,M>(x,y,g,matrix,nx,ny,flag,values,indices,k);}
    if(mode==7){GEOMETRY(3,7);}else if(mode==6){GEOMETRY(3,6);}else if(mode==5){GEOMETRY(0,5);}else if(mode==3){GEOMETRY(3,3);}else if(mode==2){if(kind==2){GEOMETRY(2,2);}else{GEOMETRY(3,2);}}
    else if(mode==1){GEOMETRY(0,1);}else if(kind==0){GEOMETRY(0,0);}else if(kind==1){GEOMETRY(1,0);}else if(kind==2){GEOMETRY(2,0);}else{GEOMETRY(3,0);}
#undef GEOMETRY
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {values,indices};
}
// CN: 维度不超过warp宽度时，每lane计算一个距离，扫描中无块同步或成本矩阵。
// EN: For dimensions up to warp width, compute one distance per lane without per-tile barriers or a cost matrix.
template<int KIND,bool WIDE,bool STATS> __global__ void direct_top1_kernel(const float* __restrict__ x,const float* __restrict__ y,const double* __restrict__ g,
        int q,int n,int d,int width,double* values,int64_t* ids,double* stats) {
    int lane=threadIdx.x%32,warp=threadIdx.x/32,row=blockIdx.x;
    __shared__ float query[32];
    if(threadIdx.x<32)query[lane]=row<q && lane<d?x[(int64_t)row*d+lane]:0;
    __syncthreads();if(row>=q)return;
    using Acc=typename std::conditional<WIDE,double,float>::type;
    Candidate best={-INFINITY,-1};double sum=0,maximum=0;
    for(int col=blockIdx.y*width+threadIdx.x;col<min(n,(int)(blockIdx.y+1)*width);col+=blockDim.x){Acc cost=0;
        for(int k=0;k<d;++k){Acc z=static_cast<Acc>(query[k])-static_cast<Acc>(y[(int64_t)k*n+col]);
            if constexpr(KIND==0){if constexpr(WIDE)cost+=fabs(z);else cost+=fabsf(z);}
            else if constexpr(KIND==1){if constexpr(WIDE)cost=fmax(cost,fabs(z));else cost=fmaxf(cost,fabsf(z));}
            else cost+=z*z;
        }
        if constexpr(KIND==2){if constexpr(WIDE)cost=sqrt(cost);else cost=sqrtf(cost);}
        Candidate candidate={g[col]-static_cast<double>(cost),col};if(better(candidate,best))best=candidate;
        if constexpr(STATS){double c=cost;sum+=c*c;maximum=fmax(maximum,fabs(c));}
    }
    for(int offset=16;offset>0;offset/=2){Candidate other={__shfl_down_sync(0xffffffffu,best.value,offset),__shfl_down_sync(0xffffffffu,best.index,offset)};
        if(lane+offset<32 && better(other,best))best=other;
        if constexpr(STATS){sum+=__shfl_down_sync(0xffffffffu,sum,offset);maximum=fmax(maximum,__shfl_down_sync(0xffffffffu,maximum,offset));}
    }
    __shared__ Candidate warp_best[8];__shared__ double warp_sum[8],warp_max[8];
    if(lane==0){warp_best[warp]=best;if constexpr(STATS){warp_sum[warp]=sum;warp_max[warp]=maximum;}}
    __syncthreads();
    if(warp==0){
        best=lane<8?warp_best[lane]:Candidate{-INFINITY,-1};
        if constexpr(STATS){sum=lane<8?warp_sum[lane]:0;maximum=lane<8?warp_max[lane]:0;}
        for(int offset=16;offset>0;offset/=2){Candidate other={__shfl_down_sync(0xffffffffu,best.value,offset),__shfl_down_sync(0xffffffffu,best.index,offset)};
            if(lane+offset<32 && better(other,best))best=other;
            if constexpr(STATS){sum+=__shfl_down_sync(0xffffffffu,sum,offset);maximum=fmax(maximum,__shfl_down_sync(0xffffffffu,maximum,offset));}}
        if(lane==0){values[(int64_t)blockIdx.y*q+row]=-best.value;ids[(int64_t)blockIdx.y*q+row]=best.index;
            if constexpr(STATS){atomicAdd(stats,sum);auto address=reinterpret_cast<unsigned long long*>(stats+1);auto old=*address;
                while(__longlong_as_double(old)<maximum){auto assumed=old;old=atomicCAS(address,assumed,__double_as_longlong(maximum));if(old==assumed)break;}}}
    }
}
__global__ void merge_direct_top1(const double* input,const int64_t* indices,int q,int groups,double* values,int64_t* ids) {
    int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=q)return;
    Candidate best={-INFINITY,-1};
    for(int g=0;g<groups;++g){Candidate value={-input[(int64_t)g*q+row],(int32_t)indices[(int64_t)g*q+row]};if(better(value,best))best=value;}
    values[row]=-best.value;ids[row]=best.index;
}
std::vector<torch::Tensor> direct_top1(torch::Tensor x,torch::Tensor y,torch::Tensor g,int kind,int precision,bool statistics) {
    TORCH_CHECK(x.is_cuda() && y.device()==x.device() && g.device()==x.device() && x.scalar_type()==torch::kFloat32 && y.scalar_type()==x.scalar_type()
        && g.scalar_type()==torch::kFloat64 && x.is_contiguous() && y.is_contiguous() && g.is_contiguous(),"Expected contiguous norm coordinates and FP64 dual");
    TORCH_CHECK(x.dim()==2 && y.dim()==2 && x.size(0)>0 && y.size(0)>0 && x.size(1)>0 && x.size(1)<=32 && x.size(1)==y.size(1)
        && x.size(0)<INT_MAX && y.size(0)<INT_MAX && g.numel()==y.size(0) && kind>=0 && kind<=2 && (precision==0 || precision==1),"Invalid direct top-1 dimensions or precision");
    c10::cuda::CUDAGuard guard(x.device());int q=x.size(0),n=y.size(0),d=x.size(1);
    // CN: 数据库转置保证相邻lane连续读取；按SM数量补充分块并行度，不按维度调参。
    // EN: Transpose database coordinates for coalesced lane reads, splitting by SM count rather than dimension tuning.
    auto target=y.transpose(0,1).contiguous();
    auto properties=at::cuda::getCurrentDeviceProperties();
    int wanted=(2*properties->multiProcessorCount+q-1)/q;
    int groups=std::max(1,std::min(wanted,(n+1023)/1024)),width=((n+groups-1)/groups+31)/32*32;
    groups=(n+width-1)/width;
    auto values=torch::empty({groups,q},g.options()),ids=torch::empty({groups,q},g.options().dtype(torch::kInt64)),stats=torch::zeros({2},g.options());
    auto stream=at::cuda::getCurrentCUDAStream();
#define LAUNCH(C,P,S) direct_top1_kernel<C,P,S><<<dim3(q,groups),256,0,stream>>>(x.data_ptr<float>(),target.data_ptr<float>(),g.data_ptr<double>(),q,n,d,width,values.data_ptr<double>(),ids.data_ptr<int64_t>(),stats.data_ptr<double>())
#define TYPE(C,P) if(statistics){LAUNCH(C,P,true);}else{LAUNCH(C,P,false);}
#define KIND(P) if(kind==0){TYPE(0,P);}else if(kind==1){TYPE(1,P);}else{TYPE(2,P);}
    if(precision){KIND(true);}else{KIND(false);}
#undef KIND
#undef TYPE
#undef LAUNCH
    if(groups>1){auto out=torch::empty({q,1},g.options()),idx=torch::empty({q,1},ids.options());
        merge_direct_top1<<<(q+255)/256,256,0,stream>>>(values.data_ptr<double>(),ids.data_ptr<int64_t>(),q,groups,out.data_ptr<double>(),idx.data_ptr<int64_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();return {out,idx,stats};}
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {values.view({q,1}),ids.view({q,1}),stats};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("topk",&topk,pybind11::arg("x"),pybind11::arg("y"),pybind11::arg("g"),pybind11::arg("kind"),pybind11::arg("config"),pybind11::arg("matrix"),pybind11::arg("mode"),pybind11::arg("nx"),pybind11::arg("ny"),pybind11::arg("flag"),pybind11::arg("k"),pybind11::arg("collect_statistics")=false);m.def("direct_top1",&direct_top1);}
