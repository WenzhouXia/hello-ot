#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/warp/warp_merge_sort.cuh>
#include <limits>

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
    using Sort=cub::WarpMergeSort<Candidate,3,32>;
    __shared__ typename Sort::TempStorage storage[W];
    __shared__ Candidate queues[W][32],items[W][96];
    __shared__ float sq[MODE?1:F*(W+1)],sd[MODE?1:F*(T+1)];
    int lane=threadIdx.x%32,warp=threadIdx.x/32,row=blockIdx.x*W+warp,dim=FIXED?FIXED:d;
    Candidate* queue=queues[warp];Candidate* merged=items[warp];
    queue[lane]={-INFINITY,-1};__syncwarp();
    Candidate local[2]={{-INFINITY,-1},{-INFINITY,-1}};int count=0;
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
                if(better(candidate,queue[k-1])){local[1]=local[0];local[0]=candidate;++count;}
            }
            if(__any_sync(0xffffffffu,count==2))merge_queue<Sort>(local,count,queue,merged,storage[warp]);
        }
    }
    merge_queue<Sort>(local,count,queue,merged,storage[warp]);
    if constexpr(STATS){
        for(int offset=16;offset>0;offset/=2){fixed_sum+=__shfl_down_sync(0xffffffffu,fixed_sum,offset);fixed_max=fmax(fixed_max,__shfl_down_sync(0xffffffffu,fixed_max,offset));}
        if(row<q && lane==0){atomicAdd(statistics,fixed_sum);auto address=reinterpret_cast<unsigned long long*>(statistics+1);auto old=*address;
            while(__longlong_as_double(old)<fixed_max){auto assumed=old;old=atomicCAS(address,assumed,__double_as_longlong(fixed_max));if(old==assumed)break;}}
    }
    if constexpr(MODE==2 || MODE==3)if(__any_sync(0xffffffffu,ambiguous) && lane==0)atomicOr(flag,1);
    if(row<q && lane<k){values[(int64_t)row*k+lane]=-queue[lane].value;indices[(int64_t)row*k+lane]=queue[lane].index;}
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
    TORCH_CHECK(y.size(0)>0 && (k==1 || k==2 || k==4 || k==8 || k==16 || k==32) && config>=0 && config<4 && kind>=0 && kind<4,"Invalid queue configuration");
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
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("topk",&topk,pybind11::arg("x"),pybind11::arg("y"),pybind11::arg("g"),pybind11::arg("kind"),pybind11::arg("config"),pybind11::arg("matrix"),pybind11::arg("mode"),pybind11::arg("nx"),pybind11::arg("ny"),pybind11::arg("flag"),pybind11::arg("k"),pybind11::arg("collect_statistics")=false);}
