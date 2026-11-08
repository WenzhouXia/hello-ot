#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cublas_v2.h>
#include <type_traits>

// CN: 每线程累积多个点对，沿特征分块复用坐标；不调用矩阵乘法。
// EN: Accumulate multiple pairs per thread and reuse feature tiles without matrix multiplication.
template<int RM, int RN, int F, int CHAINS, bool INF, typename Acc=float, bool L2=false>
__global__ void direct_norm_kernel(const float* __restrict__ x,
                                   const float* __restrict__ y,
                                   Acc* __restrict__ out,
                                   int q, int n, int d) {
    constexpr int M=16*RM, N=16*RN;
    const int tx=threadIdx.x%16, ty=threadIdx.x/16;
    const int r0=blockIdx.x*M, c0=blockIdx.y*N;
    __shared__ float sx[F][M+1];
    __shared__ float sy[F][N+1];
    Acc acc[CHAINS==0?2:CHAINS][RM][RN]={};
    for (int base=0; base<d; base+=F) {
        for (int p=threadIdx.x;p<M*F;p+=256) {
            const int r=p/F, k=p%F;
            sx[k][r]=(r0+r<q && base+k<d) ? x[(int64_t)(r0+r)*d+base+k]:0;
        }
        for (int p=threadIdx.x;p<N*F;p+=256) {
            const int c=p/F, k=p%F;
            sy[k][c]=(c0+c<n && base+k<d) ? y[(int64_t)(c0+c)*d+base+k]:0;
        }
        __syncthreads();
        if constexpr (CHAINS==0) {
            #pragma unroll
            for(int i=0;i<RM;++i) {
                #pragma unroll
                for(int j=0;j<RN;++j) acc[1][i][j]=0;
            }
        }
        #pragma unroll
        for (int k=0;k<F;++k) {
            float a[RM], b[RN];
            #pragma unroll
            for (int i=0;i<RM;++i) a[i]=sx[k][ty+i*16];
            #pragma unroll
            for (int j=0;j<RN;++j) b[j]=sy[k][tx+j*16];
            #pragma unroll
            for (int i=0;i<RM;++i) {
                #pragma unroll
                for (int j=0;j<RN;++j) {
                    // CN: FP64模式在相减之前提升坐标，保持旧norm精度语义。
                    // EN: Promote coordinates before subtraction to preserve legacy norm precision semantics.
                    const Acc delta=static_cast<Acc>(a[i])-static_cast<Acc>(b[j]);
                    const Acc v=L2?delta*delta:fabs(delta);
                    if constexpr (CHAINS==0) {
                        if (INF) acc[1][i][j]=fmax(acc[1][i][j],v);
                        else acc[1][i][j]+=v;
                    } else {
                        if (INF) acc[k%CHAINS][i][j]=fmax(acc[k%CHAINS][i][j],v);
                        else acc[k%CHAINS][i][j]+=v;
                    }
                }
            }
        }
        if constexpr (CHAINS==0) {
            #pragma unroll
            for(int i=0;i<RM;++i) {
                #pragma unroll
                for(int j=0;j<RN;++j)
                    acc[0][i][j]=INF ? fmax(acc[0][i][j],acc[1][i][j]):acc[0][i][j]+acc[1][i][j];
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i=0;i<RM;++i) {
        #pragma unroll
        for (int j=0;j<RN;++j) {
            Acc v=acc[0][i][j];
            #pragma unroll
            for (int k=1;k<CHAINS;++k)
                v=INF ? fmax(v,acc[k][i][j]):v+acc[k][i][j];
            const int r=r0+ty+i*16, c=c0+tx+j*16;
            if (r<q && c<n) out[(int64_t)r*n+c]=L2?sqrt(v):v;
        }
    }
}

// CN: FP32坐标常驻、FP64差分与累加，复用同一坐标分块和点对复用框架。
// EN: Keep FP32 coordinates resident with FP64 differences and accumulation, reusing coordinate and pair tiles.
torch::Tensor direct_norm_fp64(torch::Tensor x,torch::Tensor y,int kind,int variant) {
    TORCH_CHECK(x.is_cuda() && y.device()==x.device() && x.is_contiguous() && y.is_contiguous()
        && x.scalar_type()==torch::kFloat32 && y.scalar_type()==torch::kFloat32
        && x.dim()==2 && y.dim()==2 && x.size(1)==y.size(1) && x.size(0)>0 && y.size(0)>0 && x.size(1)>0,
        "Expected matching nonempty contiguous CUDA FP32 features");
    TORCH_CHECK(kind>=0 && kind<=2 && variant>=0 && variant<=2,"Invalid FP64 norm configuration");
    const c10::cuda::CUDAGuard guard(x.device());
    auto out=torch::empty({x.size(0),y.size(0)},x.options().dtype(torch::kFloat64));
    const int q=x.size(0),n=y.size(0),d=x.size(1);auto stream=at::cuda::getCurrentCUDAStream();
#define FP64_CALL(RM,RN,INF,L2) direct_norm_kernel<RM,RN,32,1,INF,double,L2><<<dim3((q+16*RM-1)/(16*RM),(n+16*RN-1)/(16*RN)),256,0,stream>>>(x.data_ptr<float>(),y.data_ptr<float>(),out.data_ptr<double>(),q,n,d)
#define FP64_GEOMETRY(INF,L2) if(variant==0){FP64_CALL(1,2,INF,L2);}else if(variant==1){FP64_CALL(2,2,INF,L2);}else{FP64_CALL(2,4,INF,L2);}
    if(kind==0){FP64_GEOMETRY(false,false);}else if(kind==1){FP64_GEOMETRY(true,false);}else{FP64_GEOMETRY(false,true);}
#undef FP64_GEOMETRY
#undef FP64_CALL
    C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

// CN: raw FP64使用真正的DGEMM，禁止通过FP32点积冒充FP64成本。
// EN: Use genuine DGEMM for FP64 raw scans rather than promoting FP32 dot products.
torch::Tensor dots_fp64(torch::Tensor x,torch::Tensor y,double scale) {
    TORCH_CHECK(x.is_cuda() && y.device()==x.device() && x.is_contiguous() && y.is_contiguous()
        && x.scalar_type()==torch::kFloat64 && y.scalar_type()==torch::kFloat64
        && x.dim()==2 && y.dim()==2 && x.size(1)==y.size(1),"Expected matching CUDA FP64 matrices");
    TORCH_CHECK(scale==1. || scale==2.,"dot_scale must be 1 or 2");
    const c10::cuda::CUDAGuard guard(x.device());
    auto out=torch::empty({x.size(0),y.size(0)},x.options());
    auto handle=at::cuda::getCurrentCUDABlasHandle();
    TORCH_CHECK(cublasSetStream(handle,at::cuda::getCurrentCUDAStream())==CUBLAS_STATUS_SUCCESS,"Cannot set cuBLAS stream");
    double alpha=scale,beta=0.;
    auto status=cublasDgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,y.size(0),x.size(0),x.size(1),
        &alpha,y.data_ptr<double>(),y.size(1),x.data_ptr<double>(),x.size(1),&beta,out.data_ptr<double>(),y.size(0));
    TORCH_CHECK(status==CUBLAS_STATUS_SUCCESS,"Generic DGEMM failed");
    C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

template<int RM,int RN,int F,int C>
void launch_direct(torch::Tensor x,torch::Tensor y,torch::Tensor out,int kind) {
    const int q=x.size(0),n=y.size(0),d=x.size(1);
    const dim3 grid((q+16*RM-1)/(16*RM),(n+16*RN-1)/(16*RN));
    const auto stream=at::cuda::getCurrentCUDAStream();
    if (kind==1) direct_norm_kernel<RM,RN,F,C,true><<<grid,256,0,stream>>>(x.data_ptr<float>(),y.data_ptr<float>(),out.data_ptr<float>(),q,n,d);
    else direct_norm_kernel<RM,RN,F,C,false><<<grid,256,0,stream>>>(x.data_ptr<float>(),y.data_ptr<float>(),out.data_ptr<float>(),q,n,d);
}

torch::Tensor direct_norm(torch::Tensor x,torch::Tensor y,int kind,int variant) {
    TORCH_CHECK(x.is_cuda() && y.is_cuda() && x.device()==y.device(),"Expected same CUDA device");
    TORCH_CHECK(x.scalar_type()==torch::kFloat32 && y.scalar_type()==torch::kFloat32 && x.is_contiguous() && y.is_contiguous(),"Expected contiguous FP32");
    TORCH_CHECK(x.dim()==2 && y.dim()==2 && x.size(1)==y.size(1),"Feature shapes differ");
    TORCH_CHECK(kind==0 || kind==1,"Only L1/Linf supported");
    TORCH_CHECK(x.size(0)>0 && y.size(0)>0 && x.size(1)>0,"Expected nonempty inputs");
    const c10::cuda::CUDAGuard guard(x.device());
    auto out=torch::empty({x.size(0),y.size(0)},x.options());
    switch(variant) {
        case 0: launch_direct<2,4,32,1>(x,y,out,kind);break;
        case 1: launch_direct<2,4,32,4>(x,y,out,kind);break;
        case 2: launch_direct<4,4,32,1>(x,y,out,kind);break;
        case 3: launch_direct<4,4,32,4>(x,y,out,kind);break;
        case 4: launch_direct<2,8,32,1>(x,y,out,kind);break;
        case 5: launch_direct<2,8,32,4>(x,y,out,kind);break;
        case 6: launch_direct<4,4,64,1>(x,y,out,kind);break;
        case 7: launch_direct<4,4,64,4>(x,y,out,kind);break;
        case 8: launch_direct<4,4,32,0>(x,y,out,kind);break;
        case 9: launch_direct<4,4,64,0>(x,y,out,kind);break;
        case 10: launch_direct<2,4,32,0>(x,y,out,kind);break;
        default: TORCH_CHECK(false,"Unknown direct-norm variant");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// CN: 每行用FP64累积范数，不物化完整FP64坐标副本。
// EN: Accumulate row norms in FP64 without materializing FP64 feature copies.
__global__ void norms_kernel(const float* x,double* out,int n,int d) {
    double sum=0;
    for(int k=threadIdx.x;k<d;k+=blockDim.x){double a=x[(int64_t)blockIdx.x*d+k];sum+=a*a;}
    for(int s=16;s;s/=2)sum+=__shfl_down_sync(0xffffffffu,sum,s);
    __shared__ double partial[8];
    if(threadIdx.x%32==0)partial[threadIdx.x/32]=sum;
    __syncthreads();
    if(threadIdx.x<32){sum=threadIdx.x<8?partial[threadIdx.x]:0;
        for(int s=16;s;s/=2)sum+=__shfl_down_sync(0xffffffffu,sum,s);
        if(threadIdx.x==0)out[blockIdx.x]=sum;}
}

torch::Tensor norms(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type()==torch::kFloat32 && x.dim()==2,"Expected CUDA FP32 matrix");
    const c10::cuda::CUDAGuard guard(x.device());
    auto out=torch::empty({x.size(0)},x.options().dtype(torch::kFloat64));
    norms_kernel<<<x.size(0),256,0,at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(),out.data_ptr<double>(),x.size(0),x.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

// CN: 明确舍入FP32成本，并检查所有点对的严重相消。
// EN: Explicitly round FP32 costs and check every pair for severe cancellation.
__global__ void finish_kernel(float* m,const double* nx,const double* ny,int* flag,int64_t count,int n,int d) {
    int64_t p=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(p<count){
        float scale=__fadd_rn((float)nx[p/n],(float)ny[p%n]);
        float c=fmaxf(__fsub_rn(scale,__fmul_rn(2.f,m[p])),0.f);
        if(c<=8.*d*1.1920928955078125e-7*scale)atomicOr(flag,1);
        m[p]=sqrtf(c);
    }
}

torch::Tensor finish(torch::Tensor m,torch::Tensor nx,torch::Tensor ny,int d) {
    const c10::cuda::CUDAGuard guard(m.device());
    auto flag=torch::zeros({1},m.options().dtype(torch::kInt32));
    finish_kernel<<<(m.numel()+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(m.data_ptr<float>(),nx.data_ptr<double>(),ny.data_ptr<double>(),flag.data_ptr<int>(),m.numel(),m.size(1),d);
    C10_CUDA_KERNEL_LAUNCH_CHECK();return flag;
}

// CN: 与原raw路径使用相同SGEMM算术，不受全局TF32选项影响；恢复句柄模式。
// EN: Match native raw SGEMM arithmetic independently of global TF32 options and restore handle mode.
torch::Tensor dots(torch::Tensor x,torch::Tensor y,double scale) {
    TORCH_CHECK(x.is_cuda() && y.device()==x.device() && x.is_contiguous() && y.is_contiguous()
        && x.scalar_type()==torch::kFloat32 && y.scalar_type()==torch::kFloat32
        && x.dim()==2 && y.dim()==2 && x.size(1)==y.size(1),"Expected matching CUDA FP32 matrices");
    TORCH_CHECK(scale==1. || scale==2.,"dot_scale must be 1 or 2");
    const c10::cuda::CUDAGuard guard(x.device());
    auto out=torch::empty({x.size(0),y.size(0)},x.options());
    auto handle=at::cuda::getCurrentCUDABlasHandle();
    TORCH_CHECK(cublasSetStream(handle,at::cuda::getCurrentCUDAStream())==CUBLAS_STATUS_SUCCESS,"Cannot set cuBLAS stream");
    cublasMath_t previous;
    TORCH_CHECK(cublasGetMathMode(handle,&previous)==CUBLAS_STATUS_SUCCESS,"Cannot read cuBLAS math mode");
    TORCH_CHECK(cublasSetMathMode(handle,CUBLAS_PEDANTIC_MATH)==CUBLAS_STATUS_SUCCESS,"Cannot set cuBLAS math mode");
    float alpha=scale,beta=0.f;
    auto status=cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,y.size(0),x.size(0),x.size(1),
        &alpha,y.data_ptr<float>(),y.size(1),x.data_ptr<float>(),x.size(1),&beta,out.data_ptr<float>(),y.size(0));
    auto restore=cublasSetMathMode(handle,previous);
    TORCH_CHECK(status==CUBLAS_STATUS_SUCCESS && restore==CUBLAS_STATUS_SUCCESS,"Generic SGEMM failed");
    return out;
}

#include <cub/warp/warp_merge_sort.cuh>

// CN: 沿用HELLO的32槽warp队列和比较规则，所有普通维度共用raw扫描。
// EN: Retain HELLO's 32-slot warp queue and comparator for raw scans at every ordinary dimension.
struct RawCandidate {double value;int32_t index;};
__device__ __forceinline__ bool raw_better(RawCandidate a,RawCandidate b) {
    return a.value>b.value || (a.value==b.value && a.index>=0 && (b.index<0 || a.index<b.index));
}
struct RawCompare {
    __device__ __forceinline__ bool operator()(const RawCandidate& a,const RawCandidate& b)const{return raw_better(a,b);}
};
template<typename Sort>
__device__ __forceinline__ void raw_merge(RawCandidate (&local)[2],int& count,RawCandidate* queue,RawCandidate* items,typename Sort::TempStorage& storage) {
    int lane=threadIdx.x%32;RawCandidate values[3]={queue[lane],local[0],local[1]};
    Sort(storage).Sort(values,RawCompare{});
    #pragma unroll
    for(int j=0;j<3;++j)items[lane*3+j]=values[j];
    __syncwarp();queue[lane]=items[lane];
    local[0]={-INFINITY,-1};local[1]={-INFINITY,-1};count=0;__syncwarp();
}

template<int K> __device__ __forceinline__ void raw_insert(RawCandidate (&best)[K],RawCandidate item) {
    #pragma unroll
    for(int rank=0;rank<K;++rank)if(raw_better(item,best[rank])){RawCandidate previous=best[rank];best[rank]=item;item=previous;}
}

template<int K,typename Dot> __global__ void raw_initial_kernel(const Dot* dots,const double* qb,const double* dbb,
        double* values,int32_t* indices,int q,int n,int db_offset) {
    int lane=threadIdx.x%32;
    if constexpr(K<=2) {
        int row=blockIdx.x,warp=threadIdx.x/32;
        RawCandidate best[K];
        #pragma unroll
        for(int rank=0;rank<K;++rank)best[rank]={-INFINITY,-1};
        if(db_offset && threadIdx.x==0){
            #pragma unroll
            for(int rank=0;rank<K;++rank)best[rank]={values[(int64_t)row*K+rank],indices[(int64_t)row*K+rank]};}
        double bias=qb[row];
        for(int col=threadIdx.x;col<n;col+=256){
            RawCandidate item={(static_cast<double>(dots[(int64_t)row*n+col])+bias)+dbb[col],col+db_offset};
            raw_insert(best,item);
        }
        for(int s=16;s;s/=2){RawCandidate other[K];
            #pragma unroll
            for(int rank=0;rank<K;++rank)other[rank]={__shfl_down_sync(0xffffffffu,best[rank].value,s),__shfl_down_sync(0xffffffffu,best[rank].index,s)};
            #pragma unroll
            for(int rank=0;rank<K;++rank)if(lane+s<32)raw_insert(best,other[rank]);}
        __shared__ RawCandidate partial[8][K];
        if(lane==0){
            #pragma unroll
            for(int rank=0;rank<K;++rank)partial[warp][rank]=best[rank];}
        __syncthreads();
        if(warp==0){
            #pragma unroll
            for(int rank=0;rank<K;++rank)best[rank]=lane<8?partial[lane][rank]:RawCandidate{-INFINITY,-1};
            for(int s=16;s;s/=2){RawCandidate other[K];
                #pragma unroll
                for(int rank=0;rank<K;++rank)other[rank]={__shfl_down_sync(0xffffffffu,best[rank].value,s),__shfl_down_sync(0xffffffffu,best[rank].index,s)};
                #pragma unroll
                for(int rank=0;rank<K;++rank)if(lane+s<32)raw_insert(best,other[rank]);}
            if(lane==0){
                #pragma unroll
                for(int rank=0;rank<K;++rank){values[(int64_t)row*K+rank]=best[rank].value;indices[(int64_t)row*K+rank]=best[rank].index;}}}
    } else {
        constexpr int W=4;
        int warp=threadIdx.x/32,row=blockIdx.x*W+warp;
        if(row>=q)return;
        using Sort=cub::WarpMergeSort<RawCandidate,3,32>;
        __shared__ typename Sort::TempStorage storage[W];
        __shared__ RawCandidate queues[W][32],items[W][96];
        RawCandidate* queue=queues[warp];
        queue[lane]=db_offset && lane<K?RawCandidate{values[(int64_t)row*K+lane],indices[(int64_t)row*K+lane]}:RawCandidate{-INFINITY,-1};
        __syncwarp();RawCandidate local[2]={{-INFINITY,-1},{-INFINITY,-1}};int count=0;
        double bias=qb[row];
        for(int base=0;base<n;base+=32){int col=base+lane;
            if(col<n){RawCandidate item={(static_cast<double>(dots[(int64_t)row*n+col])+bias)+dbb[col],col+db_offset};
                if(raw_better(item,queue[K-1])){local[1]=local[0];local[0]=item;++count;}}
            if(__any_sync(0xffffffffu,count==2))raw_merge<Sort>(local,count,queue,items[warp],storage[warp]);}
        raw_merge<Sort>(local,count,queue,items[warp],storage[warp]);
        if(lane<K){values[(int64_t)row*K+lane]=queue[lane].value;indices[(int64_t)row*K+lane]=queue[lane].index;}
    }
}

template<typename Dot> std::vector<torch::Tensor> raw_initial_t(torch::Tensor x,torch::Tensor y,torch::Tensor qb,torch::Tensor dbb,
        int k,int query_tile,int database_tile,double scale) {
    TORCH_CHECK(x.is_cuda() && y.device()==x.device() && qb.device()==x.device() && dbb.device()==x.device(),"Expected matching CUDA devices");
    TORCH_CHECK(x.dim()==2 && y.dim()==2 && x.size(1)==y.size(1) && x.size(0)>0 && y.size(0)>0 && x.size(1)>0,"Expected matching nonempty feature matrices");
    TORCH_CHECK(x.scalar_type()==c10::CppTypeToScalarType<Dot>::value && y.scalar_type()==x.scalar_type() && qb.scalar_type()==torch::kFloat64 && dbb.scalar_type()==torch::kFloat64,"Expected matching FP32/FP64 features and FP64 biases");
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous() && qb.is_contiguous() && dbb.is_contiguous(),"Expected contiguous raw inputs");
    TORCH_CHECK(qb.numel()==x.size(0) && dbb.numel()==y.size(0),"Bias shapes differ");
    TORCH_CHECK(x.size(0)<INT_MAX && y.size(0)<INT_MAX && x.size(1)<INT_MAX,"Raw scan dimensions exceed int32");
    TORCH_CHECK((k==1 || k==2 || k==4 || k==8 || k==16 || k==32) && query_tile>0 && database_tile>0 && (scale==1. || scale==2.),"Invalid raw scan configuration");
    const c10::cuda::CUDAGuard guard(x.device());
    int qt=std::min<int64_t>({x.size(0),query_tile,2048}),nt=std::min<int64_t>({y.size(0),database_tile,8192}),d=x.size(1);
    auto values=torch::empty({x.size(0),k},x.options().dtype(torch::kFloat64));
    auto indices=torch::empty({x.size(0),k},x.options().dtype(torch::kInt32));
    auto scratch=torch::empty({qt,nt},x.options());
    auto stream=at::cuda::getCurrentCUDAStream();auto handle=at::cuda::getCurrentCUDABlasHandle();
    TORCH_CHECK(cublasSetStream(handle,stream)==CUBLAS_STATUS_SUCCESS,"Cannot set cuBLAS stream");
    cublasMath_t previous;
    TORCH_CHECK(cublasGetMathMode(handle,&previous)==CUBLAS_STATUS_SUCCESS,"Cannot read cuBLAS math mode");
    TORCH_CHECK(cublasSetMathMode(handle,CUBLAS_PEDANTIC_MATH)==CUBLAS_STATUS_SUCCESS,"Cannot set cuBLAS math mode");
    try {
        Dot alpha=scale,beta=0;
        for(int64_t i=0;i<x.size(0);i+=qt)for(int64_t j=0;j<y.size(0);j+=nt){
            int q=std::min<int64_t>(qt,x.size(0)-i),n=std::min<int64_t>(nt,y.size(0)-j);
            cublasStatus_t status;
            if constexpr(std::is_same<Dot,double>::value)status=cublasDgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,n,q,d,&alpha,y.data_ptr<Dot>()+(int64_t)j*d,d,x.data_ptr<Dot>()+(int64_t)i*d,d,&beta,scratch.data_ptr<Dot>(),n);
            else status=cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,n,q,d,&alpha,y.data_ptr<Dot>()+(int64_t)j*d,d,x.data_ptr<Dot>()+(int64_t)i*d,d,&beta,scratch.data_ptr<Dot>(),n);
            TORCH_CHECK(status==CUBLAS_STATUS_SUCCESS,"Raw initial GEMM failed");
#define RAW(K) case K:raw_initial_kernel<K,Dot><<<K<=2?q:(q+3)/4,K<=2?256:128,0,stream>>>(scratch.data_ptr<Dot>(),qb.data_ptr<double>()+i,dbb.data_ptr<double>()+j,values.data_ptr<double>()+(int64_t)i*K,indices.data_ptr<int32_t>()+(int64_t)i*K,q,n,j);break
            switch(k){RAW(1);RAW(2);RAW(4);RAW(8);RAW(16);RAW(32);}
#undef RAW
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    } catch(...) {cublasSetMathMode(handle,previous);throw;}
    TORCH_CHECK(cublasSetMathMode(handle,previous)==CUBLAS_STATUS_SUCCESS,"Cannot restore cuBLAS math mode");
    return {values,indices};
}

// CN: 两种精度共用原生流式循环、候选队列和调用方分块限制。
// EN: Share native streaming loops, candidate queues and caller tile limits across both precisions.
std::vector<torch::Tensor> raw_initial(torch::Tensor x,torch::Tensor y,torch::Tensor qb,torch::Tensor dbb,
        int k,int query_tile,int database_tile,double scale) {
    if(x.scalar_type()==torch::kFloat64)return raw_initial_t<double>(x,y,qb,dbb,k,query_tile,database_tile,scale);
    return raw_initial_t<float>(x,y,qb,dbb,k,query_tile,database_tile,scale);
}

enum PerturbMode { kRank2=0,kIndexHash=1 };
// CN: 以下噪声公式及参数转换沿用旧 norm 扩展，保留显式舍入和方向约定。
// EN: Retain the legacy norm noise formulas and parameter conversion, including explicit rounding and direction conventions.
constexpr int64_t kHashPrime0 = 2003;
constexpr int64_t kHashPrime1 = 2011;
constexpr float kHashModulusF = 4028033.0f;
// CN: KeOps 把“除以 IntCst”改写成“乘以 RatCst((float)(1/prime))”的 fp32 常数，
//     复刻噪声时必须用同一常数才能逐位一致。
// EN: KeOps rewrites division by an IntCst as a multiply by the fp32 constant
//     (float)(1/prime); bitwise replication needs the same constants.
constexpr float kRecipHash0 = (float)(1.0 / 2003.0);
constexpr float kRecipHash1 = (float)(1.0 / 2011.0);

struct NoiseCoeffs {
    float sc0, tc0, off0;
    float sc1, tc1, off1;
};

// CN: 扰动的运行参数；sigma == 0 时关闭扰动，seed_term 为
//     (float)((double)seed * 0.001)，与 KeOps 的 fp32 标量嵌入一致。
// EN: Runtime perturbation parameters; sigma == 0 disables it. seed_term is
//     (float)((double)seed * 0.001), matching KeOps fp32 scalar embedding.
struct NoiseParams {
    int perturb_mode;
    double sigma;
    float seed_term;
    bool row_is_source;
    NoiseCoeffs coeffs;
};

__device__ __forceinline__ double sigma_noise(
        const NoiseParams& params,
        int64_t q_idx,
        int64_t d_idx) {
    if (params.perturb_mode == kRank2) {
        float src = static_cast<float>(q_idx);
        float tgt = static_cast<float>(d_idx);
        if (!params.row_is_source) {
            const float tmp = src;
            src = tgt;
            tgt = tmp;
        }
        // CN: 复刻 KeOps（pykeops 默认 --use_fast_math 编译）：
        //     z 的乘加被 NVRTC 收缩为 fmaf(tgt, 78.233, src*12.9898)，
        //     sin 解析为 __sinf，两者都是确定性硬件指令，可逐位一致。
        // EN: Replicate KeOps compiled with pykeops' default --use_fast_math:
        //     NVRTC contracts z's mul-add into fmaf(tgt, 78.233, src*12.9898)
        //     and sin resolves to __sinf; both are deterministic HW ops.
        const float z = __fadd_rn(fmaf(tgt, 78.233f, __fmul_rn(src, 12.9898f)), params.seed_term);
        return __fmul_rn(params.sigma, __fadd_rn(__fmul_rn(__sinf(z), 0.5f), 0.5f));
    }
    int64_t s0 = q_idx % kHashPrime0;
    int64_t t0 = d_idx % kHashPrime0;
    int64_t s1 = q_idx % kHashPrime1;
    int64_t t1 = d_idx % kHashPrime1;
    if (!params.row_is_source) {
        const int64_t tmp0 = s0;
        s0 = t0;
        t0 = tmp0;
        const int64_t tmp1 = s1;
        s1 = t1;
            t1 = tmp1;
        }
    // CN: 与 NumPy/Torch 共用 FP32 近似取模；显式舍入禁止 FMA 改变顺序。
    // EN: Share FP32 approximate remainders with NumPy/Torch; explicit rounding prevents FMA reordering.
    const float s0f = static_cast<float>(s0), t0f = static_cast<float>(t0);
    const float s1f = static_cast<float>(s1), t1f = static_cast<float>(t1);
    const float x0 = __fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(params.coeffs.sc0, s0f), __fmul_rn(params.coeffs.tc0, t0f)), __fmul_rn(s0f, t0f)), params.coeffs.off0);
    const float x1 = __fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(params.coeffs.sc1, s1f), __fmul_rn(params.coeffs.tc1, t1f)), __fmul_rn(s1f, t1f)), params.coeffs.off1);
    const float h0 = __fsub_rn(x0, __fmul_rn(2003.0f, floorf(__fmul_rn(kRecipHash0, x0))));
    const float h1 = __fsub_rn(x1, __fmul_rn(2011.0f, floorf(__fmul_rn(kRecipHash1, x1))));
    const float value = __fmul_rn(__fadd_rn(__fadd_rn(h0, __fmul_rn(2003.0f, h1)), 0.5f), (float)(1.0 / 4028033.0));
    return __dmul_rn(params.sigma, static_cast<double>(value));
}


NoiseParams make_noise_params(
        const torch::Tensor& query_global_index,
        const torch::Tensor& database_global_index,
        double sigma,
        int64_t seed,
        int64_t perturb_mode,
        bool row_is_source,
        const torch::Tensor& index_hash_coeffs) {
    NoiseParams params;
    params.perturb_mode = static_cast<int>(perturb_mode);
    params.sigma = sigma;
    params.seed_term = static_cast<float>(static_cast<double>(seed) * 0.001);
    params.row_is_source = row_is_source;
    params.coeffs = NoiseCoeffs{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (sigma != 0.0 && perturb_mode == kIndexHash) {
        TORCH_CHECK(index_hash_coeffs.defined() && index_hash_coeffs.numel() == 6, "index_hash_coeffs must have 6 elements");
        TORCH_CHECK(index_hash_coeffs.scalar_type() == torch::kInt64, "index_hash_coeffs must be int64");
        auto c = index_hash_coeffs.to(torch::kCPU).contiguous();
        const int64_t* p = c.data_ptr<int64_t>();
        params.coeffs = NoiseCoeffs{
                static_cast<float>(p[0]), static_cast<float>(p[1]), static_cast<float>(p[2]),
                static_cast<float>(p[3]), static_cast<float>(p[4]), static_cast<float>(p[5])};
    }
    (void)query_global_index;
    (void)database_global_index;
    return params;
}


template<typename Cost> __global__ void perturb_cost_kernel(const Cost* costs,double* output,int64_t count,int n,
        const int64_t* qi,const int64_t* di,NoiseParams noise) {
    for(int64_t p=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;p<count;p+=static_cast<int64_t>(gridDim.x)*blockDim.x){
        output[p]=static_cast<double>(costs[p])+sigma_noise(noise,qi[p/n],di[p%n]);
    }
}
torch::Tensor perturb_cost(torch::Tensor costs,torch::Tensor qi,torch::Tensor di,double sigma,
        int64_t seed,int mode,bool row_is_source,torch::Tensor coefficients) {
    TORCH_CHECK(costs.is_cuda() && costs.dim()==2 && costs.size(0)>0 && costs.size(1)>0 && costs.is_contiguous()
        && (costs.scalar_type()==torch::kFloat32 || costs.scalar_type()==torch::kFloat64),"Expected a contiguous CUDA FP32/FP64 cost tile");
    TORCH_CHECK(std::isfinite(sigma) && sigma>0 && (mode==0 || mode==1),"Invalid perturbation configuration");
    TORCH_CHECK(qi.device()==costs.device() && di.device()==costs.device() && qi.scalar_type()==torch::kInt64
        && di.scalar_type()==torch::kInt64 && qi.is_contiguous() && di.is_contiguous()
        && qi.numel()==costs.size(0) && di.numel()==costs.size(1),"Expected matching global int64 indices");
    c10::cuda::CUDAGuard guard(costs.device());
    auto noise=make_noise_params(qi,di,sigma,seed,mode,row_is_source,coefficients);
    auto output=torch::empty(costs.sizes(),costs.options().dtype(torch::kFloat64));
    int blocks=std::min<int64_t>((costs.numel()+255)/256,4*at::cuda::getCurrentDeviceProperties()->multiProcessorCount);
#define PERTURB(T) perturb_cost_kernel<T><<<blocks,256,0,at::cuda::getCurrentCUDAStream()>>>(costs.data_ptr<T>(),output.data_ptr<double>(),costs.numel(),costs.size(1),qi.data_ptr<int64_t>(),di.data_ptr<int64_t>(),noise)
    if(costs.scalar_type()==torch::kFloat32){PERTURB(float);}else{PERTURB(double);}
#undef PERTURB
    C10_CUDA_KERNEL_LAUNCH_CHECK();return output;
}

// CN: 只计算完整证书；不生成候选，也不分配行列部分候选。
// EN: Compute only the complete certificate, without candidate generation or partial-candidate buffers.
__device__ void certificate_max(double* target,double value) {
    auto address=reinterpret_cast<unsigned long long*>(target);auto old=*address;
    while(__longlong_as_double(old)<value){auto assumed=old;old=atomicCAS(address,assumed,__double_as_longlong(value));if(old==assumed)break;}
}
template<typename Cost,bool RAW=false> __global__ void certificate_kernel(const Cost* matrix,const double* f,const double* g,
        int64_t count,int n,bool compute_denominator,double* output,const double* nx=nullptr,const double* ny=nullptr) {
    double num=0,den=0,maximum=0,costmax=0;unsigned long long positive=0;
    for(int64_t p=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;p<count;p+=static_cast<int64_t>(gridDim.x)*blockDim.x){
        double cost=RAW?(nx[p/n]+ny[p%n])-static_cast<double>(matrix[p]):static_cast<double>(matrix[p]);
        double score=(f[p/n]+g[p%n])-cost;
        if(compute_denominator)den+=cost*cost;
        costmax=fmax(costmax,fabs(cost));
        if(score>0){num+=score*score;maximum=fmax(maximum,score);++positive;}
    }
    for(int s=16;s;s/=2){num+=__shfl_down_sync(0xffffffffu,num,s);den+=__shfl_down_sync(0xffffffffu,den,s);
        maximum=fmax(maximum,__shfl_down_sync(0xffffffffu,maximum,s));costmax=fmax(costmax,__shfl_down_sync(0xffffffffu,costmax,s));
        positive+=__shfl_down_sync(0xffffffffu,positive,s);}
    // CN: 先合并同一block的八个warp，每个block仅一次全局更新，降低统计原子操作竞争。
    // EN: Merge the block's eight warps before one global update per block, reducing statistics atomic contention.
    __shared__ double partial[4][8];
    __shared__ unsigned long long counts[8];
    const int lane=threadIdx.x%32,warp=threadIdx.x/32;
    if(lane==0){partial[0][warp]=num;partial[1][warp]=den;partial[2][warp]=maximum;partial[3][warp]=costmax;counts[warp]=positive;}
    __syncthreads();
    if(warp==0){
        num=lane<8?partial[0][lane]:0;den=lane<8?partial[1][lane]:0;
        maximum=lane<8?partial[2][lane]:0;costmax=lane<8?partial[3][lane]:0;positive=lane<8?counts[lane]:0;
        for(int s=4;s;s/=2){num+=__shfl_down_sync(0xffffffffu,num,s);den+=__shfl_down_sync(0xffffffffu,den,s);
            maximum=fmax(maximum,__shfl_down_sync(0xffffffffu,maximum,s));costmax=fmax(costmax,__shfl_down_sync(0xffffffffu,costmax,s));positive+=__shfl_down_sync(0xffffffffu,positive,s);}
        if(lane==0){atomicAdd(output,num);if(compute_denominator)atomicAdd(output+1,den);atomicAdd(output+4,static_cast<double>(positive));
            certificate_max(output+2,maximum);certificate_max(output+3,costmax);}
    }
}

// CN: 各cost与精度使用相同的归约并行度规则，按工作量封顶，避免只发射少量常驻warp。
// EN: Share one work-capped reduction parallelism rule across costs and precisions, avoiding too few resident warps.
int certificate_blocks(int64_t count) {
    return static_cast<int>(std::min<int64_t>((count+255)/256,8*at::cuda::getCurrentDeviceProperties()->multiProcessorCount));
}
torch::Tensor certificate(torch::Tensor matrix,torch::Tensor f,torch::Tensor g,bool compute_denominator) {
    TORCH_CHECK(matrix.is_cuda() && matrix.dim()==2 && matrix.size(0)>0 && matrix.size(1)>0 && matrix.is_contiguous()
        && (matrix.scalar_type()==torch::kFloat32 || matrix.scalar_type()==torch::kFloat64),"Expected a CUDA FP32/FP64 norm cost tile");
    TORCH_CHECK(f.device()==matrix.device() && g.device()==matrix.device() && f.scalar_type()==torch::kFloat64
        && g.scalar_type()==torch::kFloat64 && f.is_contiguous() && g.is_contiguous()
        && f.numel()==matrix.size(0) && g.numel()==matrix.size(1),"Expected matching FP64 potentials");
    c10::cuda::CUDAGuard guard(matrix.device());
    auto output=torch::zeros({5},f.options());int blocks=certificate_blocks(matrix.numel());
#define CERT(T) certificate_kernel<T><<<blocks,256,0,at::cuda::getCurrentCUDAStream()>>>(matrix.data_ptr<T>(),f.data_ptr<double>(),g.data_ptr<double>(),matrix.numel(),matrix.size(1),compute_denominator,output.data_ptr<double>())
    if(matrix.scalar_type()==torch::kFloat32){CERT(float);}else{CERT(double);}
#undef CERT
    C10_CUDA_KERNEL_LAUNCH_CHECK();return output;
}

torch::Tensor raw_certificate(torch::Tensor dots,torch::Tensor nx,torch::Tensor ny,torch::Tensor f,torch::Tensor g,bool denominator) {
    // CN: 复用完整证书归约，仅在寄存器内重建raw成本，不生成top-k或额外成本矩阵。
    // EN: Reuse full certificate reduction, reconstructing raw costs in registers without top-k or another cost matrix.
    TORCH_CHECK(dots.is_cuda() && dots.dim()==2 && dots.numel()>0 && dots.is_contiguous()
        && (dots.scalar_type()==torch::kFloat32 || dots.scalar_type()==torch::kFloat64),"Expected contiguous CUDA dot tile");
    for(auto pair:std::vector<std::pair<torch::Tensor,int64_t>>{{nx,dots.size(0)},{ny,dots.size(1)},{f,dots.size(0)},{g,dots.size(1)}})
        TORCH_CHECK(pair.first.device()==dots.device() && pair.first.scalar_type()==torch::kFloat64
            && pair.first.is_contiguous() && pair.first.numel()==pair.second,"Expected matching FP64 cost/potential vectors");
    c10::cuda::CUDAGuard guard(dots.device());
    auto output=torch::zeros({5},f.options());int blocks=certificate_blocks(dots.numel());
#define RAW_CERT(T) certificate_kernel<T,true><<<blocks,256,0,at::cuda::getCurrentCUDAStream()>>>(dots.data_ptr<T>(),f.data_ptr<double>(),g.data_ptr<double>(),dots.numel(),dots.size(1),denominator,output.data_ptr<double>(),nx.data_ptr<double>(),ny.data_ptr<double>())
    if(dots.scalar_type()==torch::kFloat32){RAW_CERT(float);}else{RAW_CERT(double);}
#undef RAW_CERT
    C10_CUDA_KERNEL_LAUNCH_CHECK();return output;
}

// CN: 只调整既有统计的返回顺序，不做候选选择或FP64到整数再转回的拷贝。
// EN: Reorder existing statistics without candidate selection or FP64-to-integer round trips.
__global__ void pack_raw_certificate(const double* input,double* output) {
    const int i=threadIdx.x;
    if(i<5)output[i]=input[i<2?i:(i==2?4:i-1)];
}

template<typename Dot> torch::Tensor raw_certificate_tiled_t(torch::Tensor x,torch::Tensor y,
        torch::Tensor nx,torch::Tensor ny,torch::Tensor f,torch::Tensor g,
        int qt,int nt,double scale,bool denominator,bool caller_layout) {
    const c10::cuda::CUDAGuard guard(x.device());
    const int q=x.size(0),n=y.size(0),d=x.size(1);
    qt=std::min(qt,q);nt=std::min(nt,n);
    const auto dtype=c10::CppTypeToScalarType<Dot>::value;
    torch::Tensor scratch=torch::empty({qt,nt},x.options().dtype(dtype));
    auto output=torch::zeros({5},f.options());
    nx=nx.contiguous();ny=ny.contiguous();f=f.contiguous();g=g.contiguous();
    const auto stream=at::cuda::getCurrentCUDAStream();
    auto handle=at::cuda::getCurrentCUDABlasHandle();
    TORCH_CHECK(cublasSetStream(handle,stream)==CUBLAS_STATUS_SUCCESS,"Cannot set cuBLAS stream");
    cublasMath_t previous;
    TORCH_CHECK(cublasGetMathMode(handle,&previous)==CUBLAS_STATUS_SUCCESS,"Cannot read cuBLAS math mode");
    TORCH_CHECK(cublasSetMathMode(handle,CUBLAS_PEDANTIC_MATH)==CUBLAS_STATUS_SUCCESS,"Cannot set cuBLAS math mode");
    const Dot alpha=scale,beta=0;
    try {
        for(int64_t i=0;i<q;i+=qt){
            const int cq=std::min<int64_t>(qt,q-i);
            torch::Tensor xx=x.narrow(0,i,cq).contiguous().to(dtype);
            for(int64_t j=0;j<n;j+=nt){
                const int cn=std::min<int64_t>(nt,n-j);
                torch::Tensor yy=y.narrow(0,j,cn).contiguous().to(dtype);
                cublasStatus_t status;
                if constexpr(std::is_same<Dot,double>::value)status=cublasDgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,cn,cq,d,&alpha,yy.data_ptr<Dot>(),d,xx.data_ptr<Dot>(),d,&beta,scratch.data_ptr<Dot>(),cn);
                else status=cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,cn,cq,d,&alpha,yy.data_ptr<Dot>(),d,xx.data_ptr<Dot>(),d,&beta,scratch.data_ptr<Dot>(),cn);
                TORCH_CHECK(status==CUBLAS_STATUS_SUCCESS,"Raw certificate GEMM failed");
                // CN: 所有分块共用统计输出和点积缓冲，原cost/score的FP64结合顺序不变。
                // EN: Share statistics and dot scratch across tiles, preserving FP64 cost/score association.
                const int blocks=certificate_blocks(static_cast<int64_t>(cq)*cn);
                certificate_kernel<Dot,true><<<blocks,256,0,stream>>>(scratch.data_ptr<Dot>(),f.data_ptr<double>()+i,g.data_ptr<double>()+j,
                    static_cast<int64_t>(cq)*cn,cn,denominator,output.data_ptr<double>(),nx.data_ptr<double>()+i,ny.data_ptr<double>()+j);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
        }
    } catch(...) {cublasSetMathMode(handle,previous);throw;}
    TORCH_CHECK(cublasSetMathMode(handle,previous)==CUBLAS_STATUS_SUCCESS,"Cannot restore cuBLAS math mode");
    if(caller_layout){
        auto packed=torch::empty_like(output);
        pack_raw_certificate<<<1,32,0,stream>>>(output.data_ptr<double>(),packed.data_ptr<double>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();return packed;
    }
    return output;
}

torch::Tensor raw_certificate_tiled(torch::Tensor x,torch::Tensor y,torch::Tensor nx,torch::Tensor ny,
        torch::Tensor f,torch::Tensor g,int qt,int nt,double scale,bool denominator,bool fp64,bool caller_layout) {
    TORCH_CHECK(x.is_cuda() && y.device()==x.device() && x.dim()==2 && y.dim()==2 && x.size(1)==y.size(1)
        && x.size(0)>0 && y.size(0)>0 && x.size(1)>0 && x.size(0)<2147483648LL && y.size(0)<2147483648LL && x.size(1)<2147483648LL
        && y.scalar_type()==x.scalar_type() && (x.scalar_type()==torch::kFloat32 || x.scalar_type()==torch::kFloat64),"Expected matching nonempty CUDA features");
    TORCH_CHECK(qt>0 && nt>0 && (scale==1. || scale==2.),"Invalid raw certificate tile or scale");
    for(auto pair:std::vector<std::pair<torch::Tensor,int64_t>>{{nx,x.size(0)},{ny,y.size(0)},{f,x.size(0)},{g,y.size(0)}})
        TORCH_CHECK(pair.first.device()==x.device() && pair.first.scalar_type()==torch::kFloat64 && pair.first.numel()==pair.second,"Expected matching FP64 cost and dual vectors");
    if(fp64 || x.scalar_type()==torch::kFloat64)return raw_certificate_tiled_t<double>(x,y,nx,ny,f,g,qt,nt,scale,denominator,caller_layout);
    return raw_certificate_tiled_t<float>(x,y,nx,ny,f,g,qt,nt,scale,denominator,caller_layout);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("direct_norm",&direct_norm);m.def("norms",&norms);m.def("finish",&finish);m.def("dots",&dots);m.def("raw_initial",&raw_initial);
    m.def("perturb_cost",&perturb_cost);m.def("certificate",&certificate);m.def("raw_certificate",&raw_certificate);
    m.def("direct_norm_fp64",&direct_norm_fp64);m.def("dots_fp64",&dots_fp64);
    m.def("raw_certificate_tiled",&raw_certificate_tiled);
}
