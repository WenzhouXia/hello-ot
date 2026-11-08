#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <limits>
#include <cub/warp/warp_merge_sort.cuh>
#include <cublas_v2.h>
struct Candidate {double value;int32_t index;};
__device__ __forceinline__ bool better(Candidate a,Candidate b) {
    return a.value>b.value || (a.value==b.value && a.index>=0 && (b.index<0 || a.index<b.index));
}
struct Compare {
    __device__ __forceinline__ bool operator()(const Candidate& a,const Candidate& b)const{return better(a,b);}
};
template<typename Sort>
__device__ __forceinline__ void merge_queue(Candidate (&thread_queue)[2],int& count,Candidate* queue,typename Sort::TempStorage& storage) {
    int lane=threadIdx.x%32;
    Candidate local[3]={queue[lane],thread_queue[0],thread_queue[1]};
    Sort(storage).Sort(local,Compare{});
    Candidate selected;
    #pragma unroll
    for(int j=0;j<3;++j){Candidate item={__shfl_sync(0xffffffffu,local[j].value,lane/3),__shfl_sync(0xffffffffu,local[j].index,lane/3)};
        if(lane%3==j)selected=item;}
    queue[lane]=selected;
    thread_queue[0]={-INFINITY,-1};thread_queue[1]={-INFINITY,-1};count=0;__syncwarp();
}


// CN: 普通维度运行时分块，只有2/3D允许固定维度；FP64候选及完整统计。
// EN: Tile ordinary dimensions at runtime, specializing only 2/3D; retain FP64 candidates and complete statistics.
template<int K> __device__ void init(double (&v)[K],int64_t (&i)[K]) {
    #pragma unroll
    for(int j=0;j<K;++j){v[j]=-INFINITY;i[j]=INT64_MAX;}
}
// CN: 保留FP64值，用IEEE保序键比较；零同分及小索引规则与原生一致。
// EN: Retain FP64 values with IEEE order keys, matching native signed-zero and smaller-index tie rules.
__device__ bool better(double a,int64_t ai,double b,int64_t bi) {
#if __CUDA_ARCH__ == 800 || __CUDA_ARCH__ == 900
    // CN: A100/H100具备高FP64吞吐，直接比较避免重复构造IEEE排序键。
    // EN: A100/H100 have high FP64 throughput; direct comparisons avoid rebuilding IEEE order keys.
    return a>b || (a==b && ai<bi);
#else
    unsigned long long x=__double_as_longlong(a),y=__double_as_longlong(b);
    constexpr unsigned long long sign=0x8000000000000000ULL,mag=0x7fffffffffffffffULL;
    if((x&mag)>0x7ff0000000000000ULL || (y&mag)>0x7ff0000000000000ULL)return false;
    if((x&mag)==0)x=0;if((y&mag)==0)y=0;
    x=(x&sign)?~x:x^sign;y=(y&sign)?~y:y^sign;
    return x>y || (x==y && ai<bi);
#endif
}
template<int K> __device__ void put(double (&v)[K],int64_t (&i)[K],double a,int64_t b) {
    if(!better(a,b,v[K-1],i[K-1]))return;
    int p=K-1;
    #pragma unroll
    for(int j=K-2;j>=0;--j)if(better(a,b,v[j],i[j]))p=j;
    #pragma unroll
    for(int j=K-1;j>0;--j)if(j>p){v[j]=v[j-1];i[j]=i[j-1];}
    v[p]=a;i[p]=b;
}
__device__ void max_atomic(double* p,double v) {
    auto a=reinterpret_cast<unsigned long long*>(p);auto old=*a;
    while(__longlong_as_double(old)<v){auto assumed=old;old=atomicCAS(a,assumed,__double_as_longlong(v));if(old==assumed)break;}
}
template<int KIND> __device__ float add_distance(float a,float x,float y) {
    float z=x-y;
    if constexpr(KIND==0)return a+fabsf(z);
    else if constexpr(KIND==1)return fmaxf(a,fabsf(z));
    else return a+z*z;
}
template<int K,int R,int T,int KIND,bool MATRIX,int FIXED=0,bool RAW=false>
__global__ __launch_bounds__(R*32) void scan_kernel(const float* x,const float* y,const double* f,const double* g,
        const float* matrix,const double* nx,const double* ny,const double* qb,const double* dbb,int q,int n,int d,int db_tile,double* rv,int64_t* ri,double* cv,int64_t* ci,double* stats) {
    constexpr int F=FIXED?FIXED:32,C=T/32;
    const int lane=threadIdx.x%32,warp=threadIdx.x/32,row=blockIdx.x*R+warp;
    const int dim=FIXED ? FIXED:d;
    __shared__ float sq[MATRIX?1:F*(R+1)],sd[MATRIX?1:F*(T+1)];
    __shared__ double scores[R*T];
    double values[K<=2?K:1];int64_t indices[K<=2?K:1];init(values,indices);
    using Sort=cub::WarpMergeSort<Candidate,3,32>;
    __shared__ typename Sort::TempStorage storage[K>2?R:1];
    __shared__ Candidate queues[K>2?R:1][32];
    Candidate* queue=queues[K>2?warp:0];Candidate local[2]={{-INFINITY,-1},{-INFINITY,-1}};int count=0;
    if constexpr(K>2){queue[lane]={-INFINITY,-1};__syncwarp();}
    double numerator=0,denominator=0,maximum=0,costmax=0;unsigned long long positive=0;
    for(int c0=blockIdx.y*db_tile;c0<min(n,(int)(blockIdx.y+1)*db_tile);c0+=T) {
        float a[C]={};
        if constexpr(!MATRIX) {
            for(int base=0;base<dim;base+=F) {
                for(int p=threadIdx.x;p<R*F;p+=R*32){int r=p/F,k=p%F;int rr=blockIdx.x*R+r;sq[k*(R+1)+r]=rr<q && base+k<dim?x[(int64_t)rr*dim+base+k]:0;}
                for(int p=threadIdx.x;p<T*F;p+=R*32){int c=p/F,k=p%F;sd[k*(T+1)+c]=c0+c<n && base+k<dim?y[(int64_t)(c0+c)*dim+base+k]:0;}
                __syncthreads();
                float partial[C]={};
                if(row<q) {
                    for(int k=0;k<(FIXED ? FIXED:min(F,dim-base));++k) {
                        float v=sq[k*(R+1)+warp];
                        #pragma unroll
                        for(int j=0;j<C;++j)partial[j]=add_distance<KIND>(partial[j],v,sd[k*(T+1)+lane+j*32]);
                    }
                }
                #pragma unroll
                for(int j=0;j<C;++j)a[j]=KIND==1?fmaxf(a[j],partial[j]):a[j]+partial[j];
                __syncthreads();
            }
        }
        #pragma unroll
        for(int j=0;j<C;++j) {
            const int col=c0+lane+j*32;
            double score=-INFINITY;
            if(row<q && col<n) {
double distance=MATRIX ? (KIND==4?reinterpret_cast<const double*>(matrix)[(int64_t)row*n+col]:matrix[(int64_t)row*n+col]):a[j];
                if constexpr(!MATRIX && KIND==2)distance=sqrtf(distance);
double cost=distance;score=KIND==4?(f[row]+g[col])-cost:(g[col]-cost)+f[row];
                if constexpr(RAW) {
                    cost=(nx[row]+ny[col])-static_cast<double>(distance);
                    score=(static_cast<double>(distance)+qb[row])+dbb[col];
                }
                if constexpr(K<=2)put(values,indices,score,col);
                else if(better(Candidate{score,col},queue[K-1])){local[1]=local[0];local[0]={score,col};++count;}
                if constexpr(true) {
                    denominator+=cost*cost;costmax=fmax(costmax,fabs(cost));
                    if(score>0){numerator+=score*score;maximum=fmax(maximum,score);++positive;}
                }
            }
            if constexpr(true)scores[warp*T+lane+j*32]=score;
            if constexpr(K>2)if(__any_sync(0xffffffffu,count==2))merge_queue<Sort>(local,count,queue,storage[warp]);
        }
        if constexpr(true) {
            __syncthreads();
            for(int col=threadIdx.x;col<T && c0+col<n;col+=R*32) {
                constexpr int P=K<R?K:R;double v[P];int64_t idx[P];init(v,idx);
                for(int r=0;r<R;++r)if(blockIdx.x*R+r<q)put(v,idx,scores[r*T+col],blockIdx.x*R+r);
                int64_t offset=((int64_t)blockIdx.x*n+c0+col)*P;
                for(int j=0;j<P;++j){cv[offset+j]=v[j];ci[offset+j]=idx[j];}
            }
            __syncthreads();
        }
    }
    if constexpr(K>2)merge_queue<Sort>(local,count,queue,storage[warp]);
    if constexpr(K>2)if(row<q && lane<K){rv[((int64_t)blockIdx.y*q+row)*K+lane]=queue[lane].value;ri[((int64_t)blockIdx.y*q+row)*K+lane]=queue[lane].index;}
    for(int offset=16;offset>0;offset/=2) {
        if constexpr(K<=2) {
        double other[K];int64_t ids[K];
        #pragma unroll
        for(int j=0;j<K;++j){other[j]=__shfl_down_sync(0xffffffffu,values[j],offset);ids[j]=__shfl_down_sync(0xffffffffu,indices[j],offset);}
        if(lane+offset<32) {
            #pragma unroll
            for(int j=0;j<K;++j)put(values,indices,other[j],ids[j]);
        }
        }
        if constexpr(true) {
            numerator+=__shfl_down_sync(0xffffffffu,numerator,offset);denominator+=__shfl_down_sync(0xffffffffu,denominator,offset);
            maximum=fmax(maximum,__shfl_down_sync(0xffffffffu,maximum,offset));costmax=fmax(costmax,__shfl_down_sync(0xffffffffu,costmax,offset));positive+=__shfl_down_sync(0xffffffffu,positive,offset);
        }
    }
    if(row<q && lane==0) {
        #pragma unroll
        for(int j=0;j<(K<=2?K:0);++j){rv[((int64_t)blockIdx.y*q+row)*K+j]=values[j];ri[((int64_t)blockIdx.y*q+row)*K+j]=indices[j];}
        if constexpr(true){atomicAdd(stats,numerator);atomicAdd(stats+1,denominator);max_atomic(stats+2,maximum);max_atomic(stats+3,costmax);atomicAdd(stats+4,(double)positive);}
    }
}


template<int K,int P=K> __global__ void reduce_columns(const double* v,const int64_t* idx,int tiles,int n,double* out,int64_t* ids) {
    static_assert(K<=2 && P==K,"Small bucket reduction only");
    int lane=threadIdx.x%32,col=blockIdx.x*8+threadIdx.x/32;if(col>=n)return;
    double a[K];int64_t b[K];init(a,b);
    for(int t=lane;t<tiles;t+=32)for(int j=0;j<K;++j){int64_t p=((int64_t)t*n+col)*K+j;if(idx[p]>=0)put(a,b,v[p],idx[p]);}
    for(int offset=16;offset>0;offset/=2){double value[K];int64_t index[K];
        for(int j=0;j<K;++j){value[j]=__shfl_down_sync(0xffffffffu,a[j],offset);index[j]=__shfl_down_sync(0xffffffffu,b[j],offset);}
        if(lane+offset<32)for(int j=0;j<K;++j)put(a,b,value[j],index[j]);}
    if(lane==0)for(int j=0;j<K;++j){out[(int64_t)col*K+j]=a[j];ids[(int64_t)col*K+j]=b[j];}
}
#include "matrix_selection.cuh"
template<int K,int KIND,bool MATRIX,bool RAW>
std::vector<torch::Tensor> run(torch::Tensor x,torch::Tensor y,torch::Tensor f,torch::Tensor g,torch::Tensor matrix,
        torch::Tensor nx,torch::Tensor ny,torch::Tensor qb,torch::Tensor dbb) {
    if constexpr(MATRIX){
        if constexpr(KIND==4)return matrix_run<K,double,RAW>(x,y,f,g,matrix,nx,ny,qb,dbb);
        return matrix_run<K,float,RAW>(x,y,f,g,matrix,nx,ny,qb,dbb);
    }
    constexpr int R=K<=2?16:(MATRIX?8:4),T=128,P=K<R?K:R;int q=x.size(0),n=y.size(0),blocks=(q+R-1)/R;
    int wanted=std::max(1,(2*at::cuda::getCurrentDeviceProperties()->multiProcessorCount+blocks-1)/blocks);
    // CN: 每个warp至少扫描1024列，摊薄小矩形的统计原子操作及行合并开销。
    // EN: Scan at least 1024 columns per warp to amortize statistic atomics and row merges on small rectangles.
    int db_tile=std::max(1024,((n+wanted-1)/wanted+T-1)/T*T),db_blocks=(n+db_tile-1)/db_tile;
    auto opts=x.options().dtype(torch::kFloat64),ints=x.options().dtype(torch::kInt64);
    auto rv=torch::empty({db_blocks,q,K},opts),ri=torch::empty({db_blocks,q,K},ints),pv=torch::empty({blocks,n,P},opts),pi=torch::empty({blocks,n,P},ints),stats=torch::zeros({5},opts);
#define CALL(D) scan_kernel<K,R,T,KIND,MATRIX,D,RAW><<<dim3(blocks,db_blocks),R*32,0,at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const float*>(x.data_ptr()),reinterpret_cast<const float*>(y.data_ptr()),f.data_ptr<double>(),g.data_ptr<double>(),MATRIX?reinterpret_cast<const float*>(matrix.data_ptr()):nullptr,nx.data_ptr<double>(),ny.data_ptr<double>(),qb.data_ptr<double>(),dbb.data_ptr<double>(),q,n,x.size(1),db_tile,rv.data_ptr<double>(),ri.data_ptr<int64_t>(),pv.data_ptr<double>(),pi.data_ptr<int64_t>(),stats.data_ptr<double>())
    if constexpr(MATRIX){CALL(0);}else if(x.size(1)==2){CALL(2);}else if(x.size(1)==3){CALL(3);}else{CALL(0);}
#undef CALL
    auto cv=torch::empty({n,K},opts),ci=torch::empty({n,K},ints);
    reduce_columns<K,P><<<(n+7)/8,256,0,at::cuda::getCurrentCUDAStream()>>>(pv.data_ptr<double>(),pi.data_ptr<int64_t>(),blocks,n,cv.data_ptr<double>(),ci.data_ptr<int64_t>());
    if(db_blocks>1){auto rowv=torch::empty({q,K},opts),rowi=torch::empty({q,K},ints);
        reduce_columns<K><<<(q+7)/8,256,0,at::cuda::getCurrentCUDAStream()>>>(rv.data_ptr<double>(),ri.data_ptr<int64_t>(),db_blocks,q,rowv.data_ptr<double>(),rowi.data_ptr<int64_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();return {rowv,rowi,cv,ci,stats};}
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {rv.view({q,K}),ri.view({q,K}),cv,ci,stats};
}
template<int K> std::vector<torch::Tensor> metric(torch::Tensor x,torch::Tensor y,torch::Tensor f,torch::Tensor g,int kind,torch::Tensor matrix,bool raw,torch::Tensor nx,torch::Tensor ny,torch::Tensor qb,torch::Tensor dbb) {
#define RUN(C,M,A) return run<K,C,M,A>(x,y,f,g,matrix,nx,ny,qb,dbb)
    if(raw && matrix.scalar_type()==torch::kFloat64){RUN(4,true,true);}else if(raw){RUN(0,true,true);}else if(matrix.numel() && matrix.scalar_type()==torch::kFloat64){RUN(4,true,false);}else if(matrix.numel()){RUN(0,true,false);}else if(kind==0){RUN(0,false,false);}else if(kind==1){RUN(1,false,false);}else if(kind==2){RUN(2,false,false);}else{RUN(3,false,false);}
#undef RUN
}
std::vector<torch::Tensor> scan(torch::Tensor x,torch::Tensor y,torch::Tensor f,torch::Tensor g,int kind,int k,torch::Tensor matrix,bool raw,torch::Tensor nx,torch::Tensor ny,torch::Tensor qb,torch::Tensor dbb) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && y.is_contiguous() && ((x.scalar_type()==torch::kFloat32 && y.scalar_type()==torch::kFloat32) || (raw && matrix.numel() && matrix.scalar_type()==torch::kFloat64 && x.scalar_type()==torch::kFloat64 && y.scalar_type()==torch::kFloat64)),"CUDA contiguous FP32 features required");
    c10::cuda::CUDAGuard guard(x.device());
#define CASE(K) case K:return metric<K>(x,y,f,g,kind,matrix,raw,nx,ny,qb,dbb)
    switch(k){CASE(1);CASE(2);default:TORCH_CHECK(false,"Invalid bucket");}
#undef CASE
}
template<int K> std::vector<torch::Tensor> merge_k(torch::Tensor values,torch::Tensor indices) {
    auto v=torch::empty({values.size(1),K},values.options());auto i=torch::empty({values.size(1),K},indices.options());
    reduce_columns<K><<<(values.size(1)+7)/8,256,0,at::cuda::getCurrentCUDAStream()>>>(values.data_ptr<double>(),indices.data_ptr<int64_t>(),values.size(0),values.size(1),v.data_ptr<double>(),i.data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {v,i};
}
std::vector<torch::Tensor> merge(torch::Tensor v,torch::Tensor i) {
#define CASE(K) case K:return merge_k<K>(v,i)
    switch(v.size(2)){CASE(1);CASE(2);default:TORCH_CHECK(false,"Invalid bucket");}
#undef CASE
}
// CN: 小矩形在一个原生调用中完成点积、通用选择和ABI转换，避免Python逐项整理统计。
// EN: Complete dots, generic selection and ABI conversion in one native call for small rectangles.
__global__ void raw_outputs(const int64_t* rows,int32_t* row_indices,int row_count,
        const int64_t* cols,int32_t* col_indices,int col_count,const double* input,double* output,bool enabled,bool linf) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<row_count)row_indices[i]=rows[i];
    if(i<col_count)col_indices[i]=cols[i];
    if(i==0){output[0]=enabled?input[0]:0;output[1]=enabled?input[1]:0;
        output[2]=enabled?input[4]:0;output[3]=enabled && linf?input[3]:0;}
}
std::vector<torch::Tensor> raw_scan(torch::Tensor x,torch::Tensor y,torch::Tensor nx,torch::Tensor ny,
        torch::Tensor qb,torch::Tensor dbb,int k,double scale,bool stats_enabled,bool linf) {
    TORCH_CHECK(x.is_cuda() && y.device()==x.device() && x.dim()==2 && y.dim()==2 && x.size(1)==y.size(1)
        && x.size(0)>0 && y.size(0)>0 && x.size(1)>0 && x.size(0)<INT_MAX && y.size(0)<INT_MAX && x.size(1)<INT_MAX
        && (x.scalar_type()==torch::kFloat32 || x.scalar_type()==torch::kFloat64) && y.scalar_type()==x.scalar_type()
        && x.is_contiguous() && y.is_contiguous(),"Expected matching contiguous CUDA FP32 feature matrices");
    TORCH_CHECK((k==1 || k==2) && (scale==1. || scale==2.),"Invalid raw pricing configuration");
    for(auto pair:{std::make_pair(nx,x.size(0)),std::make_pair(ny,y.size(0)),std::make_pair(qb,x.size(0)),std::make_pair(dbb,y.size(0))})
        TORCH_CHECK(pair.first.device()==x.device() && pair.first.scalar_type()==torch::kFloat64 && pair.first.is_contiguous()
            && pair.first.numel()==pair.second,"Expected matching contiguous CUDA FP64 costs and biases");
    c10::cuda::CUDAGuard guard(x.device());
    auto matrix=torch::empty({x.size(0),y.size(0)},x.options());
    auto stream=at::cuda::getCurrentCUDAStream();auto handle=at::cuda::getCurrentCUDABlasHandle();cublasMath_t previous;
    TORCH_CHECK(cublasSetStream(handle,stream)==CUBLAS_STATUS_SUCCESS && cublasGetMathMode(handle,&previous)==CUBLAS_STATUS_SUCCESS,"Cannot initialize cuBLAS");
    TORCH_CHECK(cublasSetMathMode(handle,CUBLAS_PEDANTIC_MATH)==CUBLAS_STATUS_SUCCESS,"Cannot set cuBLAS math mode");
    cublasStatus_t status;
    if(x.scalar_type()==torch::kFloat64){
        double alpha=scale,beta=0;
        status=cublasDgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,y.size(0),x.size(0),x.size(1),&alpha,
            y.data_ptr<double>(),y.size(1),x.data_ptr<double>(),x.size(1),&beta,matrix.data_ptr<double>(),y.size(0));
    }else{
        float alpha=scale,beta=0;
        status=cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,y.size(0),x.size(0),x.size(1),&alpha,
            y.data_ptr<float>(),y.size(1),x.data_ptr<float>(),x.size(1),&beta,matrix.data_ptr<float>(),y.size(0));
    }
    auto restore=cublasSetMathMode(handle,previous);
    TORCH_CHECK(status==CUBLAS_STATUS_SUCCESS && restore==CUBLAS_STATUS_SUCCESS,"Raw pricing SGEMM failed");
    auto result=scan(x,y,qb,dbb,3,k,matrix,true,nx,ny,qb,dbb);
    auto stats=torch::empty({4},qb.options());
    auto row_indices=torch::empty({x.size(0),k},x.options().dtype(torch::kInt32));
    auto col_indices=torch::empty({y.size(0),k},x.options().dtype(torch::kInt32));
    raw_outputs<<<(std::max(row_indices.numel(),col_indices.numel())+255)/256,256,0,stream>>>(
        result[1].data_ptr<int64_t>(),row_indices.data_ptr<int32_t>(),row_indices.numel(),
        result[3].data_ptr<int64_t>(),col_indices.data_ptr<int32_t>(),col_indices.numel(),result[4].data_ptr<double>(),stats.data_ptr<double>(),stats_enabled,linf);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {result[0],row_indices,result[2],col_indices,stats,
        torch::zeros({4},torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU))};
}
// CN: 跨数据库及查询块原地合并，索引偏移只应用于有效候选。
// EN: Merge database and query tiles in place, shifting only valid candidate indices.
template<int K> __global__ void merge_raw_tile(const double* incoming,const int32_t* ids,double* output,int32_t* indices,
        int count,int shift,bool first) {
    int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=count)return;
    double v[K];int64_t ix[K];init(v,ix);
    if(!first)for(int j=0;j<K;++j)if(indices[row*K+j]>=0)put(v,ix,output[row*K+j],indices[row*K+j]);
    for(int j=0;j<K;++j)if(ids[row*K+j]>=0)put(v,ix,incoming[row*K+j],ids[row*K+j]+shift);
    for(int j=0;j<K;++j){output[row*K+j]=v[j];indices[row*K+j]=ix[j]==INT64_MAX?-1:ix[j];}
}
__global__ void add_raw_stats(double* target,const double* input) {
    if(threadIdx.x==0){target[0]+=input[0];target[1]+=input[1];target[2]+=input[2];target[3]=fmax(target[3],input[3]);}
}
template<int K> std::vector<torch::Tensor> raw_tiled(torch::Tensor x,torch::Tensor y,torch::Tensor nx,torch::Tensor ny,
        torch::Tensor qb,torch::Tensor dbb,double scale,bool stats_enabled,bool linf,int qt,int nt) {
    if(x.size(0)<=qt && y.size(0)<=nt)return raw_scan(x,y,nx,ny,qb,dbb,K,scale,stats_enabled,linf);
    auto rv=torch::empty({x.size(0),K},qb.options()),cv=torch::empty({y.size(0),K},qb.options());
    auto ri=torch::empty({x.size(0),K},x.options().dtype(torch::kInt32)),ci=torch::empty({y.size(0),K},ri.options());
    auto stats=torch::zeros({4},qb.options());auto stream=at::cuda::getCurrentCUDAStream();
    for(int64_t i=0;i<x.size(0);i+=qt)for(int64_t j=0;j<y.size(0);j+=nt){
        int q=std::min<int64_t>(qt,x.size(0)-i),n=std::min<int64_t>(nt,y.size(0)-j);
        std::vector<torch::Tensor> out=raw_scan(x.narrow(0,i,q),y.narrow(0,j,n),nx.narrow(0,i,q),ny.narrow(0,j,n),
            qb.narrow(0,i,q),dbb.narrow(0,j,n),K,scale,stats_enabled,linf);
        merge_raw_tile<K><<<(q+255)/256,256,0,stream>>>(out[0].data_ptr<double>(),out[1].data_ptr<int32_t>(),rv.data_ptr<double>()+i*K,ri.data_ptr<int32_t>()+i*K,q,j,j==0);
        merge_raw_tile<K><<<(n+255)/256,256,0,stream>>>(out[2].data_ptr<double>(),out[3].data_ptr<int32_t>(),cv.data_ptr<double>()+j*K,ci.data_ptr<int32_t>()+j*K,n,i,i==0);
        add_raw_stats<<<1,1,0,stream>>>(stats.data_ptr<double>(),out[4].data_ptr<double>());
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {rv,ri,cv,ci,stats,torch::zeros({4},torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU))};
}
std::vector<torch::Tensor> raw_scan_tiled(torch::Tensor x,torch::Tensor y,torch::Tensor nx,torch::Tensor ny,
        torch::Tensor qb,torch::Tensor dbb,int k,double scale,bool stats_enabled,bool linf,int qt,int nt) {
    TORCH_CHECK(qt>0 && nt>0 && (k==1 || k==2),"Invalid raw pricing tile or bucket");
    c10::cuda::CUDAGuard guard(x.device());
    if(k==1)return raw_tiled<1>(x,y,nx,ny,qb,dbb,scale,stats_enabled,linf,qt,nt);
    return raw_tiled<2>(x,y,nx,ny,qb,dbb,scale,stats_enabled,linf,qt,nt);
}
// CN: initial top-1复用矩阵行归约，不生成列候选，固定成本统计来自同一次扫描。
// EN: Initial top-1 reuses matrix row reduction without column candidates, collecting fixed cost statistics in the same pass.
std::vector<torch::Tensor> matrix_initial(torch::Tensor matrix,torch::Tensor g) {
    TORCH_CHECK(matrix.is_cuda() && matrix.dim()==2 && matrix.is_contiguous() && matrix.size(0)>0 && matrix.size(1)>0
        && (matrix.scalar_type()==torch::kFloat32 || matrix.scalar_type()==torch::kFloat64),"Expected contiguous cost matrix");
    TORCH_CHECK(g.device()==matrix.device() && g.scalar_type()==torch::kFloat64 && g.is_contiguous() && g.numel()==matrix.size(1),"Expected matching FP64 database dual");
    c10::cuda::CUDAGuard guard(matrix.device());int q=matrix.size(0),n=matrix.size(1);
    auto f=torch::zeros({q},g.options()),values=torch::empty({q,1},g.options()),ids=torch::empty({q,1},g.options().dtype(torch::kInt64)),stats=torch::zeros({5},g.options()),empty=g.new_empty({0});
    auto stream=at::cuda::getCurrentCUDAStream();
#define ROW(D) matrix_rows<1,D,false><<<q,256,0,stream>>>(matrix.data_ptr<D>(),q,n,f.data_ptr<double>(),g.data_ptr<double>(),empty.data_ptr<double>(),empty.data_ptr<double>(),empty.data_ptr<double>(),empty.data_ptr<double>(),values.data_ptr<double>(),ids.data_ptr<int64_t>(),stats.data_ptr<double>())
    if(matrix.scalar_type()==torch::kFloat64){ROW(double);}else{ROW(float);}
#undef ROW
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {values.neg_(),ids,torch::stack({stats[1],stats[3]})};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("scan",&scan);m.def("merge",&merge);m.def("raw_scan",&raw_scan);m.def("raw_scan_tiled",&raw_scan_tiled);m.def("matrix_initial",&matrix_initial);}
