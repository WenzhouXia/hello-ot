#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <limits>
#include <cub/warp/warp_merge_sort.cuh>
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
    // CN: 直接从排序后的寄存器交换前32项，省去96项共享中转。
    // EN: Shuffle the first 32 sorted register items without a 96-item shared staging buffer.
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
    // CN: 比较方式只随GPU架构变化，不依赖普通维度或数据集。
    // EN: Comparison strategy depends only on GPU architecture, never ordinary dimensions or datasets.
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
    int lane=threadIdx.x%32,warp=threadIdx.x/32,col=blockIdx.x*4+warp;
    if(col>=n)return;
    using Sort=cub::WarpMergeSort<Candidate,3,32>;
    __shared__ typename Sort::TempStorage storage[4];
    __shared__ Candidate queues[4][32];
    Candidate* queue=queues[warp];queue[lane]={-INFINITY,-1};__syncwarp();
    Candidate local[2]={{-INFINITY,-1},{-INFINITY,-1}};int count=0;
    for(int base=0;base<tiles*P;base+=32) {
        int p=base+lane;
        if(p<tiles*P){int64_t pos=((int64_t)(p/P)*n+col)*P+p%P;int64_t index=idx[pos];Candidate candidate={v[pos],static_cast<int32_t>(index)};
            if(index>=0 && index<INT_MAX && better(candidate,queue[K-1])){local[1]=local[0];local[0]=candidate;++count;}}
        if(__any_sync(0xffffffffu,count==2))merge_queue<Sort>(local,count,queue,storage[warp]);
    }
    merge_queue<Sort>(local,count,queue,storage[warp]);
    if(lane<K){out[(int64_t)col*K+lane]=queue[lane].value;ids[(int64_t)col*K+lane]=queue[lane].index;}
}

// CN: 六档实例；top1/2保留原几何，大k减少块行数以容纳warp队列。
// EN: Instantiate six buckets; retain top1/2 geometry and reduce rows for larger warp queues.
#include "matrix_selection.cuh"
template<int K,int KIND,bool MATRIX,bool RAW>
std::vector<torch::Tensor> run(torch::Tensor x,torch::Tensor y,torch::Tensor f,torch::Tensor g,torch::Tensor matrix,
        torch::Tensor nx,torch::Tensor ny,torch::Tensor qb,torch::Tensor dbb) {
    if constexpr(MATRIX && K<=8){
        if constexpr(KIND==4)return matrix_run<K,double,RAW>(x,y,f,g,matrix,nx,ny,qb,dbb);
        return matrix_run<K,float,RAW>(x,y,f,g,matrix,nx,ny,qb,dbb);
    }
    constexpr int R=K<=2?16:(MATRIX?(K<=8?16:8):4),T=MATRIX && K>2 && K<=8?64:128,P=K<R?K:R;
    int q=x.size(0),n=y.size(0),blocks=(q+R-1)/R;
    int wanted=std::max(1,(2*at::cuda::getCurrentDeviceProperties()->multiProcessorCount+blocks-1)/blocks);
    int db_tile=((n+wanted-1)/wanted+T-1)/T*T,db_blocks=(n+db_tile-1)/db_tile;
    auto opts=x.options().dtype(torch::kFloat64),ints=x.options().dtype(torch::kInt64);
    auto rv=torch::empty({db_blocks,q,K},opts),ri=torch::empty({db_blocks,q,K},ints),pv=torch::empty({blocks,n,P},opts),pi=torch::empty({blocks,n,P},ints),stats=torch::zeros({5},opts);
#define CALL(D) scan_kernel<K,R,T,KIND,MATRIX,D,RAW><<<dim3(blocks,db_blocks),R*32,0,at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const float*>(x.data_ptr()),reinterpret_cast<const float*>(y.data_ptr()),f.data_ptr<double>(),g.data_ptr<double>(),MATRIX?reinterpret_cast<const float*>(matrix.data_ptr()):nullptr,nx.data_ptr<double>(),ny.data_ptr<double>(),qb.data_ptr<double>(),dbb.data_ptr<double>(),q,n,x.size(1),db_tile,rv.data_ptr<double>(),ri.data_ptr<int64_t>(),pv.data_ptr<double>(),pi.data_ptr<int64_t>(),stats.data_ptr<double>())
    if constexpr(MATRIX){CALL(0);}else if(x.size(1)==2){CALL(2);}else if(x.size(1)==3){CALL(3);}else{CALL(0);}
#undef CALL
    auto cv=torch::empty({n,K},opts),ci=torch::empty({n,K},ints);
    reduce_columns<K,P><<<(n+3)/4,128,0,at::cuda::getCurrentCUDAStream()>>>(pv.data_ptr<double>(),pi.data_ptr<int64_t>(),blocks,n,cv.data_ptr<double>(),ci.data_ptr<int64_t>());
    if(db_blocks>1){auto rowv=torch::empty({q,K},opts),rowi=torch::empty({q,K},ints);
        reduce_columns<K><<<(q+3)/4,128,0,at::cuda::getCurrentCUDAStream()>>>(rv.data_ptr<double>(),ri.data_ptr<int64_t>(),db_blocks,q,rowv.data_ptr<double>(),rowi.data_ptr<int64_t>());
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
    switch(k){CASE(1);CASE(2);CASE(4);CASE(8);CASE(16);CASE(32);default:TORCH_CHECK(false,"Invalid bucket");}
#undef CASE
}
template<int K> std::vector<torch::Tensor> merge_k(torch::Tensor values,torch::Tensor indices) {
    auto v=torch::empty({values.size(1),K},values.options());auto i=torch::empty({values.size(1),K},indices.options());
    reduce_columns<K><<<(values.size(1)+3)/4,128,0,at::cuda::getCurrentCUDAStream()>>>(values.data_ptr<double>(),indices.data_ptr<int64_t>(),values.size(0),values.size(1),v.data_ptr<double>(),i.data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {v,i};
}
std::vector<torch::Tensor> merge(torch::Tensor v,torch::Tensor i) {
#define CASE(K) case K:return merge_k<K>(v,i)
    switch(v.size(2)){CASE(1);CASE(2);CASE(4);CASE(8);CASE(16);CASE(32);default:TORCH_CHECK(false,"Invalid bucket");}
#undef CASE
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("scan",&scan);m.def("merge",&merge);}
