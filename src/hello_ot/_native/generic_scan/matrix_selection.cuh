#pragma once

// CN: 点数上限为int32；局部索引使用int32减少寄存器和shuffle，公开输出仍为int64。
// EN: Point counts are int32-bounded; local int32 indices reduce registers and shuffles while public outputs remain int64.
template<int K> __device__ void matrix_init(double (&v)[K],int (&i)[K]) {
    #pragma unroll
    for(int j=0;j<K;++j){v[j]=-INFINITY;i[j]=INT_MAX;}
}
template<int K> __device__ void matrix_put(double (&v)[K],int (&i)[K],double value,int index) {
    if(!better(value,index,v[K-1],i[K-1]))return;
    int p=K-1;
    #pragma unroll
    for(int j=K-2;j>=0;--j)if(better(value,index,v[j],i[j]))p=j;
    #pragma unroll
    for(int j=K-1;j>0;--j)if(j>p){v[j]=v[j-1];i[j]=i[j-1];}
    v[p]=value;i[p]=index;
}
// CN: 已物化成本矩阵时，行统计与列选择分别归约，避免生成O(q*n)列候选中间缓冲。
// EN: For materialized matrices, reduce row statistics and column candidates separately, avoiding O(q*n) partial-candidate buffers.
template<typename Dot,bool RAW> __device__ double matrix_score(const Dot* matrix,int64_t p,int row,int col,
        const double* f,const double* g,const double* nx,const double* ny,const double* qb,const double* dbb,double& cost) {
    double value=matrix[p];cost=value;
    if constexpr(RAW){cost=(nx[row]+ny[col])-value;return (value+qb[row])+dbb[col];}
    if constexpr(sizeof(Dot)==8)return (f[row]+g[col])-cost;
    return (g[col]-cost)+f[row];
}
template<int K,typename Dot,bool RAW> __global__ void matrix_rows(const Dot* matrix,int q,int n,
        const double* f,const double* g,const double* nx,const double* ny,const double* qb,const double* dbb,
        double* out,int64_t* ids,double* stats) {
    int row=blockIdx.x,lane=threadIdx.x%32,warp=threadIdx.x/32;
    double values[K];int indices[K];matrix_init(values,indices);
    double sums[5]={};
    for(int col=threadIdx.x;col<n;col+=blockDim.x){
        double cost;double score=matrix_score<Dot,RAW>(matrix,(int64_t)row*n+col,row,col,f,g,nx,ny,qb,dbb,cost);
        matrix_put(values,indices,score,col);sums[1]+=cost*cost;sums[3]=fmax(sums[3],fabs(cost));
        if(score>0){sums[0]+=score*score;sums[2]=fmax(sums[2],score);sums[4]+=1;}
    }
    for(int offset=16;offset>0;offset/=2){
        double other[K];int oi[K];
        #pragma unroll
        for(int j=0;j<K;++j){other[j]=__shfl_down_sync(0xffffffffu,values[j],offset);oi[j]=__shfl_down_sync(0xffffffffu,indices[j],offset);}
        if(lane+offset<32)for(int j=0;j<K;++j)matrix_put(values,indices,other[j],oi[j]);
        for(int j=0;j<5;++j){double value=__shfl_down_sync(0xffffffffu,sums[j],offset);sums[j]=(j==2 || j==3)?fmax(sums[j],value):sums[j]+value;}
    }
    __shared__ double sv[8*K],ss[8*5];__shared__ int si[8*K];
    if(lane==0){for(int j=0;j<K;++j){sv[warp*K+j]=values[j];si[warp*K+j]=indices[j];}for(int j=0;j<5;++j)ss[warp*5+j]=sums[j];}
    __syncthreads();
    if(warp==0){
        matrix_init(values,indices);for(int j=0;j<5;++j)sums[j]=lane<8?ss[lane*5+j]:0;
        if(lane<8)for(int j=0;j<K;++j)matrix_put(values,indices,sv[lane*K+j],si[lane*K+j]);
        for(int offset=16;offset>0;offset/=2){
            double other[K];int oi[K];
            for(int j=0;j<K;++j){other[j]=__shfl_down_sync(0xffffffffu,values[j],offset);oi[j]=__shfl_down_sync(0xffffffffu,indices[j],offset);}
            if(lane+offset<32)for(int j=0;j<K;++j)matrix_put(values,indices,other[j],oi[j]);
            for(int j=0;j<5;++j){double value=__shfl_down_sync(0xffffffffu,sums[j],offset);sums[j]=(j==2 || j==3)?fmax(sums[j],value):sums[j]+value;}
        }
        if(lane==0){for(int j=0;j<K;++j){out[(int64_t)row*K+j]=values[j];ids[(int64_t)row*K+j]=indices[j]==INT_MAX?-1:indices[j];}
            atomicAdd(stats,sums[0]);atomicAdd(stats+1,sums[1]);max_atomic(stats+2,sums[2]);max_atomic(stats+3,sums[3]);atomicAdd(stats+4,sums[4]);}
    }
}
template<int K,typename Dot,bool RAW> __global__ void matrix_columns(const Dot* matrix,int q,int n,
        const double* f,const double* g,const double* nx,const double* ny,const double* qb,const double* dbb,
        double* out,int64_t* ids) {
    constexpr int L=8;int lane=threadIdx.x%32,part=lane%L;
    int col=(blockIdx.x*(blockDim.x/32)+threadIdx.x/32)*4+lane/L;
    double values[K];int indices[K];matrix_init(values,indices);
    if(col<n)for(int row=part;row<q;row+=L){double cost;
        double score=matrix_score<Dot,RAW>(matrix,(int64_t)row*n+col,row,col,f,g,nx,ny,qb,dbb,cost);
        matrix_put(values,indices,score,row);}
    for(int offset=L/2;offset>0;offset/=2){double other[K];int oi[K];
        for(int j=0;j<K;++j){other[j]=__shfl_down_sync(0xffffffffu,values[j],offset,L);oi[j]=__shfl_down_sync(0xffffffffu,indices[j],offset,L);}
        if(part<offset)for(int j=0;j<K;++j)matrix_put(values,indices,other[j],oi[j]);}
    if(col<n && part==0)for(int j=0;j<K;++j){out[(int64_t)col*K+j]=values[j];ids[(int64_t)col*K+j]=indices[j]==INT_MAX?-1:indices[j];}
}
template<int K,typename Dot,bool RAW> std::vector<torch::Tensor> matrix_run(torch::Tensor x,torch::Tensor y,
        torch::Tensor f,torch::Tensor g,torch::Tensor matrix,torch::Tensor nx,torch::Tensor ny,torch::Tensor qb,torch::Tensor dbb) {
    int q=x.size(0),n=y.size(0);auto options=f.options(),ints=x.options().dtype(torch::kInt64);
    auto rv=torch::empty({q,K},options),cv=torch::empty({n,K},options),ri=torch::empty({q,K},ints),ci=torch::empty({n,K},ints),stats=torch::zeros({5},options);
    auto stream=at::cuda::getCurrentCUDAStream();
    matrix_rows<K,Dot,RAW><<<q,256,0,stream>>>(matrix.data_ptr<Dot>(),q,n,f.data_ptr<double>(),g.data_ptr<double>(),nx.data_ptr<double>(),ny.data_ptr<double>(),qb.data_ptr<double>(),dbb.data_ptr<double>(),rv.data_ptr<double>(),ri.data_ptr<int64_t>(),stats.data_ptr<double>());
    matrix_columns<K,Dot,RAW><<<(n+31)/32,256,0,stream>>>(matrix.data_ptr<Dot>(),q,n,f.data_ptr<double>(),g.data_ptr<double>(),nx.data_ptr<double>(),ny.data_ptr<double>(),qb.data_ptr<double>(),dbb.data_ptr<double>(),cv.data_ptr<double>(),ci.data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {rv,ri,cv,ci,stats};
}
