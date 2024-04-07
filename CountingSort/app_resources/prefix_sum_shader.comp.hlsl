#include "common.hlsl"

// just a small test
#include "nbl/builtin/hlsl/jit/device_capabilities.hlsl"
#include "nbl/builtin/hlsl/workgroup/arithmetic.hlsl"

NBL_CONSTEXPR uint32_t BucketsPerThread = ceil((float) BucketCount / WorkgroupSize);

[[vk::push_constant]] PushConstantData pushConstants;
[[vk::binding(0,0)]] RWStructuredBuffer<uint32_t> scratch;

groupshared uint32_t prefixScratch[BucketCount];

struct ScratchProxy
{
    uint32_t get(const uint32_t ix)
    {
        return prefixScratch[ix];
    }
    void set(const uint32_t ix, const uint32_t value)
    {
        prefixScratch[ix] = value;
    }

    void workgroupExecutionAndMemoryBarrier()
    {
        nbl::hlsl::glsl::barrier();
    }
};

static ScratchProxy arithmeticAccessor;

groupshared uint32_t sdata[BucketCount];

uint32_t3 nbl::hlsl::glsl::gl_WorkGroupSize()
{
    return uint32_t3(WorkgroupSize, 1, 1);
}

[numthreads(WorkgroupSize,1,1)]
void main(uint32_t3 ID : SV_GroupThreadID, uint32_t3 GroupID : SV_GroupID)
{
    uint32_t tid = nbl::hlsl::workgroup::SubgroupContiguousIndex();
    
    [unroll]
    for (int i = 0; i < BucketsPerThread; i++)
        sdata[BucketsPerThread * tid + i] = 0;
    uint32_t index = (nbl::hlsl::glsl::gl_WorkGroupID().x * WorkgroupSize + tid) * pushConstants.elementsPerWT;

    nbl::hlsl::glsl::barrier();

    for (int i = 0; i < pushConstants.elementsPerWT; i++)
    {
        if (index + i >= pushConstants.dataElementCount)
            break;
        uint32_t value = vk::RawBufferLoad(pushConstants.inputAddress + sizeof(uint32_t) * (index + i));
        nbl::hlsl::glsl::atomicAdd(sdata[value - pushConstants.minimum], (uint32_t) 1);
    }

    nbl::hlsl::glsl::barrier();

    uint32_t sum = 0;
    uint32_t scan_sum = 0;
    
    for (int i = 0; i < BucketsPerThread; i++)
    {
        sum = nbl::hlsl::workgroup::exclusive_scan < nbl::hlsl::plus < uint32_t >, WorkgroupSize > ::
        template __call <ScratchProxy>
        (sdata[WorkgroupSize * i + tid], arithmeticAccessor);

        arithmeticAccessor.workgroupExecutionAndMemoryBarrier();
    
        nbl::hlsl::glsl::atomicAdd(scratch[WorkgroupSize * i + tid], sum);
        if ((tid == WorkgroupSize - 1) && i > 0)
            nbl::hlsl::glsl::atomicAdd(scratch[WorkgroupSize * i], scan_sum);
        
        arithmeticAccessor.workgroupExecutionAndMemoryBarrier();
        
        if ((tid == WorkgroupSize - 1) && i < (BucketsPerThread - 1))
        {
            scan_sum = sum + sdata[WorkgroupSize * i + tid];
            sdata[WorkgroupSize * (i + 1)] += scan_sum;
        }

        arithmeticAccessor.workgroupExecutionAndMemoryBarrier();
    }
}