#include "common.hlsl"

// just a small test
#include "nbl/builtin/hlsl/jit/device_capabilities.hlsl"
#include "nbl/builtin/hlsl/workgroup/scratch_size.hlsl"
#include "nbl/builtin/hlsl/workgroup/arithmetic.hlsl"

static const uint32_t ArithmeticSz = nbl::hlsl::workgroup::scratch_size_arithmetic<WorkgroupSize>::value;
static const uint32_t BallotSz = nbl::hlsl::workgroup::scratch_size_ballot<WorkgroupSize>::value;
static const uint32_t ScratchSz = ArithmeticSz + BallotSz;

groupshared uint32_t prefixScratch[ScratchSz];

[[vk::push_constant]] PushConstantData pushConstants;
[[vk::binding(0,0)]] RWStructuredBuffer<uint32_t> scratch;

template<uint16_t offset>
struct ScratchProxy
{
    uint32_t get(const uint32_t ix)
    {
        return prefixScratch[ix + offset];
    }
    void set(const uint32_t ix, const uint32_t value)
    {
        prefixScratch[ix + offset] = value;
    }

    void workgroupExecutionAndMemoryBarrier()
    {
        nbl::hlsl::glsl::barrier();
    }
};

static ScratchProxy<0> arithmeticAccessor;

static ScratchProxy<ArithmeticSz> ballotAccessor;

groupshared uint32_t sdata[WorkgroupSize];

uint32_t3 nbl::hlsl::glsl::gl_WorkGroupSize()
{
    return uint32_t3(WorkgroupSize, 1, 1);
}

[numthreads(WorkgroupSize,1,1)]
void main(uint32_t3 ID : SV_GroupThreadID, uint32_t3 GroupID : SV_GroupID)
{
    uint32_t tid = nbl::hlsl::workgroup::SubgroupContiguousIndex();
    sdata[tid] = 0;
    uint32_t index = nbl::hlsl::glsl::gl_WorkGroupID().x * WorkgroupSize + tid;

    nbl::hlsl::glsl::barrier();

    if (index < pushConstants.dataElementCount)
    {
        uint32_t value = vk::RawBufferLoad(pushConstants.inputAddress + sizeof(uint32_t) * index);
        nbl::hlsl::glsl::atomicAdd(sdata[value - pushConstants.minimum], (uint32_t) 1);
    }

    nbl::hlsl::glsl::barrier();

    //nbl::hlsl::workgroup::ballot < decltype(ballotAccessor) > (bool(sdata[tid]), ballotAccessor);
    //ballotAccessor.workgroupExecutionAndMemoryBarrier();
    //uint32_t dest_value = nbl::hlsl::workgroup::ballotExclusiveBitCount < WorkgroupSize, decltype(ballotAccessor), decltype(arithmeticAccessor), nbl::hlsl::jit::device_capabilities > (ballotAccessor, arithmeticAccessor);

    for (uint32_t i = 1; i < WorkgroupSize; i *= 2)
    {
        nbl::hlsl::glsl::barrier();
        if (tid - i >= 0)
            sdata[tid] += sdata[tid - i];
        nbl::hlsl::glsl::barrier();
    }

    nbl::hlsl::glsl::atomicAdd(scratch[tid], sdata[tid]);    
}