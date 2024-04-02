#include "common.hlsl"

// just a small test
#include "nbl/builtin/hlsl/workgroup/arithmetic.hlsl"
#include "nbl/builtin/hlsl/jit/device_capabilities.hlsl"
#include "nbl/builtin/hlsl/workgroup/scratch_size.hlsl"

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
    sdata[ID.x] = 0;
    uint32_t index = WorkgroupSize * GroupID.x + ID.x;

    nbl::hlsl::glsl::barrier();

    if (index < pushConstants.dataElementCount)
    {
        uint32_t value = vk::RawBufferLoad(pushConstants.inputAddress + sizeof(uint32_t) * index);
        nbl::hlsl::glsl::atomicAdd(sdata[value - pushConstants.minimum], (uint32_t) 1);
    }

    nbl::hlsl::glsl::barrier();

    // we can only ballot booleans, so low bit
    nbl::hlsl::workgroup::ballot < decltype(ballotAccessor) > (bool(sdata[ID.x] & 0x1u), ballotAccessor);
	// need to barrier between ballot and usages of a ballot by myself
    ballotAccessor.workgroupExecutionAndMemoryBarrier();
    sdata[ID.x] = nbl::hlsl::workgroup::ballotInclusiveBitCount < WorkgroupSize, decltype(ballotAccessor), decltype(arithmeticAccessor), nbl::hlsl::jit::device_capabilities > (ballotAccessor, arithmeticAccessor);

    nbl::hlsl::glsl::atomicAdd(scratch[ID.x], sdata[ID.x]);

    nbl::hlsl::glsl::barrier();

    //vk::RawBufferStore(pushConstants.outputAddress + sizeof(uint32_t) * index, ID.x);
}