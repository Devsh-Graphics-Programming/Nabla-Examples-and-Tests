#pragma shader_stage(compute)


#include "nbl/builtin/hlsl/workgroup/scratch_size.hlsl"

static const uint32_t ArithmeticSz = nbl::hlsl::workgroup::scratch_size_arithmetic<ITEMS_PER_WG>::value;
static const uint32_t BallotSz = nbl::hlsl::workgroup::scratch_size_ballot<ITEMS_PER_WG>::value;
static const uint32_t ScratchSz = ArithmeticSz+BallotSz;

// TODO: Can we make it a static variable?
groupshared uint32_t scratch[ScratchSz];


#include "nbl/builtin/hlsl/workgroup/arithmetic.hlsl"


template<uint16_t offset>
struct ScratchProxy
{
	uint32_t get(const uint32_t ix)
	{
		return scratch[ix+offset];
	}
	void set(const uint32_t ix, const uint32_t value)
	{
		scratch[ix+offset] = value;
	}

	// using MemorySemanticsMaskNone because the execution barrier later ensures availability and visibility for smem
	uint32_t atomicAdd(const uint32_t ix, const uint32_t data)
	{
		return nbl::hlsl::spirv::atomicAdd(scratch[ix+offset],spv::ScopeWorkgroup,spv::MemorySemanticsMaskNone,data);
	}
	uint32_t atomicOr(const uint32_t ix, const uint32_t data)
	{
		return nbl::hlsl::spirv::atomicOr(scratch[ix+offset],spv::ScopeWorkgroup,spv::MemorySemanticsMaskNone,data);
	}

	void workgroupExecutionAndMemoryBarrier()
	{
		nbl::hlsl::glsl::barrier();
		//nbl::hlsl::glsl::memoryBarrierShared(); implied by the above
	}
};

static ScratchProxy<0> arithmeticAccessor;


#include "nbl/builtin/hlsl/workgroup/broadcast.hlsl"


template<class Binop>
struct operation_t
{
	using type_t = typename Binop::type_t;

	type_t operator()(type_t value)
	{
		type_t retval = nbl::hlsl::OPERATION<Binop,ITEMS_PER_WG>::template __call<ScratchProxy<0> >(value,arithmeticAccessor);
		// we barrier before because we alias the accessors for Binop
		arithmeticAccessor.workgroupExecutionAndMemoryBarrier();
		return retval;
	}
};


#include "shaderCommon.hlsl"

static ScratchProxy<ArithmeticSz> ballotAccessor;


uint32_t globalIndex()
{
	return nbl::hlsl::glsl::gl_WorkGroupID().x*ITEMS_PER_WG+nbl::hlsl::workgroup::SubgroupContiguousIndex();
}

[numthreads(WORKGROUP_SIZE,1,1)]
void main(uint32_t invIdx : SV_GroupIndex, uint32_t3 wgId : SV_GroupID)
{
	__gl_LocalInvocationIndex = invIdx;
	__gl_WorkGroupID = wgId;

	const type_t sourceVal = test();
	if (globalIndex()==0u)
		output[ballot<type_t>::BindingIndex].template Store<uint32_t>(0,nbl::hlsl::glsl::gl_SubgroupSize());

	// we can only ballot booleans, so low bit
	nbl::hlsl::workgroup::ballot<ScratchProxy<ArithmeticSz> >(bool(sourceVal&0x1u),ballotAccessor);
	uint32_t destVal = 0xdeadbeefu;
#if 1
	if (true)
		destVal = nbl::hlsl::workgroup::ballotBitCount<ITEMS_PER_WG>(ballotAccessor,arithmeticAccessor);
	else
	{
		assert(false);
	}
#endif
	output[ballot<type_t>::BindingIndex].template Store<type_t>(sizeof(uint32_t)+sizeof(type_t)*globalIndex(),destVal);
}