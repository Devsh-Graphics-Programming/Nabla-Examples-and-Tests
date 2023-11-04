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

bool canStore()
{
	return nbl::hlsl::workgroup::SubgroupContiguousIndex()<ITEMS_PER_WG;
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
	// need to barrier between ballot and usages of a ballot by myself
	ballotAccessor.workgroupExecutionAndMemoryBarrier();

	uint32_t destVal = 0xdeadbeefu;
#define CONSTEXPR_OP_TYPE_TEST(IS_OP) nbl::hlsl::is_same<nbl::hlsl::OPERATION<nbl::hlsl::bit_xor<float>,0x45>,nbl::hlsl::workgroup::IS_OP<nbl::hlsl::bit_xor<float>,0x45> >::value
	if (CONSTEXPR_OP_TYPE_TEST(reduction))
		destVal = nbl::hlsl::workgroup::ballotBitCount<ITEMS_PER_WG>(ballotAccessor,arithmeticAccessor);
	else if (CONSTEXPR_OP_TYPE_TEST(inclusive_scan))
		destVal = nbl::hlsl::workgroup::ballotInclusiveBitCount<ITEMS_PER_WG>(ballotAccessor,arithmeticAccessor);
	else if (CONSTEXPR_OP_TYPE_TEST(exclusive_scan))
		destVal = nbl::hlsl::workgroup::ballotExclusiveBitCount<ITEMS_PER_WG>(ballotAccessor,arithmeticAccessor);
	else
	{
		assert(false);
	}
#undef CONSTEXPR_OP_TYPE_TEST

	if (canStore())
		output[ballot<type_t>::BindingIndex].template Store<type_t>(sizeof(uint32_t)+sizeof(type_t)*globalIndex(),destVal);
}