#pragma shader_stage(compute)

#include "nbl/builtin/hlsl/workgroup/scratch_size.hlsl"

uint32_t3 gl_WorkGroupSize() { return uint32_t3(WORKGROUP_SIZE, 1, 1); }

static const uint32_t ArithmeticSz = nbl::hlsl::workgroup::scratch_size_arithmetic<ITEMS_PER_WG>::value;
static const uint32_t BallotSz = nbl::hlsl::workgroup::scratch_size_ballot<ITEMS_PER_WG>::value;
static const uint32_t ScratchSz = ArithmeticSz+BallotSz;

// TODO: Can we make it a static variable?
groupshared uint32_t scratch[ScratchSz];

template<uint32_t offset>
struct ScratchProxy
{
	uint32_t get(const uint32_t ix)
	{
		return scratch[ix + offset];
	}

	void set(const uint32_t ix, const uint32_t value)
	{
		scratch[ix + offset] = value;
	}

	uint32_t atomicAdd(const uint32_t ix, const uint32_t data)
	{
		return nbl::hlsl::glsl::atomicAdd(scratch[ix + offset], data);
	}

	uint32_t atomicOr(const uint32_t ix, const uint32_t data)
	{
		return nbl::hlsl::glsl::atomicOr(scratch[ix + offset], data);
	}

	void workgroupExecutionAndMemoryBarrier()
	{
		nbl::hlsl::glsl::barrier();
		//nbl::hlsl::glsl::memoryBarrierShared(); implied by the above
	}
};

ScratchProxy<0> arithmeticAccessor;
ScratchProxy<ArithmeticSz> ballotAccessor;


#include "nbl/builtin/hlsl/workgroup/broadcast.hlsl"

template<class Binop>
struct operation_t : nbl::hlsl::OPERATION<Binop>
{
	using base_t = nbl::hlsl::OPERATION<Binop>;
	using type_t = typename Binop::type_t;

	SharedMemory accessors;

	type_t operator()(type_t value)
	{
		// we barrier before because we alias the accessors for Binop
		memoryAccessor.workgroupExecutionAndMemoryBarrier();
		// ballots need a special codepath
		if (nbl::hlsl::is_same<ballot<type_t>,Binop>::value)
		{
			// we can only ballot booleans, so low bit
			nbl::hlsl::workgroup::ballot(bool(value&0x1u),ballotAccessor);
			if (nbl::hlsl::is_same<nbl::hlsl::workgroup::reduction<Binop>,base_t>::value)
				return nbl::hlsl::workgroup::ballotBitCount<SharedMemory>(ballotAccessor);
			else if (nbl::hlsl::is_same<nbl::hlsl::workgroup::inclusive_scan<Binop>,base_t>::value)
				return nbl::hlsl::workgroup::ballotInclusiveBitCount<SharedMemory>(ballotAccessor);
			else if (nbl::hlsl::is_same<nbl::hlsl::workgroup::exclusive_scan<Binop>,base_t>::value)
				return nbl::hlsl::workgroup::ballotExclusiveBitCount<SharedMemory>(ballotAccessor);
			else
			{
				assert(false);
				return 0xdeadbeefu;
			}
		}
		else
			return oxdeadbeefu;//base_t::operator()(value,arithmeticAccessor);
	}
};

#include "shaderCommon.hlsl"