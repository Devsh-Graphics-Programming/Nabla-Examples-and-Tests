#include "common.hlsl"
#include "nbl/builtin/hlsl/workgroup/fft.hlsl"
#include "nbl/builtin/hlsl/bda/legacy_bda_accessor.hlsl"

[[vk::push_constant]] PushConstantData pushConstants;

using namespace nbl::hlsl;

// careful: change size according to Scalar type
groupshared uint32_t sharedmem[ workgroup::fft::sharedMemSize<scalar_t, WorkgroupSize> ];

// Users MUST define this method for FFT to work
uint32_t3 glsl::gl_WorkGroupSize() { return uint32_t3(WorkgroupSize, 1, 1); }

struct SharedMemoryAccessor 
{
	void set(uint32_t idx, uint32_t value) 
	{
		sharedmem[idx] = value;
	}
	
	void get(uint32_t idx, NBL_REF_ARG(uint32_t) value) 
	{
		value = sharedmem[idx];
	}

	void workgroupExecutionAndMemoryBarrier() 
	{
		glsl::barrier();
    }

};

struct Accessor : DoubleLegacyBdaAccessor< complex_t<scalar_t> >
{
	static Accessor create(const uint64_t inputAddress, const uint64_t outputAddress)
    {
        Accessor accessor;
        accessor.inputAddress = inputAddress;
        accessor.outputAddress = outputAddress;
        return accessor;
    }

	void memoryBarrier() 
	{
		// only one workgroup is touching any memory it wishes to trade
		spirv::memoryBarrier(spv::ScopeWorkgroup, spv::MemorySemanticsAcquireReleaseMask | spv::MemorySemanticsUniformMemoryMask);
	}
};

[numthreads(WorkgroupSize,1,1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
	Accessor accessor = Accessor::create(pushConstants.inputAddress, pushConstants.outputAddress);
	SharedMemoryAccessor sharedmemAccessor;

	// FFT

	workgroup::FFT<ElementsPerThread, true, WorkgroupSize, scalar_t>::template __call<Accessor, SharedMemoryAccessor>(accessor, sharedmemAccessor);
	accessor.workgroupExecutionAndMemoryBarrier();
	workgroup::FFT<ElementsPerThread, false, WorkgroupSize, scalar_t>::template __call<Accessor, SharedMemoryAccessor>(accessor, sharedmemAccessor);	
}