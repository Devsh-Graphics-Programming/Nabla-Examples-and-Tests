#include "common.hlsl"
#include "nbl/builtin/hlsl/workgroup/fft.hlsl"
#include "nbl/builtin/hlsl/bda/bda_accessor.hlsl"

[[vk::push_constant]] PushConstantData pushConstants;

// careful: change size according to Scalar type
groupshared uint32_t sharedmem[4 * WorkgroupSize];

using namespace nbl::hlsl;

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

// Commented out until we can get BDA working for this
/* 
struct Accessor : DoubleBdaAccessor< complex_t<scalar_t> >
{
	static Accessor create(const bda::__ptr< complex_t<scalar_t> > inputPtr, const bda::__ptr< complex_t<scalar_t> > outputPtr)
    {
        Accessor accessor;
        accessor.inputPtr = inputPtr;
        accessor.outputPtr = outputPtr;
        return accessor;
    }

	void memoryBarrier() 
	{
		// only one workgroup is touching any memory it wishes to trade
		spirv::memoryBarrier(spv::ScopeWorkgroup, spv::MemorySemanticsAcquireReleaseMask | spv::MemorySemanticsUniformMemoryMask);
	}

	void workgroupExecutionAndMemoryBarrier() 
	{
		// we're only barriering the workgroup and trading memory within a workgroup
		spirv::controlBarrier(spv::ScopeWorkgroup, spv::ScopeWorkgroup, spv::MemorySemanticsAcquireReleaseMask | spv::MemorySemanticsUniformMemoryMask);
    }
};
*/

struct Accessor
{
	void set(uint32_t idx, complex_t<scalar_t> value) 
	{
		vk::RawBufferStore<complex_t<scalar_t> >(pushConstants.outputAddress + sizeof(complex_t<scalar_t>) * idx, value);
	}
	
	void get(uint32_t idx, NBL_REF_ARG(complex_t<scalar_t>) value) 
	{
		value = vk::RawBufferLoad<complex_t<scalar_t> >(pushConstants.inputAddress + sizeof(complex_t<scalar_t>) * idx);
	}

	// Note: Its a funny quirk of the SPIR-V Vulkan Env spec that `MemorySemanticsUniformMemoryMask` means SSBO as well :facepalm: (and probably BDA)
	void workgroupExecutionAndMemoryBarrier() 
	{
		// we're only barriering the workgroup and trading memory within a workgroup
		spirv::controlBarrier(spv::ScopeWorkgroup, spv::ScopeWorkgroup, spv::MemorySemanticsAcquireReleaseMask | spv::MemorySemanticsUniformMemoryMask);
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
	// Commented out until we can get BDA working for this
	/*
	bda::__ptr< complex_t<scalar_t> > inputPtr  = bda::__ptr< complex_t<scalar_t> >::create(pushConstants.inputAddress);
	bda::__ptr< complex_t<scalar_t> > outputPtr = bda::__ptr< complex_t<scalar_t> >::create(pushConstants.outputAddress);
	Accessor accessor = Accessor::create(inputPtr, outputPtr);
	*/
	Accessor accessor;
	SharedMemoryAccessor sharedmemAccessor;

	// FFT

	workgroup::FFT<ElementsPerThread, true, scalar_t>::template __call<Accessor, SharedMemoryAccessor>(accessor, sharedmemAccessor);
	accessor.workgroupExecutionAndMemoryBarrier();
	workgroup::FFT<ElementsPerThread, false, scalar_t>::template __call<Accessor, SharedMemoryAccessor>(accessor, sharedmemAccessor);	
}