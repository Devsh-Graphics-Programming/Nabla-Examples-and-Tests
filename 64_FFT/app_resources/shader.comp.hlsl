#include "common.hlsl"
#include "nbl/builtin/hlsl/workgroup/fft.hlsl"

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
	Accessor accessor;
	SharedMemoryAccessor sharedmemAccessor;

	// Workgroup	

	workgroup::FFT<ElementsPerThread, true, scalar_t>::template __call<Accessor, SharedMemoryAccessor>(accessor, sharedmemAccessor);
	accessor.workgroupExecutionAndMemoryBarrier();
	workgroup::FFT<ElementsPerThread, false, scalar_t>::template __call<Accessor, SharedMemoryAccessor>(accessor, sharedmemAccessor);	

}