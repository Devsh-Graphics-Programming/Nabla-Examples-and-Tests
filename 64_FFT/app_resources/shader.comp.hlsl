#include "common.hlsl"
#include "nbl/builtin/hlsl/workgroup/fft.hlsl"

[[vk::push_constant]] PushConstantData pushConstants;

// careful: change size according to Scalar type
groupshared uint32_t sharedmem[4 * WorkgroupSize];

// Users MUST define this method for FFT to work
namespace nbl { namespace hlsl { namespace glsl
{
uint32_t3 gl_WorkGroupSize() { return uint32_t3(WorkgroupSize, 1, 1); }
} } }

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
		AllMemoryBarrierWithGroupSync();
    }

};

struct Accessor
{
	void set(uint32_t idx, nbl::hlsl::complex_t<scalar_t> value) 
	{
		vk::RawBufferStore< nbl::hlsl::complex_t<scalar_t> >(pushConstants.outputAddress + sizeof(nbl::hlsl::complex_t<scalar_t>) * idx, value);
	}
	
	void get(uint32_t idx, NBL_REF_ARG(nbl::hlsl::complex_t<scalar_t>) value) 
	{
		value = vk::RawBufferLoad< nbl::hlsl::complex_t<scalar_t> >(pushConstants.inputAddress + sizeof(nbl::hlsl::complex_t<scalar_t>) * idx);
	}

	void workgroupExecutionAndMemoryBarrier() 
	{
		AllMemoryBarrierWithGroupSync();
    }

	void memoryBarrier() 
	{
		AllMemoryBarrier();
	}
};

[numthreads(WorkgroupSize,1,1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
	Accessor accessor;
	SharedMemoryAccessor sharedmemAccessor;

	// Workgroup	

	nbl::hlsl::workgroup::FFT<ElementsPerThread, true, scalar_t>::template __call<Accessor, SharedMemoryAccessor>(accessor, sharedmemAccessor);
	accessor.workgroupExecutionAndMemoryBarrier();
	nbl::hlsl::workgroup::FFT<ElementsPerThread, false, scalar_t>::template __call<Accessor, SharedMemoryAccessor>(accessor, sharedmemAccessor);	

}