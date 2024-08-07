#include "common.hlsl"
#include "nbl/builtin/hlsl/workgroup/fft.hlsl"
//#include "nbl/builtin/hlsl/subgroup/fft.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_basic.hlsl"

[[vk::push_constant]] PushConstantData pushConstants;

// careful: change size according to Scalar type
groupshared uint32_t sharedmem[4 * WorkgroupSize];

// Users MUST define this method for FFT to work
namespace nbl { namespace hlsl { namespace glsl
{
uint32_t3 gl_WorkGroupSize() { return uint32_t3(WorkgroupSize, 1, 1); }
} } }

struct SharedMemoryAccessor {
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

struct Accessor {
	void set(uint32_t idx, nbl::hlsl::complex_t<input_t> value) 
	{
		vk::RawBufferStore< vector<input_t, 2> >(pushConstants.outputAddress + sizeof(vector<input_t, 2>) * idx, vector<input_t, 2>(value.real(), value.imag()));
	}
	
	void get(uint32_t idx, NBL_REF_ARG(nbl::hlsl::complex_t<input_t>) value) 
	{
		vector<input_t, 2> aux = vk::RawBufferLoad< vector<input_t, 2> >(pushConstants.inputAddress + sizeof(vector<input_t, 2>) * idx);
		value.real(aux.x);
		value.imag(aux.y);
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

	nbl::hlsl::workgroup::FFT<ElementsPerThread, true, input_t>::template __call<Accessor, SharedMemoryAccessor>(accessor, sharedmemAccessor);
	accessor.workgroupExecutionAndMemoryBarrier();
	nbl::hlsl::workgroup::FFT<ElementsPerThread, false, input_t>::template __call<Accessor, SharedMemoryAccessor>(accessor, sharedmemAccessor);	

}