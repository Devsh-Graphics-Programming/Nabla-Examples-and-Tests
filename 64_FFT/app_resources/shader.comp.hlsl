#include "common.hlsl"
#include "nbl/builtin/hlsl/workgroup/fft.hlsl"
//#include "nbl/builtin/hlsl/subgroup/fft.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_basic.hlsl"

[[vk::push_constant]] PushConstantData pushConstants;

groupshared input_t sharedmem[2 * WorkgroupSize];

// Users MUST define this method for FFT to work
namespace nbl { namespace hlsl { namespace glsl
{
uint32_t3 gl_WorkGroupSize() { return uint32_t3(WorkgroupSize, 1, 1); }
} } }

struct SharedMemoryAccessor {
	void set(uint32_t idx, nbl::hlsl::complex_t<input_t> x) {
		sharedmem[idx] = x.real();
		sharedmem[idx + WorkgroupSize] = x.imag();
	}
	
	nbl::hlsl::complex_t<input_t> get(uint32_t idx) {
		nbl::hlsl::complex_t<input_t> retVal = {sharedmem[idx], sharedmem[idx + WorkgroupSize]};
		return retVal;
	}

	void workgroupExecutionAndMemoryBarrier() {
		AllMemoryBarrierWithGroupSync();
    }
};

struct Accessor {
	void set(uint32_t idx, nbl::hlsl::complex_t<input_t> x) {
		vk::RawBufferStore< vector<input_t, 2> >(pushConstants.outputAddress + sizeof(vector<input_t, 2>) * idx, vector<input_t, 2>(x.real(), x.imag()));
	}
	
	nbl::hlsl::complex_t<input_t> get(uint32_t idx) {
		vector<input_t, 2> aux = vk::RawBufferLoad< vector<input_t, 2> >(pushConstants.inputAddress + sizeof(vector<input_t, 2>) * idx);
		nbl::hlsl::complex_t<input_t> retVal = {aux.x, aux.y};
		return retVal;
	}

	void workgroupExecutionAndMemoryBarrier() {
		AllMemoryBarrierWithGroupSync();
    }
};

[numthreads(WorkgroupSize,1,1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{

	Accessor accessor;
	SharedMemoryAccessor sharedmemAccessor;

	// Workgroup	

	nbl::hlsl::workgroup::FFT<2, true, input_t>::template __call<Accessor, SharedMemoryAccessor>(accessor, sharedmemAccessor);
	accessor.workgroupExecutionAndMemoryBarrier();
	nbl::hlsl::workgroup::FFT<2, false, input_t>::template __call<Accessor, SharedMemoryAccessor>(accessor, sharedmemAccessor);	

}