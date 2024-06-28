#include "common.hlsl"
#include "nbl/builtin/hlsl/workgroup/fft.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_basic.hlsl"

// Users must define this function for the FFT to work
namespace nbl
{
namespace hlsl
{
namespace glsl 
{

// Define this method from glsl_compat/core.hlsl 
uint32_t3 gl_WorkGroupSize() {
    return uint32_t3(_NBL_HLSL_WORKGROUP_SIZE_, 1, 1);
}

} //namespace glsl
} //namespace hlsl
} //namespace nbl


[[vk::push_constant]] PushConstantData pushConstants;

groupshared uint32_t sharedmem[2 * WorkgroupSize];

struct SharedMemoryAccessor {
	void set(uint32_t idx, uint32_t x) {
		sharedmem[idx] = x;
	}
	
	uint32_t get(uint32_t idx) {
		return sharedmem[idx];
	}

	void workgroupExecutionAndMemoryBarrier() {
        AllMemoryBarrierWithGroupSync();
    }
};

struct Accessor {
	void set(uint32_t idx, uint32_t x) {
		vk::RawBufferStore<uint32_t>(pushConstants.outputAddress + sizeof(uint32_t) * idx, x);
	}
	
	uint32_t get(uint32_t idx) {
		return vk::RawBufferLoad<uint32_t>(pushConstants.inputAddress + sizeof(uint32_t) * idx);
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

	nbl::hlsl::workgroup::fft::FFT<2, true, input_t>::template __call<Accessor, SharedMemoryAccessor>(accessor, sharedmemAccessor);
	accessor.workgroupExecutionAndMemoryBarrier();
	nbl::hlsl::workgroup::fft::FFT<2, false, input_t>::template __call<Accessor, SharedMemoryAccessor>(accessor, sharedmemAccessor);	

}