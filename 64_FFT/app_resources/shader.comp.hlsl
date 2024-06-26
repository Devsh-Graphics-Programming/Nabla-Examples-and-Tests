#include "common.hlsl"
#include "nbl/builtin/hlsl/workgroup/fft.hlsl"
//#include "nbl/builtin/hlsl/subgroup/fft.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_basic.hlsl"

[[vk::push_constant]] PushConstantData pushConstants;

groupshared output_t sharedmem[2 * WorkgroupSize];

struct SharedMemoryAccessor {
	void set(uint32_t idx, uint x) {
		sharedmem[idx] = x;
	}
	
	uint get(uint32_t idx) {
		return sharedmem[idx];
	}

	void workgroupExecutionAndMemoryBarrier() {
        AllMemoryBarrierWithGroupSync();
    }
};

struct Accessor {
	void set(uint32_t idx, uint x) {
		vk::RawBufferStore<uint>(pushConstants.outputAddress + sizeof(uint) * idx, x);
	}
	
	uint get(uint32_t idx) {
		return vk::RawBufferLoad<uint>(pushConstants.inputAddress + sizeof(uint) * idx);
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