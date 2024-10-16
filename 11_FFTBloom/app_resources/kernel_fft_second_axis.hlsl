#include "common.hlsl"
#include "nbl/builtin/hlsl/workgroup/fft.hlsl"

// TODO: There's a lot of redundant stuff in every FFT file, I'd like to move that to another file that I can sourceFmt at runtime then include in all of them (something like 
// a runtime common.hlsl)

/*
 * Remember we have these defines:
 * _NBL_HLSL_WORKGROUP_SIZE_
 * ELEMENTS_PER_THREAD
 * (may be defined) USE_HALF_PRECISION
 * KERNEL_SCALE
*/

[[vk::push_constant]] PushConstantData pushConstants;

#ifdef USE_HALF_PRECISION
#define scalar_t float16_t
#else
#define scalar_t float32_t
#endif

#define IMAGE_SIDE_LENGTH (_NBL_HLSL_WORKGROUP_SIZE_ * ELEMENTS_PER_THREAD)

groupshared uint32_t sharedmem[workgroup::fft::SharedMemoryDWORDs<scalar_t, _NBL_HLSL_WORKGROUP_SIZE_>];

// Users MUST define this method for FFT to work
uint32_t3 glsl::gl_WorkGroupSize() { return uint32_t3(_NBL_HLSL_WORKGROUP_SIZE_, 1, 1); }

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

// ---------------------------------------------------- Utils ---------------------------------------------------------
uint32_t colMajorOffset(uint32_t x, uint32_t y)
{
	return x * (IMAGE_SIDE_LENGTH / 2) | y;
}

uint32_t rowMajorOffset(uint32_t x, uint32_t y)
{
	return y * IMAGE_SIDE_LENGTH | x;
}

uint32_t getColMajorChannelStartAddress(uint32_t channel)
{
	return pushConstants.colMajorBufferAddress + channel * IMAGE_SIDE_LENGTH * IMAGE_SIDE_LENGTH / 2 * sizeof(complex_t<scalar_t>);
}

uint32_t getRowMajorChannelStartAddress(uint32_t channel)
{
	return pushConstants.rowMajorBufferAddress + channel * IMAGE_SIDE_LENGTH * IMAGE_SIDE_LENGTH / 2 * sizeof(complex_t<scalar_t>);
}

struct PreloadedAccessorBase {

	void set(uint32_t idx, nbl::hlsl::complex_t<scalar_t> value)
	{
		preloaded[idx / _NBL_HLSL_WORKGROUP_SIZE_] = value;
	}

	void get(uint32_t idx, NBL_REF_ARG(nbl::hlsl::complex_t<scalar_t>) value)
	{
		value = preloaded[idx / _NBL_HLSL_WORKGROUP_SIZE_]
	}

	void memoryBarrier()
	{
		// only one workgroup is touching any memory it wishes to trade
		spirv::memoryBarrier(spv::ScopeWorkgroup, spv::MemorySemanticsAcquireReleaseMask | spv::MemorySemanticsUniformMemoryMask);
	}

	complex_t<scalar_t> preloaded[ELEMENTS_PER_THREAD];
};


// ------------------------------------------ SECOND AXIS FFT -------------------------------------------------------------

// This time each Workgroup will compute the FFT along a horizontal line (fixed y for the whole Workgroup). We get the y coordinate for the
// row a workgroup is working on via `gl_WorkGroupID().x`. We have launched IMAGE_SIDE_LENGTH / 2 workgroups, as there are exactly 
// that amount of rows in the buffer. We have to keep this in mind: What's stored as the first row is actually `Z + iN`, 
// where `Z` is the actual 0th row and `N` is the Nyquist row (the one with index IMAGE_SIDE_LENGTH / 2). Those are packed together
// so they need to be unpacked properly after FFT like we did earlier.

struct PreloadedSecondAxisAccessor : PreloadedAccessorBase
{
	void preload(uint32_t channel)
	{
		const uint32_t startAddress = getColMajorChannelStartAddress(channel);

		for (uint32_t elementIndex = 0; elementIndex < ELEMENTS_PER_THREAD; elementIndex++)
		{
			const uint32_t index = _NBL_HLSL_WORKGROUP_SIZE_ * elementIndex | workgroup::SubgroupContiguousIndex();
			preloaded[elementIndex] = vk::RawBufferLoad<complex_t<scalar_t> >(startAddress + colMajorOffset(index, gl_WorkGroupID().x) * sizeof(complex_t<scalar_t>));
		}
	}

	// Util to write values to output buffer in row major order
	void storeRowMajor(uint32_t startAddress, uint32_t index, NBL_CONST_REF_ARG(complex_t<scalar_t>) value)
	{
		vk::RawBufferStore<complex_t<scalar_t> >(startAddress + rowMajorOffset(index, gl_WorkGroupID().x) * sizeof(complex_t<scalar_t>), value);
	}

	// Save a column back in row major order. Remember that the first row (one with `gl_WorkGroupID().x == 0`) will actually hold the packed FFT of Zero and Nyquist rows.
	void unload(uint32_t channel)
	{
		const uint32_t startAddress = getRowMajorChannelStartAddress(channel);

		for (uint32_t elementIndex = 0; elementIndex < ELEMENTS_PER_THREAD; elementIndex++)
		{
			const uint32_t index = _NBL_HLSL_WORKGROUP_SIZE_ * elementIndex | workgroup::SubgroupContiguousIndex();
			storeRowMajor(startAddress, index, preloaded[elementIndex]);
		}
	}
};

void secondAxisFFT()
{
	SharedMemoryAccessor sharedmemAccessor;

	PreloadedSecondAxisAccessor preloadedAccessor;
	for (uint32_t channel = 0; channel < CHANNELS; channel++)
	{
		preloadedAccessor.preload(channel);
		FFT<ELEMENTS_PER_THREAD, false, _NBL_HLSL_WORKGROUP_SIZE_, scalar_t>::template __call(preloadedAccessor, sharedmemAccessor);
		preloadedAccessor.unload(channel);
	}
}

[numthreads(_NBL_HLSL_WORKGROUP_SIZE_, 1, 1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
	secondAxisFFT();
}