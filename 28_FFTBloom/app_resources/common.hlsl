#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/workgroup/fft.hlsl"

NBL_CONSTEXPR_STATIC_INLINE uint32_t Channels = 3;

using namespace nbl::hlsl;

// Necessary bits for each int type (except BDAs) are being generous and considering a resolution of up to 4k
// Also, an extra bit is given since it fits: they're uints being passed as ints so we don't litter the code with int32 casts
// The extra bit is to make sure MSB is a 0 and it doesn't sign extend and give negative numbers when using the maximum amount of considered bits
struct PushConstantData
{
	// After running FFT along a column, we want to store the result in column major order for coalesced writes, and similarly after running an FFT in row major order
	// All FFTs that read from a buffer and write to a buffer (kernel second FFT, image second FFT + conv + IFFT) read in one order and write in another. 
	// Since workgroups start and finish at arbitrary times it's not possible to pickup the data from a buffer in one layout and write it back in another layout due to
	// possibility of writing over data that still hasn't been read by some workgroups.
	uint64_t colMajorBufferAddress;
	uint64_t rowMajorBufferAddress;
	// To save some work, we don't mirror the image along both directions when doing the FFT. This means that when doing the FFT along the second axis, we do an FFT of length
	// `RoundUpToPoT(imageRowLength + kernelPadding)` where `imageRowLength` is the actual length of the image along the second axis. We need it to keep track of the image's original dimension.
	// The following three fields being push constants allow dynamic resizing of the image without recompiling shaders (limited by the FFT length)
	int32_t imageRowLength : 13; 
	int32_t imageHalfRowLength : 12;
	// Actually only needs at worst 10 bits, but we don't pack it into a bitfield so we can use offsetof and update only this field from CPP side
	// Alternatively, we could do the packing/unpacking manually to save 32 bits
	int32_t padding;
	// Used by IFFT to tell if an index belongs to an image or is in the padding
	int32_t imageColumnLength : 13;
	int32_t halfPadding : 10;
	float32_t2 imageHalfPixelSize;
	float32_t2 imagePixelSize;
	float32_t imageTwoPixelSize_x;
	float32_t imageWorkgroupSizePixelSize_y;
	float32_t interpolatingFactor;
};

#ifdef __HLSL_VERSION

#include "nbl/builtin/hlsl/bda/legacy_bda_accessor.hlsl"

[[vk::push_constant]] PushConstantData pushConstants;

using scalar_t = ShaderConstevalParameters::scalar_t;
using FFTParameters = workgroup::fft::ConstevalParameters<ShaderConstevalParameters::ElementsPerInvocationLog2, ShaderConstevalParameters::WorkgroupSizeLog2, scalar_t>;

using FFTIndexingUtils = workgroup::fft::FFTIndexingUtils<FFTParameters::ElementsPerInvocationLog2, FFTParameters::WorkgroupSizeLog2>;

// Users MUST define this method for FFT to work - workgroup::SubgroupContiguousIndex also needs it defined
uint32_t3 glsl::gl_WorkGroupSize() { return uint32_t3(uint32_t(FFTParameters::WorkgroupSize), 1, 1); }

#endif
