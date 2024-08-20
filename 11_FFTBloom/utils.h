#ifndef FFT_UTILS
#define FFT_UTILS

// Everything in this file will probably be eventually refactored into the FFT Class if this is made into an extension again
#include "nbl/application_templates/MonoDeviceApplication.hpp"
#include "nbl/application_templates/MonoAssetManagerAndBuiltinResourceApplication.hpp"


using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace video;


static inline asset::VkExtent3D padDimensions(asset::VkExtent3D dimension)
{
	for (auto i = 0u; i < 3u; i++)
	{
		auto& coord = (&dimension.width)[i];
		if (coord <= 1u)
			continue;
		coord = core::roundUpToPoT(coord);
	}
	return dimension;
}

static inline size_t getOutputBufferSize(const asset::VkExtent3D& inputDimensions, uint32_t numChannels, bool halfFloats = true)
{
	auto paddedDims = padDimensions(inputDimensions);
	size_t retval = paddedDims.width * paddedDims.height * paddedDims.depth * numChannels;
	return retval * (halfFloats ? sizeof(uint16_t) : sizeof(uint32_t)) * 2; // 2 because it's complex here
}



#endif