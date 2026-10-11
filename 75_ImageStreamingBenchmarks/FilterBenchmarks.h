// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_THIS_EXAMPLE_FILTER_BENCHMARKS_H_INCLUDED_
#define _NBL_THIS_EXAMPLE_FILTER_BENCHMARKS_H_INCLUDED_

#include "nbl/examples/examples.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <immintrin.h>
#include <vector>

namespace nbl::this_example
{

// same as the promotion swizzle `ImageRegionIterator` uses for `IUtilities` uploads
template<unsigned int SRC_CHANNELS>
struct SPromotionSwizzle
{
	static constexpr asset::ICPUImageView::SComponentMapping componentMapping = {
		asset::ICPUImageView::SComponentMapping::ES_R,
		SRC_CHANNELS>1 ? asset::ICPUImageView::SComponentMapping::ES_G:asset::ICPUImageView::SComponentMapping::ES_ZERO,
		SRC_CHANNELS>2 ? asset::ICPUImageView::SComponentMapping::ES_B:asset::ICPUImageView::SComponentMapping::ES_ZERO,
		SRC_CHANNELS>3 ? asset::ICPUImageView::SComponentMapping::ES_A:asset::ICPUImageView::SComponentMapping::ES_ONE
	};

	template<typename InT, typename OutT>
	void operator()(const InT* in, OutT* out) const
	{
		using in_t = std::conditional_t<std::is_void_v<InT>, uint64_t, InT>;
		using out_t = std::conditional_t<std::is_void_v<OutT>, uint64_t, OutT>;

		reinterpret_cast<out_t*>(out)[0u] = reinterpret_cast<const in_t*>(in)[0u];

		if constexpr (SRC_CHANNELS > 1)
			reinterpret_cast<out_t*>(out)[1u] = reinterpret_cast<const in_t*>(in)[1u];
		else
			reinterpret_cast<out_t*>(out)[1u] = static_cast<in_t>(0);

		if constexpr (SRC_CHANNELS > 2)
			reinterpret_cast<out_t*>(out)[2u] = reinterpret_cast<const in_t*>(in)[2u];
		else
			reinterpret_cast<out_t*>(out)[2u] = static_cast<in_t>(0);

		if constexpr (SRC_CHANNELS > 3)
			reinterpret_cast<out_t*>(out)[3u] = reinterpret_cast<const in_t*>(in)[3u];
		else
			reinterpret_cast<out_t*>(out)[3u] = static_cast<in_t>(1);
	}
};

// same swizzle without `componentMapping`, so the convert filter cannot take its byte reshuffle path
template<unsigned int SRC_CHANNELS>
struct SPromotionSwizzleGeneric
{
	template<typename InT, typename OutT>
	void operator()(const InT* in, OutT* out) const
	{
		SPromotionSwizzle<SRC_CHANNELS>{}.template operator()<InT,OutT>(in,out);
	}
};

// CPU image filter and `IUtilities` format conversion benchmarks for issue #889, run the example with `--filters`
// Every result line starts with `FILTERBENCH` so before/after logs can be diffed, `hash` is of the destination bytes
class CFilterBenchmarks
{
		static constexpr uint8_t Sentinel = 0xa5u;
		static constexpr uint32_t BigWidth = 20000u, BigHeight = 7000u;
		static constexpr uint32_t MidWidth = 4096u, MidHeight = 2048u;
		static constexpr uint32_t SmallWidth = 2048u, SmallHeight = 1024u;
		static constexpr size_t MappedBytes = size_t(SmallWidth)*SmallHeight*4u;

		struct SLayout
		{
			asset::E_FORMAT format;
			asset::VkExtent3D extent;
			uint32_t layers = 1u;
			uint32_t rowLength = 0u;
			uint32_t imageHeight = 0u;

			inline core::vector3du32_SIMD blockStrides() const
			{
				const asset::TexelBlockInfo info(format);
				return info.convertTexelsToBlocks(core::vector3du32_SIMD(rowLength ? rowLength:extent.width,imageHeight ? imageHeight:extent.height,extent.depth));
			}
			inline size_t bytes() const
			{
				const auto strides = blockStrides();
				return size_t(strides.x)*strides.y*strides.z*layers*asset::getTexelOrBlockBytesize(format);
			}
			inline size_t blockOffset(uint32_t x, uint32_t y, uint32_t z, uint32_t layer) const
			{
				const auto strides = blockStrides();
				return (((size_t(layer)*strides.z+z)*strides.y+y)*strides.x+x)*asset::getTexelOrBlockBytesize(format);
			}
		};

		struct SMapped
		{
			core::smart_refctd_ptr<video::IGPUBuffer> buffer;
			video::IDeviceMemoryAllocator::SAllocation allocation;
			uint8_t* ptr = nullptr;
		};

		struct STiming
		{
			double minMs = 0.0;
			double medianMs = 0.0;
		};

	public:
		CFilterBenchmarks(video::ILogicalDevice* device, video::IQueue* queue, system::ILogger* logger) : m_device(device), m_queue(queue), m_logger(logger) {}
		~CFilterBenchmarks()
		{
			for (auto* mapped : {&m_sysram,&m_bar})
			if (mapped->ptr)
				mapped->allocation.memory->unmap();
		}

		bool run(const bool quick)
		{
			m_logger->log("FILTERBENCH begin", system::ILogger::ELL_PERFORMANCE);
			const auto* physDev = m_device->getPhysicalDevice();
			const uint32_t hostVisible = physDev->getHostVisibleMemoryTypeBits();
			const uint32_t deviceLocal = physDev->getDeviceLocalMemoryTypeBits();
			const uint32_t hostCached = physDev->getMemoryTypeBitsFromMemoryTypeFlags(video::IDeviceMemoryAllocation::EMPF_HOST_CACHED_BIT);
			bool ok = allocateMapped(m_sysram,hostVisible&~deviceLocal&~hostCached,"SysRAM");
			ok = allocateMapped(m_bar,hostVisible&deviceLocal&~hostCached,"BAR") && ok;
			ok = copyCases() && ok;
			ok = convertCases() && ok;
			if (!quick)
				ablations();
			ok = iutilitiesCases() && ok;
			m_logger->log("FILTERBENCH end %s",ok ? system::ILogger::ELL_PERFORMANCE:system::ILogger::ELL_ERROR,ok ? "PASS":"FAIL");
			return ok;
		}

	private:
		// ---------------------------------------------------------------- helpers
		static void fillPattern(uint8_t* data, size_t size, uint32_t seed)
		{
			uint32_t state = 0x9e3779b9u^(seed*0x85ebca6bu+1u);
			size_t i = 0u;
			for (; i+4u<=size; i+=4u)
			{
				state ^= state<<13; state ^= state>>17; state ^= state<<5;
				memcpy(data+i,&state,4u);
			}
			for (; i<size; i++)
				data[i] = uint8_t(i*31u+seed);
		}

		static uint64_t hashBytes(const uint8_t* data, size_t size)
		{
			uint64_t h = 0xcbf29ce484222325ull^size;
			size_t i = 0u;
			for (; i+8u<=size; i+=8u)
			{
				uint64_t v;
				memcpy(&v,data+i,8u);
				h = (h^v)*0x100000001b3ull;
				h ^= h>>32;
			}
			for (; i<size; i++)
				h = (h^data[i])*0x100000001b3ull;
			return h;
		}

		template<typename F>
		static STiming time(F&& f, const uint32_t reps)
		{
			f();
			std::vector<double> ms(reps);
			for (auto& m : ms)
			{
				const auto start = std::chrono::steady_clock::now();
				f();
				m = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
			}
			std::sort(ms.begin(),ms.end());
			return {ms.front(),ms[ms.size()/2u]};
		}

		void report(const char* name, const char* dst, const char* policy, const STiming& t, size_t bytesWritten, size_t texels, uint64_t hash)
		{
			const double gbps = double(bytesWritten)/(t.minMs*1e-3)/1e9;
			const double nsPerTexel = t.minMs*1e6/double(texels);
			m_logger->log("FILTERBENCH %-52s %-6s %-4s min %10.3f ms  med %10.3f ms  %8.3f GB/s  %8.3f ns/texel  hash %016llx",system::ILogger::ELL_PERFORMANCE,
				name,dst,policy,t.minMs,t.medianMs,gbps,nsPerTexel,(unsigned long long)hash);
		}

		void verdict(const char* name, bool ok, const char* detail = "")
		{
			m_logger->log("FILTERBENCH CHECK %-46s %s %s",ok ? system::ILogger::ELL_PERFORMANCE:system::ILogger::ELL_ERROR,name,ok ? "PASS":"FAIL",detail);
		}

		static core::smart_refctd_ptr<asset::ICPUImage> wrap(const SLayout& layout, void* data)
		{
			asset::ICPUImage::SCreationParams params = {};
			params.type = layout.extent.depth>1u ? asset::IImage::E_TYPE::ET_3D:asset::IImage::E_TYPE::ET_2D;
			params.samples = asset::IImage::E_SAMPLE_COUNT_FLAGS::ESCF_1_BIT;
			params.format = layout.format;
			params.extent = layout.extent;
			params.mipLevels = 1u;
			params.arrayLayers = layout.layers;
			params.flags = asset::IImage::ECF_NONE;
			auto image = asset::ICPUImage::create(std::move(params));
			if (!image)
				return nullptr;

			auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<asset::ICPUImage::SBufferCopy>>(1u);
			auto& region = regions->front();
			region = {};
			region.bufferOffset = 0ull;
			region.bufferRowLength = layout.rowLength;
			region.bufferImageHeight = layout.imageHeight;
			region.imageSubresource.aspectMask = asset::IImage::EAF_COLOR_BIT;
			region.imageSubresource.mipLevel = 0u;
			region.imageSubresource.baseArrayLayer = 0u;
			region.imageSubresource.layerCount = layout.layers;
			region.imageOffset = {0,0,0};
			region.imageExtent = layout.extent;

			auto buffer = asset::ICPUBuffer::create({{layout.bytes()},data,core::getNullMemoryResource()},core::adopt_memory);
			if (!buffer || !image->setBufferAndRegions(std::move(buffer),regions))
				return nullptr;
			return image;
		}

		bool allocateMapped(SMapped& out, const uint32_t memoryTypeBits, const char* name)
		{
			video::IGPUBuffer::SCreationParams params = {};
			params.size = MappedBytes;
			params.usage = video::IGPUBuffer::EUF_TRANSFER_SRC_BIT;
			out.buffer = m_device->createBuffer(std::move(params));
			if (!out.buffer)
				return false;
			auto reqs = out.buffer->getMemoryReqs();
			reqs.memoryTypeBits &= memoryTypeBits;
			if (!reqs.memoryTypeBits)
			{
				m_logger->log("FILTERBENCH no %s memory type",system::ILogger::ELL_ERROR,name);
				return false;
			}
			out.allocation = m_device->allocate(reqs,{out.buffer.get(),video::IDeviceMemoryAllocation::EMAF_NONE});
			if (!out.allocation.isValid())
				return false;
			out.ptr = reinterpret_cast<uint8_t*>(out.allocation.memory->map({0ull,out.allocation.memory->getAllocationSize()},video::IDeviceMemoryAllocation::EMCAF_READ_AND_WRITE));
			m_logger->log("FILTERBENCH mapped %s memory flags 0x%x",system::ILogger::ELL_PERFORMANCE,name,uint32_t(out.allocation.memory->getMemoryPropertyFlags().value));
			return out.ptr;
		}

		// destination memory a case writes into, staging memory only fits the small cases
		uint8_t* destination(const char* dst, std::vector<uint8_t>& heap, const size_t size)
		{
			if (!strcmp(dst,"SysRAM"))
				return size<=MappedBytes ? m_sysram.ptr:nullptr;
			if (!strcmp(dst,"BAR"))
				return size<=MappedBytes ? m_bar.ptr:nullptr;
			heap.resize(size);
			return heap.data();
		}

		// ---------------------------------------------------------------- copy filter
		struct SCopyBox
		{
			core::vectorSIMDu32 inOffsetBaseLayer = core::vectorSIMDu32(0u,0u,0u,0u);
			core::vectorSIMDu32 outOffsetBaseLayer = core::vectorSIMDu32(0u,0u,0u,0u);
			asset::VkExtent3D extent;
			uint32_t layers = 1u;
		};

		// independent per block reference, also requires every byte outside the box to be untouched
		static size_t copyMismatches(const SLayout& inLayout, const uint8_t* in, const SLayout& outLayout, const uint8_t* out, const SCopyBox& box)
		{
			const asset::TexelBlockInfo info(inLayout.format);
			const size_t blockSize = info.getBlockByteSize();
			const auto boxIn = info.convertTexelsToBlocks(core::vector3du32_SIMD(box.inOffsetBaseLayer.x,box.inOffsetBaseLayer.y,box.inOffsetBaseLayer.z));
			const auto boxOut = info.convertTexelsToBlocks(core::vector3du32_SIMD(box.outOffsetBaseLayer.x,box.outOffsetBaseLayer.y,box.outOffsetBaseLayer.z));
			const auto boxExtent = info.convertTexelsToBlocks(core::vector3du32_SIMD(box.extent.width,box.extent.height,box.extent.depth));
			const auto outExtent = info.convertTexelsToBlocks(core::vector3du32_SIMD(outLayout.extent.width,outLayout.extent.height,outLayout.extent.depth));
			const auto outStrides = outLayout.blockStrides();
			size_t mismatches = 0u;
			for (uint32_t layer=0u; layer<outLayout.layers; layer++)
			for (uint32_t z=0u; z<outStrides.z; z++)
			for (uint32_t y=0u; y<outStrides.y; y++)
			for (uint32_t x=0u; x<outStrides.x; x++)
			{
				const uint8_t* got = out+outLayout.blockOffset(x,y,z,layer);
				const bool inBox = layer>=box.outOffsetBaseLayer.w && layer<box.outOffsetBaseLayer.w+box.layers &&
					x>=boxOut.x && x<boxOut.x+boxExtent.x && y>=boxOut.y && y<boxOut.y+boxExtent.y && z>=boxOut.z && z<boxOut.z+boxExtent.z;
				if (inBox && x<outExtent.x && y<outExtent.y && z<outExtent.z)
				{
					const uint8_t* expected = in+inLayout.blockOffset(x-boxOut.x+boxIn.x,y-boxOut.y+boxIn.y,z-boxOut.z+boxIn.z,layer-box.outOffsetBaseLayer.w+box.inOffsetBaseLayer.w);
					mismatches += memcmp(got,expected,blockSize)!=0;
				}
				else
				{
					for (size_t b=0u; b<blockSize; b++)
						mismatches += got[b]!=Sentinel;
				}
			}
			return mismatches;
		}

		template<typename Policy>
		bool copyCase(const char* name, const char* dst, const char* policyName, Policy&& policy, const SLayout& inLayout, const SLayout& outLayout, const SCopyBox& box, const uint32_t reps)
		{
			std::vector<uint8_t> in(inLayout.bytes()), heap;
			fillPattern(in.data(),in.size(),1u);
			uint8_t* out = destination(dst,heap,outLayout.bytes());
			if (!out)
				return true;
			memset(out,Sentinel,outLayout.bytes());

			auto inImage = wrap(inLayout,in.data());
			auto outImage = wrap(outLayout,out);
			if (!inImage || !outImage)
			{
				verdict(name,false,"image creation failed");
				return false;
			}
			asset::CCopyImageFilter::state_type state;
			state.extent = box.extent;
			state.layerCount = box.layers;
			state.inImage = inImage.get();
			state.outImage = outImage.get();
			state.inOffsetBaseLayer = box.inOffsetBaseLayer;
			state.outOffsetBaseLayer = box.outOffsetBaseLayer;
			state.inMipLevel = 0u;
			state.outMipLevel = 0u;

			bool executed = true;
			const auto t = time([&]() -> void {executed = asset::CCopyImageFilter::execute(policy,&state) && executed;},reps);

			std::vector<uint8_t> readback(out,out+outLayout.bytes());
			const size_t texels = size_t(box.extent.width)*box.extent.height*box.extent.depth*box.layers;
			const asset::TexelBlockInfo info(inLayout.format);
			const auto boxBlocks = info.convertTexelsToBlocks(core::vector3du32_SIMD(box.extent.width,box.extent.height,box.extent.depth));
			const size_t bytesWritten = size_t(boxBlocks.x)*boxBlocks.y*boxBlocks.z*box.layers*info.getBlockByteSize();
			report(name,dst,policyName,t,bytesWritten,texels,hashBytes(readback.data(),readback.size()));
			const size_t mismatches = copyMismatches(inLayout,in.data(),outLayout,readback.data(),box);
			const bool ok = executed && mismatches==0u;
			char detail[64];
			snprintf(detail,sizeof(detail),"(%zu bad blocks/bytes)",mismatches);
			verdict(name,ok,detail);
			return ok;
		}

		bool copyCases()
		{
			using namespace asset;
			bool ok = true;
			{
				const SLayout layout = {EF_R8G8B8A8_UNORM,{BigWidth,BigHeight,1u}};
				ok = copyCase("copy rgba8 20000x7000 tight","heap","par",core::execution::par_unseq,layout,layout,{.extent=layout.extent},3u) && ok;
			}
			{
				const SLayout layout = {EF_R8G8B8A8_UNORM,{MidWidth,MidHeight,1u}};
				ok = copyCase("copy rgba8 4096x2048 tight","heap","seq",core::execution::seq,layout,layout,{.extent=layout.extent},5u) && ok;
				ok = copyCase("copy rgba8 4096x2048 tight","heap","par",core::execution::par_unseq,layout,layout,{.extent=layout.extent},5u) && ok;
			}
			{
				const SLayout layout = {EF_R8G8B8A8_UNORM,{SmallWidth,SmallHeight,1u}};
				for (const char* dst : {"heap","SysRAM","BAR"})
					ok = copyCase("copy rgba8 2048x1024 tight",dst,"par",core::execution::par_unseq,layout,layout,{.extent=layout.extent},3u) && ok;
			}
			{
				const SLayout inLayout = {EF_R8G8B8A8_UNORM,{1024u,768u,1u},3u,1100u,800u};
				const SLayout outLayout = {EF_R8G8B8A8_UNORM,{640u,400u,1u},2u,700u,420u};
				const SCopyBox box = {core::vectorSIMDu32(37u,19u,0u,1u),core::vectorSIMDu32(5u,7u,0u,0u),{601u,383u,1u},2u};
				ok = copyCase("copy rgba8 clipped padded 2 of 3 layers",
					"heap","seq",core::execution::seq,inLayout,outLayout,box,5u) && ok;
				ok = copyCase("copy rgba8 clipped padded 2 of 3 layers","heap","par",core::execution::par_unseq,inLayout,outLayout,box,5u) && ok;
			}
			{
				const SLayout layout = {EF_BC7_UNORM_BLOCK,{8192u,8192u,1u}};
				ok = copyCase("copy bc7 8192x8192 tight","heap","par",core::execution::par_unseq,layout,layout,{.extent=layout.extent},3u) && ok;
				const SLayout inLayout = {EF_BC7_UNORM_BLOCK,{1024u,1024u,1u},1u,1040u,0u};
				const SLayout outLayout = {EF_BC7_UNORM_BLOCK,{512u,512u,1u},1u,528u,0u};
				const SCopyBox box = {core::vectorSIMDu32(64u,32u,0u,0u),core::vectorSIMDu32(16u,8u,0u,0u),{256u,200u,1u},1u};
				ok = copyCase("copy bc7 clipped padded","heap","seq",core::execution::seq,inLayout,outLayout,box,5u) && ok;
			}
			{
				const SLayout inLayout = {EF_R8G8B8A8_UNORM,{256u,256u,64u},1u,0u,260u};
				const SLayout outLayout = {EF_R8G8B8A8_UNORM,{256u,256u,64u}};
				ok = copyCase("copy rgba8 3d 256x256x64 padded slices","heap","par",core::execution::par_unseq,inLayout,outLayout,{.extent=outLayout.extent},5u) && ok;
			}
			return ok;
		}

		// ---------------------------------------------------------------- convert filter
		// the promotion `IUtilities` asks for: source channels kept bit exact, missing ones zero, alpha one
		static size_t promotionMismatches(const SLayout& inLayout, const uint8_t* in, const uint8_t* out, const uint32_t srcChannels)
		{
			const size_t texels = size_t(inLayout.extent.width)*inLayout.extent.height;
			size_t mismatches = 0u;
			for (size_t t=0u; t<texels; t++)
			{
				const uint8_t* s = in+t*srcChannels;
				const uint8_t* d = out+t*4u;
				bool bad = d[3]!=255u;
				for (uint32_t c=0u; c<3u; c++)
					bad |= d[c]!=(c<srcChannels ? s[c]:0u);
				mismatches += bad;
			}
			return mismatches;
		}

		template<typename Filter, typename Policy>
		bool convertCase(const char* name, const char* dst, const char* policyName, Policy&& policy, const SLayout& inLayout, const SLayout& outLayout, const uint32_t reps, const int32_t promotedChannels)
		{
			// padded because the 3 byte texel decodes read 4 bytes
			std::vector<uint8_t> in(inLayout.bytes()+4u), heap;
			fillPattern(in.data(),in.size(),2u);
			uint8_t* out = destination(dst,heap,outLayout.bytes());
			if (!out)
				return true;
			memset(out,Sentinel,outLayout.bytes());

			auto inImage = wrap(inLayout,in.data());
			auto outImage = wrap(outLayout,out);
			if (!inImage || !outImage)
			{
				verdict(name,false,"image creation failed");
				return false;
			}
			typename Filter::state_type state;
			state.extent = inLayout.extent;
			state.layerCount = inLayout.layers;
			state.inImage = inImage.get();
			state.outImage = outImage.get();
			state.inOffsetBaseLayer = core::vectorSIMDu32(0u,0u,0u,0u);
			state.outOffsetBaseLayer = core::vectorSIMDu32(0u,0u,0u,0u);
			state.inMipLevel = 0u;
			state.outMipLevel = 0u;

			bool executed = true;
			const auto t = time([&]() -> void {executed = Filter::execute(policy,&state) && executed;},reps);

			std::vector<uint8_t> readback(out,out+outLayout.bytes());
			const size_t texels = size_t(inLayout.extent.width)*inLayout.extent.height*inLayout.extent.depth*inLayout.layers;
			report(name,dst,policyName,t,outLayout.bytes(),texels,hashBytes(readback.data(),readback.size()));
			if (promotedChannels<0)
			{
				verdict(name,executed,"(hash only)");
				return executed;
			}
			const size_t mismatches = promotionMismatches(inLayout,in.data(),readback.data(),uint32_t(promotedChannels));
			char detail[96];
			snprintf(detail,sizeof(detail),"(%zu of %zu texels differ from the exact promotion)",mismatches,texels);
			// the generic decode/encode path is not exact by construction, only reported
			if (strstr(name,"generic path"))
				m_logger->log("FILTERBENCH CHECK %-46s INFO %s",system::ILogger::ELL_PERFORMANCE,name,detail);
			else
				verdict(name,executed && mismatches==0u,detail);
			return executed;
		}

		bool convertCases()
		{
			using namespace asset;
			using runtime3_t = CSwizzleAndConvertImageFilter<EF_UNKNOWN,EF_UNKNOWN,SPromotionSwizzle<3u>>;
			using runtime1_t = CSwizzleAndConvertImageFilter<EF_UNKNOWN,EF_UNKNOWN,SPromotionSwizzle<1u>>;
			using runtime4_t = CSwizzleAndConvertImageFilter<EF_UNKNOWN,EF_UNKNOWN,SPromotionSwizzle<4u>>;
			using compileTimeSrgb_t = CSwizzleAndConvertImageFilter<EF_R8G8B8_SRGB,EF_R8G8B8A8_SRGB,SPromotionSwizzle<3u>>;
			using generic3_t = CSwizzleAndConvertImageFilter<EF_UNKNOWN,EF_UNKNOWN,SPromotionSwizzleGeneric<3u>>;
			bool ok = true;
			{
				const asset::VkExtent3D extent = {MidWidth,MidHeight,1u};
				ok = convertCase<generic3_t>("convert rgb8_srgb->rgba8_srgb 4096x2048 generic path","heap","seq",core::execution::seq,{EF_R8G8B8_SRGB,extent},{EF_R8G8B8A8_SRGB,extent},3u,3) && ok;
				ok = convertCase<generic3_t>("convert rgb8_unorm->rgba8_unorm 4096x2048 generic path","heap","seq",core::execution::seq,{EF_R8G8B8_UNORM,extent},{EF_R8G8B8A8_UNORM,extent},3u,3) && ok;
				ok = convertCase<generic3_t>("convert rgb8_srgb->rgba8_srgb 4096x2048 generic path","heap","par",core::execution::par_unseq,{EF_R8G8B8_SRGB,extent},{EF_R8G8B8A8_SRGB,extent},5u,3) && ok;
			}
			{
				const asset::VkExtent3D extent = {SmallWidth,SmallHeight,1u};
				for (const char* dst : {"heap","SysRAM","BAR"})
					ok = convertCase<generic3_t>("convert rgb8_srgb->rgba8_srgb 2048x1024 generic path",dst,"par",core::execution::par_unseq,{EF_R8G8B8_SRGB,extent},{EF_R8G8B8A8_SRGB,extent},3u,3) && ok;
				for (const char* dst : {"SysRAM","BAR"})
					ok = convertCase<generic3_t>("convert rgb8_srgb->rgba8_srgb 2048x1024 generic path",dst,"seq",core::execution::seq,{EF_R8G8B8_SRGB,extent},{EF_R8G8B8A8_SRGB,extent},3u,3) && ok;
			}
			{
				const asset::VkExtent3D extent = {BigWidth,BigHeight,1u};
				ok = convertCase<runtime3_t>("convert rgb8_srgb->rgba8_srgb 20000x7000","heap","par",core::execution::par_unseq,{EF_R8G8B8_SRGB,extent},{EF_R8G8B8A8_SRGB,extent},3u,3) && ok;
				ok = convertCase<runtime3_t>("convert rgb8_unorm->rgba8_unorm 20000x7000","heap","par",core::execution::par_unseq,{EF_R8G8B8_UNORM,extent},{EF_R8G8B8A8_UNORM,extent},3u,3) && ok;
			}
			{
				const asset::VkExtent3D extent = {MidWidth,MidHeight,1u};
				ok = convertCase<runtime3_t>("convert rgb8_srgb->rgba8_srgb 4096x2048","heap","seq",core::execution::seq,{EF_R8G8B8_SRGB,extent},{EF_R8G8B8A8_SRGB,extent},3u,3) && ok;
				ok = convertCase<runtime3_t>("convert rgb8_srgb->rgba8_srgb 4096x2048","heap","par",core::execution::par_unseq,{EF_R8G8B8_SRGB,extent},{EF_R8G8B8A8_SRGB,extent},5u,3) && ok;
				ok = convertCase<runtime3_t>("convert rgb8_unorm->rgba8_unorm 4096x2048","heap","seq",core::execution::seq,{EF_R8G8B8_UNORM,extent},{EF_R8G8B8A8_UNORM,extent},3u,3) && ok;
				ok = convertCase<compileTimeSrgb_t>("convert rgb8_srgb->rgba8_srgb 4096x2048 compile time","heap","seq",core::execution::seq,{EF_R8G8B8_SRGB,extent},{EF_R8G8B8A8_SRGB,extent},3u,3) && ok;
				ok = convertCase<runtime1_t>("convert r8_unorm->rgba8_unorm 4096x2048","heap","par",core::execution::par_unseq,{EF_R8_UNORM,extent},{EF_R8G8B8A8_UNORM,extent},5u,1) && ok;
				ok = convertCase<runtime4_t>("convert rgba8_unorm->rgba16_sfloat 4096x2048","heap","par",core::execution::par_unseq,{EF_R8G8B8A8_UNORM,extent},{EF_R16G16B16A16_SFLOAT,extent},5u,-1) && ok;
			}
			{
				const asset::VkExtent3D extent = {SmallWidth,SmallHeight,1u};
				for (const char* dst : {"heap","SysRAM","BAR"})
					ok = convertCase<runtime3_t>("convert rgb8_srgb->rgba8_srgb 2048x1024",dst,"par",core::execution::par_unseq,{EF_R8G8B8_SRGB,extent},{EF_R8G8B8A8_SRGB,extent},3u,3) && ok;
				for (const char* dst : {"heap","SysRAM","BAR"})
					ok = convertCase<runtime3_t>("convert rgb8_srgb->rgba8_srgb 2048x1024",dst,"seq",core::execution::seq,{EF_R8G8B8_SRGB,extent},{EF_R8G8B8A8_SRGB,extent},3u,3) && ok;
			}
			return ok;
		}

		// ---------------------------------------------------------------- single threaded ablations
		template<typename F>
		void ablation(const char* name, const char* dst, const size_t texels, const size_t bytesWritten, F&& f)
		{
			const auto t = time(f,5u);
			report(name,dst,"seq",t,bytesWritten,texels,0ull);
		}

		void ablations()
		{
			using namespace asset;
			const size_t midTexels = size_t(MidWidth)*MidHeight;
			const size_t smallTexels = size_t(SmallWidth)*SmallHeight;
			std::vector<uint8_t> rgb(midTexels*3u+1u), rgba(midTexels*4u), heap(midTexels*4u);
			fillPattern(rgb.data(),rgb.size(),3u);
			fillPattern(rgba.data(),rgba.size(),4u);
			volatile double doubleSink = 0.0;
			volatile uint64_t intSink = 0u;

			ablation("ablation memcpy rgba8 4096x2048","heap",midTexels,midTexels*4u,[&]() -> void {memcpy(heap.data(),rgba.data(),midTexels*4u);});
			for (auto* mapped : {&m_sysram,&m_bar})
			{
				const char* dst = mapped==&m_sysram ? "SysRAM":"BAR";
				ablation("ablation memcpy rgba8 2048x1024",dst,smallTexels,smallTexels*4u,[&]() -> void {memcpy(mapped->ptr,rgba.data(),smallTexels*4u);});
				for (const size_t piece : {size_t(1u)<<12,size_t(1u)<<14,size_t(1u)<<16,size_t(1u)<<18,size_t(1u)<<20})
				{
					char name[64];
					snprintf(name,sizeof(name),"ablation memcpy rgba8 2048x1024 in %zu KiB calls",piece>>10);
					ablation(name,dst,smallTexels,smallTexels*4u,[&]() -> void
					{
						for (size_t offset=0u; offset<smallTexels*4u; offset+=piece)
							memcpy(mapped->ptr+offset,rgba.data()+offset,piece);
					});
				}
			}

			{
				const SLayout layout = {EF_R8G8B8A8_UNORM,{MidWidth,MidHeight,1u}};
				auto image = wrap(layout,rgba.data());
				const auto& region = image->getRegions()[0];
				ablation("ablation executePerBlock iteration only 4096x2048","none",midTexels,0u,[&]() -> void
				{
					uint64_t sum = 0u;
					auto f = [&sum](uint32_t readBlockArrayOffset, core::vectorSIMDu32 readBlockPos) -> void {sum += readBlockArrayOffset+readBlockPos.x;};
					CBasicImageFilterCommon::executePerBlock(core::execution::seq,image.get(),region,f);
					intSink = intSink+sum;
				});
			}

			auto decodeLoop = [&](const E_FORMAT format) -> void
			{
				double sum = 0.0;
				for (size_t t=0u; t<midTexels; t++)
				{
					const void* pix[4] = {rgb.data()+t*3u,nullptr,nullptr,nullptr};
					double decoded[4];
					decodePixelsRuntime(format,pix,decoded,0u,0u);
					sum += decoded[0]+decoded[1]+decoded[2];
				}
				doubleSink = doubleSink+sum;
			};
			ablation("ablation decodePixelsRuntime rgb8_srgb 4096x2048","none",midTexels,0u,[&]() -> void {decodeLoop(EF_R8G8B8_SRGB);});
			ablation("ablation decodePixelsRuntime rgb8_unorm 4096x2048","none",midTexels,0u,[&]() -> void {decodeLoop(EF_R8G8B8_UNORM);});
			ablation("ablation decodePixels<rgb8_srgb> 4096x2048","none",midTexels,0u,[&]() -> void
			{
				double sum = 0.0;
				for (size_t t=0u; t<midTexels; t++)
				{
					const void* pix[4] = {rgb.data()+t*3u,nullptr,nullptr,nullptr};
					double decoded[4];
					decodePixels<EF_R8G8B8_SRGB,double>(pix,decoded,0u,0u);
					sum += decoded[0]+decoded[1]+decoded[2];
				}
				doubleSink = doubleSink+sum;
			});

			// encode varies its input per texel so nothing can be hoisted out of the loop
			auto encodeLoop = [&](const E_FORMAT format, uint8_t* out, const size_t texels, const bool viaTemporary) -> void
			{
				for (size_t t=0u; t<texels; t++)
				{
					const double value = double(t&0xffu)/255.0;
					const double input[4] = {value,1.0-value,0.5*value,1.0};
					if (viaTemporary)
					{
						uint32_t texel = 0u;
						encodePixelsRuntime(format,&texel,input);
						memcpy(out+t*4u,&texel,4u);
					}
					else
						encodePixelsRuntime(format,out+t*4u,input);
				}
			};
			ablation("ablation encodePixelsRuntime rgba8_srgb in place 4096x2048","heap",midTexels,midTexels*4u,[&]() -> void {encodeLoop(EF_R8G8B8A8_SRGB,heap.data(),midTexels,false);});
			ablation("ablation encodePixelsRuntime rgba8_unorm in place 4096x2048","heap",midTexels,midTexels*4u,[&]() -> void {encodeLoop(EF_R8G8B8A8_UNORM,heap.data(),midTexels,false);});
			for (auto* mapped : {&m_sysram,&m_bar})
			{
				const char* dst = mapped==&m_sysram ? "SysRAM":"BAR";
				ablation("ablation encodePixelsRuntime rgba8_unorm in place 2048x1024",dst,smallTexels,smallTexels*4u,[&]() -> void {encodeLoop(EF_R8G8B8A8_UNORM,mapped->ptr,smallTexels,false);});
				ablation("ablation encodePixelsRuntime rgba8_unorm via temp 2048x1024",dst,smallTexels,smallTexels*4u,[&]() -> void {encodeLoop(EF_R8G8B8A8_UNORM,mapped->ptr,smallTexels,true);});
			}

			auto shuffleLoop = [&](uint8_t* out, const size_t texels) -> void
			{
				for (size_t t=0u; t<texels; t++)
				{
					out[t*4u+0u] = rgb[t*3u+0u];
					out[t*4u+1u] = rgb[t*3u+1u];
					out[t*4u+2u] = rgb[t*3u+2u];
					out[t*4u+3u] = 0xffu;
				}
			};
			ablation("ablation byte shuffle rgb8->rgba8 4096x2048","heap",midTexels,midTexels*4u,[&]() -> void {shuffleLoop(heap.data(),midTexels);});
			for (auto* mapped : {&m_sysram,&m_bar})
				ablation("ablation byte shuffle rgb8->rgba8 2048x1024",mapped==&m_sysram ? "SysRAM":"BAR",smallTexels,smallTexels*4u,[&]() -> void {shuffleLoop(mapped->ptr,smallTexels);});
		}

		// ---------------------------------------------------------------- IUtilities end to end
		struct SGPUContext
		{
			core::smart_refctd_ptr<video::IUtilities> utils;
			core::smart_refctd_ptr<video::IGPUCommandPool> pool;
			core::smart_refctd_ptr<video::IGPUCommandBuffer> cmdbuf;
			core::smart_refctd_ptr<video::ISemaphore> semaphore;
			video::IQueue::SSubmitInfo::SCommandBufferInfo scratch = {};
			video::SIntendedSubmitInfo submit = {};
		};

		bool transfer(SGPUContext& ctx, video::IGPUImage* image, const bool upload, asset::IImage::LAYOUT& layout, const void* src, const asset::E_FORMAT srcFormat, void* dst, const asset::IImage::SBufferCopy& region)
		{
			using namespace asset;
			auto* rec = ctx.submit.getCommandBufferForRecording();
			video::IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t barrier = {};
			barrier.image = image;
			barrier.subresourceRange = {.aspectMask=IImage::EAF_COLOR_BIT,.baseMipLevel=0u,.levelCount=1u,.baseArrayLayer=0u,.layerCount=1u};
			barrier.oldLayout = layout;
			barrier.newLayout = upload ? IImage::LAYOUT::TRANSFER_DST_OPTIMAL:IImage::LAYOUT::TRANSFER_SRC_OPTIMAL;
			const bool fresh = layout==IImage::LAYOUT::UNDEFINED;
			barrier.barrier.dep = {
				.srcStageMask = fresh ? PIPELINE_STAGE_FLAGS::NONE:PIPELINE_STAGE_FLAGS::COPY_BIT,
				.srcAccessMask = fresh ? ACCESS_FLAGS::NONE:(layout==IImage::LAYOUT::TRANSFER_DST_OPTIMAL ? ACCESS_FLAGS::TRANSFER_WRITE_BIT:ACCESS_FLAGS::TRANSFER_READ_BIT),
				.dstStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT,
				.dstAccessMask = upload ? ACCESS_FLAGS::TRANSFER_WRITE_BIT:ACCESS_FLAGS::TRANSFER_READ_BIT
			};
			video::IGPUCommandBuffer::SPipelineBarrierDependencyInfo dep = {};
			dep.imgBarriers = {&barrier,1};
			bool ok = rec && rec->cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE,dep);
			if (ok)
				layout = barrier.newLayout;
			if (ok)
				ok = upload ? ctx.utils->updateImageViaStagingBuffer(ctx.submit,src,srcFormat,image,barrier.newLayout,{&region,1}):
					ctx.utils->downloadImageViaStagingBuffer(ctx.submit,image,barrier.newLayout,dst,{&region,1});
			rec = ctx.submit.getCommandBufferForRecording();
			const auto completed = ctx.submit.getFutureScratchSemaphore();
			if (!rec || ctx.submit.submit(rec,{})!=video::IQueue::RESULT::SUCCESS || m_device->blockForSemaphores({&completed,1})!=video::ISemaphore::WAIT_RESULT::SUCCESS)
				return false;
			if (upload)
				ctx.utils->getDefaultUpStreamingBuffer()->cull_frees();
			else
				ctx.utils->getDefaultDownStreamingBuffer()->cull_frees();
			return ctx.submit.beginNextCommandBuffer(rec) && ok;
		}

		bool iutilitiesCases()
		{
			using namespace asset;
			const auto* physDev = m_device->getPhysicalDevice();
			const uint32_t width = std::min<uint32_t>(BigWidth,physDev->getLimits().maxImageDimension2D);
			const uint32_t height = BigHeight;
			const E_FORMAT imageFormat = EF_R8G8B8A8_SRGB;
			const auto& usages = physDev->getImageFormatUsagesOptimalTiling()[imageFormat];
			if (!usages.transferSrc || !usages.transferDst)
			{
				m_logger->log("FILTERBENCH IUtilities skipped, no transfer support for RGBA8_SRGB",system::ILogger::ELL_WARNING);
				return true;
			}

			SGPUContext ctx;
			ctx.utils = video::IUtilities::create(core::smart_refctd_ptr<video::ILogicalDevice>(m_device),core::smart_refctd_ptr<system::ILogger>(m_logger));
			ctx.pool = m_device->createCommandPool(m_queue->getFamilyIndex(),video::IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
			if (!ctx.utils || !ctx.pool || !ctx.pool->createCommandBuffers(video::IGPUCommandPool::BUFFER_LEVEL::PRIMARY,1u,&ctx.cmdbuf))
				return false;
			ctx.semaphore = m_device->createSemaphore(0u);
			if (!ctx.semaphore || !ctx.cmdbuf->begin(video::IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT))
				return false;
			ctx.scratch = {.cmdbuf=ctx.cmdbuf.get()};
			ctx.submit.queue = m_queue;
			ctx.submit.waitSemaphores = {};
			ctx.submit.prevCommandBuffers = {};
			ctx.submit.scratchCommandBuffers = {&ctx.scratch,1};
			ctx.submit.scratchSemaphore = {.semaphore=ctx.semaphore.get(),.value=0u,.stageMask=PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS};
			ctx.submit.initialScratchValue = 0u;
			{
				const auto upMemory = ctx.utils->getDefaultUpStreamingBuffer()->getBuffer()->getBoundMemory().memory;
				const auto downMemory = ctx.utils->getDefaultDownStreamingBuffer()->getBuffer()->getBoundMemory().memory;
				m_logger->log("FILTERBENCH IUtilities %ux%u, upload staging memory flags 0x%x, download staging memory flags 0x%x",system::ILogger::ELL_PERFORMANCE,
					width,height,uint32_t(upMemory->getMemoryPropertyFlags().value),uint32_t(downMemory->getMemoryPropertyFlags().value));
			}

			// row sized writes into the IUtilities upload staging memory itself, what one upload chunk of this image does
			{
				uint8_t* const staging = reinterpret_cast<uint8_t*>(ctx.utils->getDefaultUpStreamingBuffer()->getBufferPointer());
				const size_t stagingSize = ctx.utils->getDefaultUpStreamingBuffer()->get_total_size();
				const size_t rowTexels = width, rows = core::min<size_t>(stagingSize/(rowTexels*4u),1024u);
				std::vector<uint8_t> srcRows(rows*rowTexels*4u), heapRows(rows*rowTexels*4u);
				fillPattern(srcRows.data(),srcRows.size(),6u);
				STexelReshuffle reshuffle;
				if (!STexelReshuffle::create(EF_R8G8B8_SRGB,EF_R8G8B8A8_SRGB,SPromotionSwizzle<3u>::componentMapping,reshuffle))
					return false;
				const core::vectorSIMDu32 extent(uint32_t(rowTexels),1u,1u,1u);
				const core::vectorSIMDu32 inStrides(uint32_t(rowTexels),1u,1u,1u);
				const core::vectorSIMDu32 outStrides(uint32_t(rowTexels),1u,1u,1u);
				ablation("ablation memcpy rows into IUtilities staging","stage",rows*rowTexels,rows*rowTexels*4u,[&]() -> void
				{
					for (size_t r=0u; r<rows; r++)
						memcpy(staging+r*rowTexels*4u,srcRows.data()+r*rowTexels*4u,rowTexels*4u);
				});
				ablation("ablation STexelReshuffle rows into IUtilities staging","stage",rows*rowTexels,rows*rowTexels*4u,[&]() -> void
				{
					for (size_t r=0u; r<rows; r++)
						reshuffle.execute(core::execution::seq,staging+r*rowTexels*4u,outStrides,srcRows.data()+r*rowTexels*3u,inStrides,extent);
				});
				ablation("ablation STexelReshuffle rows into heap","heap",rows*rowTexels,rows*rowTexels*4u,[&]() -> void
				{
					for (size_t r=0u; r<rows; r++)
						reshuffle.execute(core::execution::seq,heapRows.data()+r*rowTexels*4u,outStrides,srcRows.data()+r*rowTexels*3u,inStrides,extent);
				});
				// store type into the staging memory: one large memcpy, small memcpy calls, and explicit streaming stores
				const size_t bytes = size_t(32u)<<20;
				ablation("ablation memcpy 32 MiB at once into IUtilities staging","stage",bytes/4u,bytes,[&]() -> void {memcpy(staging,srcRows.data(),bytes);});
				ablation("ablation memcpy 32 MiB in 64 KiB calls into IUtilities staging","stage",bytes/4u,bytes,[&]() -> void
				{
					for (size_t offset=0u; offset<bytes; offset+=size_t(64u)<<10)
						memcpy(staging+offset,srcRows.data()+offset,size_t(64u)<<10);
				});
				ablation("ablation streaming stores 32 MiB into IUtilities staging","stage",bytes/4u,bytes,[&]() -> void
				{
					for (size_t offset=0u; offset<bytes; offset+=16u)
						_mm_stream_si128(reinterpret_cast<__m128i*>(staging+offset),_mm_loadu_si128(reinterpret_cast<const __m128i*>(srcRows.data()+offset)));
					_mm_sfence();
				});
			}

			video::IGPUImage::SCreationParams params = {};
			params.type = IImage::E_TYPE::ET_2D;
			params.extent = {width,height,1u};
			params.format = imageFormat;
			params.mipLevels = 1u;
			params.arrayLayers = 1u;
			params.samples = IImage::E_SAMPLE_COUNT_FLAGS::ESCF_1_BIT;
			params.usage = IImage::EUF_TRANSFER_SRC_BIT|IImage::EUF_TRANSFER_DST_BIT;
			auto image = m_device->createImage(std::move(params));
			if (!image)
				return false;
			auto reqs = image->getMemoryReqs();
			reqs.memoryTypeBits &= physDev->getDeviceLocalMemoryTypeBits();
			if (!m_device->allocate(reqs,{image.get(),video::IDeviceMemoryAllocation::EMAF_NONE}).isValid())
				return false;

			IImage::SBufferCopy region = {};
			region.imageSubresource.aspectMask = IImage::EAF_COLOR_BIT;
			region.imageSubresource.layerCount = 1u;
			region.imageExtent = {width,height,1u};

			const size_t texels = size_t(width)*height;
			std::vector<uint8_t> rgb(texels*3u+4u), rgba(texels*4u), readback(texels*4u);
			fillPattern(rgb.data(),rgb.size(),5u);
			for (size_t t=0u; t<texels; t++)
			{
				memcpy(rgba.data()+t*4u,rgb.data()+t*3u,3u);
				rgba[t*4u+3u] = 255u;
			}

			bool ok = true;
			IImage::LAYOUT layout = IImage::LAYOUT::UNDEFINED;
			auto timedUpload = [&](const void* src, const E_FORMAT srcFormat) -> double
			{
				const auto start = std::chrono::steady_clock::now();
				ok = transfer(ctx,image.get(),true,layout,src,srcFormat,nullptr,region) && ok;
				return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
			};
			auto reportUploads = [&](const char* name, std::vector<double>& ms, const size_t srcBytes) -> void
			{
				std::sort(ms.begin(),ms.end());
				report(name,"stage","par",{ms.front(),ms[ms.size()/2u]},srcBytes,texels,0ull);
			};
			// the GPU drops its clocks and PCIe link speed when idle, which caps CPU writes into staging, so both uploads are warmed up and interleaved
			{
				std::vector<double> promotionMs, sameFormatMs;
				for (const auto warmupStart=std::chrono::steady_clock::now(); std::chrono::steady_clock::now()-warmupStart<std::chrono::seconds(2);)
					timedUpload(rgba.data(),EF_R8G8B8A8_SRGB);
				promotionMs.push_back(timedUpload(rgb.data(),EF_R8G8B8_SRGB));
				// one slow pass is enough to show the problem
				for (uint32_t rep=0u; rep<3u && promotionMs.front()<30000.0; rep++)
				{
					sameFormatMs.push_back(timedUpload(rgba.data(),EF_R8G8B8A8_SRGB));
					promotionMs.push_back(timedUpload(rgb.data(),EF_R8G8B8_SRGB));
				}
				if (promotionMs.size()>1u)
					promotionMs.erase(promotionMs.begin());
				reportUploads("IUtilities upload rgb8_srgb->rgba8_srgb",promotionMs,texels*3u);
				if (!sameFormatMs.empty())
					reportUploads("IUtilities upload rgba8_srgb same format",sameFormatMs,rgba.size());
			}
			auto downloadAndCheck = [&](const char* name) -> void
			{
				std::fill(readback.begin(),readback.end(),Sentinel);
				ok = transfer(ctx,image.get(),false,layout,nullptr,EF_UNKNOWN,readback.data(),region) && ok;
				size_t mismatches = 0u;
				for (size_t t=0u; t<texels; t++)
					mismatches += memcmp(readback.data()+t*4u,rgba.data()+t*4u,4u)!=0;
				char detail[96];
				snprintf(detail,sizeof(detail),"(%zu of %zu texels differ from the exact promotion, hash %016llx)",mismatches,texels,(unsigned long long)hashBytes(readback.data(),readback.size()));
				verdict(name,mismatches==0u,detail);
			};

			downloadAndCheck("IUtilities rgb8_srgb->rgba8_srgb round trip");
			timedUpload(rgba.data(),EF_R8G8B8A8_SRGB);
			downloadAndCheck("IUtilities rgba8_srgb same format round trip");

			ctx.cmdbuf->end();
			return ok;
		}

		video::ILogicalDevice* m_device;
		video::IQueue* m_queue;
		system::ILogger* m_logger;
		SMapped m_sysram, m_bar;
};

}
#endif
