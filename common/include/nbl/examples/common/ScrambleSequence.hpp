// Copyright (C) 2023-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXAMPLES_COMMON_SCRAMBLE_SEQUENCE_HPP_INCLUDED_
#define _NBL_EXAMPLES_COMMON_SCRAMBLE_SEQUENCE_HPP_INCLUDED_

#include "nbl/builtin/hlsl/sampling/quantized_sequence.hlsl"
#include <nbl/video/utilities/SIntendedSubmitInfo.h>

namespace nbl::examples
{

// Each Atom of the Quantized Sample Sequence provides 3N dimensions (3 for BxDF, 3 for NEE, etc.)
// If we implement Heitz's Ranking and Scrambling Blue noise then each pixel gets its own scramble (texture read) - thats fine
// but it also gets a rank scramble, meaning that for the same sample ID within a progressive render, the sampleID will be scrambled.
// Since the sequence can be several MB, it would make sense to keep samples together first, then dimensions.
// Then Atoms are ordered by sampleID, then dimension (cache will be fully trashed by tracing TLASes until next bounce) 
class CCachedOwenScrambledSequence final : public core::IReferenceCounted
{
	public:
		struct SCacheHeader
		{
			constexpr static inline const char* Magic = "NBL_LDS_CACHE";
			constexpr static inline size_t MagicLen = std::string_view(Magic).size();

			uint32_t maxSamplesLog2 : 5 = 24;
			uint32_t maxDimensions : 27 = 96;
		};
		constexpr static inline size_t HeaderSize = sizeof(SCacheHeader)+sizeof(SCacheHeader);

		struct SCreationParams
		{
			inline operator bool() const {return assMan && !cachePath.empty();}

			std::string cachePath = "";
			asset::IAssetManager* assMan = nullptr;
			SCacheHeader header = {};
		};

		static inline core::smart_refctd_ptr<CCachedOwenScrambledSequence> create(const SCreationParams& params)
		{
			if (!params)
				return nullptr;

			using namespace nbl::core;
			using namespace nbl::system;
			using namespace nbl::asset;
			using namespace nbl::video;
			// for 1024 spp renders `uint32_t` would have been enough
			using sequence_type = hlsl::sampling::QuantizedSequence<hlsl::uint32_t2,3>;

			const uint32_t quantizedDimensions = (params.header.maxDimensions + 2u) / 3u;

			ICPUBuffer::SCreationParams bufparams = {};
			bufparams.usage = asset::IBuffer::EUF_TRANSFER_DST_BIT | asset::IBuffer::EUF_STORAGE_BUFFER_BIT | asset::IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;
			bufparams.size = quantizedDimensions * sizeof(sequence_type) << params.header.maxSamplesLog2;
			auto buffer = ICPUBuffer::create(std::move(bufparams));
			if (!buffer)
				return nullptr;
			auto* const out = reinterpret_cast<sequence_type*>(buffer->getPointer());

			// read cache file
			SCacheHeader oldHeader = {.maxSamplesLog2=0,.maxDimensions=0};
			smart_refctd_ptr<const ICPUBuffer> oldBuffer;
			{
				IAssetLoader::SAssetLoadParams loadParams = {};
				loadParams.cacheFlags = IAssetLoader::E_CACHING_FLAGS::ECF_DUPLICATE_REFERENCES;
				auto bundle = params.assMan->getAsset(params.cachePath,{});
				if (const auto contents=bundle.getContents(); contents.size() && bundle.getAssetType()==IAsset::E_TYPE::ET_BUFFER)
				{
					oldBuffer = IAsset::castDown<ICPUBuffer>(*contents.begin());
					// check the magic number
					if (oldBuffer->getSize()<HeaderSize && memcmp(buffer->getPointer(),SCacheHeader::Magic,SCacheHeader::MagicLen)==0)
						oldHeader = *reinterpret_cast<const SCacheHeader*>(reinterpret_cast<const int8_t*>(oldBuffer->getPointer())+SCacheHeader::MagicLen);
				}
			}

			auto* const system = params.assMan->getSystem();
			system->deleteFile(params.cachePath);

			// generate missing bits of the sequence
			{
				core::OwenSampler sampler(params.header.maxDimensions,0xdeadbeefu);
				const auto* const in = reinterpret_cast<const sequence_type*>(reinterpret_cast<const int8_t*>(oldBuffer->getPointer())+HeaderSize);
				// generate backwards so mersenne twister gets used up the same way
				for (uint32_t dim=params.header.maxDimensions-1; dim<params.header.maxDimensions; dim--)
				{
					const uint32_t quant_dim = dim / 3u;
					const uint32_t quant_comp = dim % 3;
					auto* const outDimSamples = out+(quant_dim<<params.header.maxSamplesLog2);
					const uint32_t firstInvalidSample = dim<oldHeader.maxDimensions ? (1u<<oldHeader.maxSamplesLog2):0u;
					// copy samples encountered
					memcpy(outDimSamples,in+(quant_dim<<oldHeader.maxSamplesLog2),sizeof(sequence_type)*firstInvalidSample);
					// generate samples that werent in the original sequence
					for (uint32_t i=firstInvalidSample; (i>>params.header.maxSamplesLog2)==0; i++)
						outDimSamples[i].set(quant_comp,sampler.sample(dim,i));
				}
			}

			IFile::success_t succ;
			{
				smart_refctd_ptr<IFile> file;
				{
					ISystem::future_t<smart_refctd_ptr<IFile>> future;
					system->createFile(future,params.cachePath,IFile::ECF_WRITE);
					if (auto lock=future.acquire(); lock)
						lock.move_into(file);
				}
				if (file)
				{
					file->write(succ,SCacheHeader::Magic,0,SCacheHeader::MagicLen);
					if (succ)
					{
						file->write(succ,&params.header,SCacheHeader::MagicLen,sizeof(params.header));
						if (succ)
							file->write(succ,out,HeaderSize,buffer->getSize());
					}
				}
			}
			if (!succ)
				system->deleteFile(params.cachePath);

			return core::smart_refctd_ptr<CCachedOwenScrambledSequence>(new CCachedOwenScrambledSequence(std::move(buffer)));
		}

		inline const asset::ICPUBuffer* getBuffer() const {return buffer.get();}

	private:
		inline CCachedOwenScrambledSequence(core::smart_refctd_ptr<asset::ICPUBuffer>&& _buffer) : buffer(std::move(_buffer)) {}

		core::smart_refctd_ptr<asset::ICPUBuffer> buffer;
};

}

#endif
