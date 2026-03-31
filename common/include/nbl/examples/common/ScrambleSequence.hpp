// Copyright (C) 2023-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXAMPLES_COMMON_SCRAMBLE_SEQUENCE_HPP_INCLUDED_
#define _NBL_EXAMPLES_COMMON_SCRAMBLE_SEQUENCE_HPP_INCLUDED_

#include "nbl/builtin/hlsl/sampling/quantized_sequence.hlsl"
#include <nbl/video/utilities/SIntendedSubmitInfo.h>

namespace nbl::examples
{

class ScrambleSequence : public core::IReferenceCounted
{
public:
    struct SCreationParams
    {
		video::CThreadSafeQueueAdapter* queue = nullptr;
		core::smart_refctd_ptr<video::IUtilities> utilities = nullptr;
		core::smart_refctd_ptr<system::ISystem> system = nullptr;
		system::path localOutputCWD;
		system::path sharedOutputCWD;
        std::string owenSamplerCachePath = "";

		uint32_t MaxBufferDimensions;
		uint32_t MaxSamplesBuffer;
    };

    static core::smart_refctd_ptr<ScrambleSequence> create(const SCreationParams& params)
    {
		auto createBufferFromCacheFile = [&](
			system::path filename,
			size_t bufferSize,
			void* data,
			core::smart_refctd_ptr<asset::ICPUBuffer>& buffer
			) -> std::pair<core::smart_refctd_ptr<system::IFile>, bool>
			{
				system::ISystem::future_t<core::smart_refctd_ptr<nbl::system::IFile>> owenSamplerFileFuture;
				system::ISystem::future_t<size_t> owenSamplerFileReadFuture;
				size_t owenSamplerFileBytesRead;

				params.system->createFile(owenSamplerFileFuture, params.localOutputCWD / filename, system::IFile::ECF_READ);
				core::smart_refctd_ptr<system::IFile> owenSamplerFile;

				if (owenSamplerFileFuture.wait())
				{
					owenSamplerFileFuture.acquire().move_into(owenSamplerFile);
					if (!owenSamplerFile)
						return { nullptr, false };

					owenSamplerFile->read(owenSamplerFileReadFuture, data, 0, bufferSize);
					if (owenSamplerFileReadFuture.wait())
					{
						owenSamplerFileReadFuture.acquire().move_into(owenSamplerFileBytesRead);

						if (owenSamplerFileBytesRead < bufferSize)
						{
							buffer = asset::ICPUBuffer::create({ sizeof(uint32_t) * bufferSize });
							return { owenSamplerFile, false };
						}

						buffer = asset::ICPUBuffer::create({ { sizeof(uint32_t) * bufferSize }, data });
					}
				}

				return { owenSamplerFile, true };
			};
		auto writeBufferIntoCacheFile = [&](core::smart_refctd_ptr<system::IFile> file, size_t bufferSize, void* data)
			{
				system::ISystem::future_t<size_t> owenSamplerFileWriteFuture;
				size_t owenSamplerFileBytesWritten;

				file->write(owenSamplerFileWriteFuture, data, 0, bufferSize);
				if (owenSamplerFileWriteFuture.wait())
					owenSamplerFileWriteFuture.acquire().move_into(owenSamplerFileBytesWritten);
			};

		const uint32_t quantizedDimensions = params.MaxBufferDimensions / 3u;
		const size_t bufferSize = quantizedDimensions * params.MaxSamplesBuffer;
		using sequence_type = hlsl::sampling::QuantizedSequence<hlsl::uint32_t2, 3>;
		std::vector<sequence_type> data(bufferSize);
		core::smart_refctd_ptr<asset::ICPUBuffer> sampleSeq;

		auto cacheBufferResult = createBufferFromCacheFile(params.sharedOutputCWD / params.owenSamplerCachePath, bufferSize, data.data(), sampleSeq);
		if (!cacheBufferResult.second)
		{
			core::OwenSampler sampler(params.MaxBufferDimensions, 0xdeadbeefu);

			asset::ICPUBuffer::SCreationParams bufparams = {};
			bufparams.size = quantizedDimensions * params.MaxSamplesBuffer * sizeof(sequence_type);
			sampleSeq = asset::ICPUBuffer::create(std::move(bufparams));

			auto out = reinterpret_cast<sequence_type*>(sampleSeq->getPointer());
			for (auto dim = 0u; dim < params.MaxBufferDimensions; dim++)
			{
				const auto dimSampler = sampler.prepareDimension(dim);
				for (uint32_t i = 0; i < params.MaxSamplesBuffer; i++)
				{
					const uint32_t quant_dim = dim / 3u;
					const uint32_t offset = dim % 3u;
					auto& seq = out[i * quantizedDimensions + quant_dim];
					const uint32_t sample = dimSampler.sample(i);
					seq.set(offset, sample);
				}
			}
			if (cacheBufferResult.first)
				writeBufferIntoCacheFile(cacheBufferResult.first, bufferSize, out);
		}

		video::IGPUBuffer::SCreationParams bufparams = {};
		bufparams.usage = asset::IBuffer::EUF_TRANSFER_DST_BIT | asset::IBuffer::EUF_STORAGE_BUFFER_BIT | asset::IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;
		bufparams.size = bufferSize;

		core::smart_refctd_ptr<video::IGPUBuffer> buffer;
		params.utilities->createFilledDeviceLocalBufferOnDedMem(
			video::SIntendedSubmitInfo{ .queue = params.queue },
			std::move(bufparams),
			sampleSeq->getPointer()
		).move_into(buffer);

		buffer->setObjectDebugName("Sequence buffer");

		return core::smart_refctd_ptr<ScrambleSequence>(new ScrambleSequence(std::move(buffer)));
    }

    ScrambleSequence(core::smart_refctd_ptr<video::IGPUBuffer>&& buffer) : buffer(std::move(buffer)) {}

    core::smart_refctd_ptr<video::IGPUBuffer> buffer;
};

}

#endif
