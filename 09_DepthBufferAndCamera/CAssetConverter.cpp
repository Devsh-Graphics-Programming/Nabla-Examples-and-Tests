// Copyright (C) 2024-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
#include "CAssetConverter.h"

#include <type_traits>


using namespace nbl::core;
using namespace nbl::asset;


namespace nbl::video
{
CAssetConverter::patch_t<ICPUShader>::patch_t(
	const ICPUShader* shader,
	const SPhysicalDeviceFeatures& features,
	const SPhysicalDeviceLimits& limits
)
{
	const auto _stage = shader->getStage();
	switch (_stage)
	{
		// supported always
		case IGPUShader::E_SHADER_STAGE::ESS_VERTEX:
		case IGPUShader::E_SHADER_STAGE::ESS_FRAGMENT:
		case IGPUShader::E_SHADER_STAGE::ESS_COMPUTE:
			stage = _stage;
			break;
		case IGPUShader::E_SHADER_STAGE::ESS_TESSELLATION_CONTROL:
		case IGPUShader::E_SHADER_STAGE::ESS_TESSELLATION_EVALUATION:
			if (features.tessellationShader)
				stage = _stage;
			break;
		case IGPUShader::E_SHADER_STAGE::ESS_GEOMETRY:
			if (features.geometryShader)
				stage = _stage;
			break;
		case IGPUShader::E_SHADER_STAGE::ESS_TASK:
//			if (features.taskShader)
//				stage = _stage;
			break;
		case IGPUShader::E_SHADER_STAGE::ESS_MESH:
//			if (features.meshShader)
//				stage = _stage;
			break;
		case IGPUShader::E_SHADER_STAGE::ESS_RAYGEN:
		case IGPUShader::E_SHADER_STAGE::ESS_ANY_HIT:
		case IGPUShader::E_SHADER_STAGE::ESS_CLOSEST_HIT:
		case IGPUShader::E_SHADER_STAGE::ESS_MISS:
		case IGPUShader::E_SHADER_STAGE::ESS_INTERSECTION:
		case IGPUShader::E_SHADER_STAGE::ESS_CALLABLE:
			if (features.rayTracingPipeline)
				stage = _stage;
			break;
		default:
			break;
	}
}

CAssetConverter::patch_t<ICPUPipelineLayout>::patch_t(
	const ICPUPipelineLayout* pplnLayout,
	const SPhysicalDeviceFeatures& features,
	const SPhysicalDeviceLimits& limits
) : patch_t()
{
	// TODO: some way to do bound checking and indicate validity
	const auto pc = pplnLayout->getPushConstantRanges();
	for (auto it=pc.begin(); it!=pc.end(); it++)
	{
		if (it->offset>=limits.maxPushConstantsSize)
			return;
		const auto end = it->offset+it->size;
		if (end<it->offset || end>limits.maxPushConstantsSize)
			return;
		for (auto byte=it->offset; byte<end; byte++)
			pushConstantBytes[byte] = it->stageFlags;
	}
	invalid = false;
}

CAssetConverter::patch_t<ICPUBuffer>::patch_t(
	const ICPUBuffer* buffer,
	const SPhysicalDeviceFeatures& features,
	const SPhysicalDeviceLimits& limits
)
{
	const auto _usage = buffer->getUsageFlags();
	if (_usage.hasFlags(usage_flags_t::EUF_CONDITIONAL_RENDERING_BIT_EXT) && !features.conditionalRendering)
		return;
	if ((_usage.hasFlags(usage_flags_t::EUF_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT)||_usage.hasFlags(usage_flags_t::EUF_ACCELERATION_STRUCTURE_STORAGE_BIT)) && !features.accelerationStructure)
		return;
	if (_usage.hasFlags(usage_flags_t::EUF_SHADER_BINDING_TABLE_BIT) && !features.rayTracingPipeline)
		return;
	usage = _usage;
	// good default
	usage |= usage_flags_t::EUF_INLINE_UPDATE_VIA_CMDBUF;
}

CAssetConverter::patch_t<ICPUSampler>::patch_t(
	const ICPUSampler* sampler,
	const SPhysicalDeviceFeatures& features,
	const SPhysicalDeviceLimits& limits
) : anisotropyLevelLog2(sampler->getParams().AnisotropicFilter)
{
	if (anisotropyLevelLog2>limits.maxSamplerAnisotropyLog2)
		anisotropyLevelLog2 = limits.maxSamplerAnisotropyLog2;
}


//
void CAssetConverter::CCache<asset::ICPUShader>::lookup_t::hash_impl(blake3_hasher* hasher) const
{
	blake3_hasher_update(hasher,asset->getContentType());
	const auto* content = asset->getContent();
	blake3_hasher_update(hasher,content->getPointer(),content->getSize());
	// TODO: filepath hint?
}
void CAssetConverter::CCache<asset::ICPUDescriptorSetLayout>::lookup_t::hash_impl(blake3_hasher* hasher) const
{
	// TODO: hash the bindings!
}
void CAssetConverter::CCache<asset::ICPUPipelineLayout>::lookup_t::hash_impl(blake3_hasher* hasher) const
{
	blake3_hasher_update(hasher,patch.pushConstantBytes);
	for (auto i=0; i<asset::ICPUPipelineLayout::DESCRIPTOR_SET_COUNT; i++)
	{
		// TODO: need the hashes of patched descriptor set layouts!!! FIXME
//		blake3_hasher_update(hasher,asset->getDescriptorSetLayout(i));
	}
}

//
template<Asset AssetType>
using patch_cache_t = core::vector<CAssetConverter::patch_t<AssetType>>;

//
auto CAssetConverter::reserve(const SInputs& inputs) -> SResults
{
	SResults retval = {};
	if (inputs.readCache && inputs.readCache->m_params.device!=m_params.device)
		return retval;

	// gather all dependencies (DFS graph search) and patch, this happens top-down
	// do not deduplicate/merge assets at this stage, only patch GPU creation parameters
	{
		std::stack<SInputs::instance_t> dfsStack;
		// This cache stops us adding an asset more than once.
		// Note that its not a simple multimap because order of duplicate patches needs to be deterministic for an input.
		std::unordered_map<SInputs::instance_t,core::variant_transform_t<patch_cache_t,supported_asset_types>> dfsCache = {};
		// returns true if new element was inserted
		auto cache = [&]<Asset AssetType>(const SInputs::instance_t& user, const AssetType* asset, patch_t<AssetType>&& patch)->bool
		{
			assert(asset);
			// skip invalid inputs silently
			if (!patch.valid())
				return false;

			// get unique group and see if we visited already
			SInputs::instance_t record = {
				.asset = asset,
				.uniqueCopyGroupID = inputs.getDependantUniqueCopyGroupID(user,asset)
			};
#if 0
			auto found = dfsCache.find(record);
			if (found!=dfsCache.end())
			{
				auto& uniquePatches = std::get<patch_cache_t<AssetType>>(found->second);
				// iterate over all intended EXTRA copies of an asset
				for (auto& candidate : uniquePatches)
				{
					// found a thing, try-combine the patches
					auto combined = candidate.combine(patch);
					// check whether the item is creatable after patching
					if (combined.valid())
					{
						candidate = std::move(combined);
						return false;
					}
					// else try the next one
				}
				// haven't managed to combine with anything, ergo new entry
				uniquePatches.push_back(std::move(patch));
			}
			else // not found, insert new entry
				dfsCache.emplace(record,patch_cache_t<AssetType>{std::move(patch)});
#endif

			if (AssetType::HasDependents)
				dfsStack.push(record);
			return true;
		};
		//
		auto* const device = m_params.device;
		const auto& features = device->getEnabledFeatures();
		const auto& limits = device->getPhysicalDevice()->getLimits();
		// initialize stacks
		auto initialize = [&]<typename AssetType>(const std::span<const AssetType* const> assets)->void
		{
			const auto& patches = std::get<SInputs::patch_span_t<AssetType>>(inputs.patches);
			// size and fill the result array with nullptr
			std::get<SResults::vector_t<AssetType>>(retval.m_gpuObjects).resize(assets.size());
			for (size_t i=0; i<assets.size(); i++)
			if (auto asset=assets[i]; asset)
			{
				// skip invalid inputs silently
				if (!asset)
					continue;
				patch_t<AssetType> patch = i<patches.size() ? patches[i]:patch_t<AssetType>(asset,features,limits);
				cache({},asset,std::move(patch));
			}
		};
		core::for_each_in_tuple(inputs.assets,initialize);
		// everything that's not explicit has `uniqueCopyForUser==nullptr` and default patch params
		while (!dfsStack.empty())
		{
			const auto entry = dfsStack.top();
			dfsStack.pop();
			// everything we popped, has already been cached, now time to go over dependents
			switch (entry.asset->getAssetType())
			{
#if 0
				case ICPUPipelineLayout::AssetType:
				{
					auto pplnLayout = static_cast<const ICPUPipelineLayout*>(asset);
					for (auto i=0; i<ICPUPipelineLayout::DESCRIPTOR_SET_COUNT; i++)
					if (auto layout=pplnLayout->getDescriptorSetLayout(i); layout)
					if (cache({/*TODO*/}))
						dfsStack.push(layout);
					break;
				}
#endif
				case ICPUDescriptorSetLayout::AssetType:
				{
					auto layout = static_cast<const ICPUDescriptorSetLayout*>(entry.asset);
					for (const auto& sampler : layout->getImmutableSamplers())
					if (sampler)
						cache.operator()<ICPUSampler>(entry,sampler.get(),{sampler.get(),features,limits});
					break;
				}
				case ICPUDescriptorSet::AssetType:
				{
					_NBL_TODO();
					break;
				}
				case ICPUImageView::AssetType:
				{
					_NBL_TODO();
					break;
				}
				case ICPUBufferView::AssetType:
				{
					auto view = static_cast<const ICPUBufferView*>(entry.asset);
					const auto buffer = view->getUnderlyingBuffer();
					if (buffer)
					{
						patch_t<ICPUBuffer> patch = {buffer,features,limits};
						// we have no clue how this will be used, so we mark both usages
						patch.usage = IGPUBuffer::EUF_STORAGE_TEXEL_BUFFER_BIT|IGPUBuffer::EUF_UNIFORM_TEXEL_BUFFER_BIT;
						cache.operator()<ICPUBuffer>(entry,buffer,std::move(patch));
					}
					break;
				}
				// these assets have no dependants, should have never been pushed on the stack
				default:
					assert(false);
					break;
			}
		}
	}
	// now we have a set of implicit gpu creation parameters we want to create resources with,
	// and a mapping from (Asset,Patch) -> UniqueAsset
	// If there's a readCache we need to look for an item there first.
#if 0
	auto dedup = [&]<Asset AssetType>()->void
	{
		using cache_t = SResults::dag_cache_t<AssetType>;
		core::unordered_set<typename cache_t::value_type*/*TODO: Hash,Equals*/> conser;
		for (auto& asset : std::get<cache_t>(retval.m_typedDagNodes))
		{
			assert(!asset.second.canonical);
			//assert(!asset.second.result);
			auto [it, inserted] = conser.insert(&asset);
			if (inserted)
			{
				if (input.readCache)
					continue; // TODO: below
//				if (auto found=input.readCache->find(); found!=input.readCache->end())
// 				{
// 				   TODO: insert to our own cache
//					asset.second.result = &((*it)->second.result = found->result);
//				}
			}
			else
				asset.second.canonical = &(*it)->second;
		}
	};
	// Lets see if we can collapse any of the (Asset Content) into the same thing,
	// to correctly de-dup we need to go bottom-up!!!
	dedup.operator()<ICPUShader>();
#endif
// Shader, DSLayout, PipelineLayout, Compute Pipeline
// Renderpass, Graphics Pipeline
// Buffer, BufferView, Sampler, Image, Image View, Bottom Level AS, Top Level AS, Descriptor Set, Framebuffer  
// Buffer -> SRange, patched usage, owner(s)
// BufferView -> SRange, promoted format
// Sampler -> Clamped Params (only aniso, really)
// Image -> this, patched usage, promoted format
// Image View -> ref to patched Image, patched usage, promoted format
// Descriptor Set -> unique layout, 

	return retval;
}

// read[key,patch,blake3] -> look up GPU Object based on hash
// We can't look up "near misses" (supersets) because they'd have different hashes
// can't afford to split hairs like finding overlapping buffer ranges, etc.
// Stuff like that would require a completely different hashing/lookup strategy (or multiple fake entries).

// reservation<type>[key,patch,blake3] -> reserve GPU Object (how?) based on hash

// afterwards
// iterate over reservation<type>, create the GPU objects and insert into write cache

// now I need to find the GPU objects for my inputs
// but I don't know how the inputs have been patched, also don't want to re-hash, so lets just store the values per input element?

//
bool CAssetConverter::convert(SResults& reservations, SConvertParams& params)
{
	if (!reservations.reserveSuccess())
		return false;

	const auto reqQueueFlags = reservations.getRequiredQueueFlags();
	if (reqQueueFlags.hasFlags(IQueue::FAMILY_FLAGS::TRANSFER_BIT | IQueue::FAMILY_FLAGS::COMPUTE_BIT) && !params.utilities)
		return false;

	auto device = m_params.device;
	if (!device)
		return false;

	auto invalidQueue = [reqQueueFlags,device,&params](const IQueue::FAMILY_FLAGS flag, IQueue* queue)->bool
	{
		if (!reqQueueFlags.hasFlags(flag))
			return false;
		if (!params.utilities || params.utilities->getLogicalDevice()!=device)
			return true;
		if (!queue || queue->getOriginDevice()!=device)
			return true;
		const auto& qFamProps = device->getPhysicalDevice()->getQueueFamilyProperties();
		if (!qFamProps[queue->getFamilyIndex()].queueFlags.hasFlags(flag))
			return true;
		return false;
	};
	// If the transfer queue will be used, the transfer Intended Submit Info must be valid and utilities must be provided
	if (invalidQueue(IQueue::FAMILY_FLAGS::TRANSFER_BIT,params.transfer.queue))
		return false;
	// If the compute queue will be used, the compute Intended Submit Info must be valid and utilities must be provided
	if (invalidQueue(IQueue::FAMILY_FLAGS::COMPUTE_BIT,params.transfer.queue))
		return false;

	const core::string debugPrefix = "Created by Converter "+std::to_string(ptrdiff_t(this))+" with hash ";
	auto setDebugName = [&debugPrefix](IBackendObject* obj, const size_t hashval)->void
	{
		if (obj)
			obj->setObjectDebugName((debugPrefix+std::to_string(hashval)).c_str());
	};

	// create shaders
	{
		ILogicalDevice::SShaderCreationParameters params = {
			.optimizer = m_params.optimizer.get(),
			.readCache = m_params.compilerCache.get(),
			.writeCache = m_params.compilerCache.get()
		};
#if 0
		for (auto& shader : std::get<SResults::dag_cache_t<ICPUShader>>(reservations.m_typedDagNodes))
		if (!shader.second.canonical)
		{
			assert(!shader.second.result);
			// custom code start
			params.cpushader = shader.first.asset;
			shader.second.result = device->createShader(params);
		}
#endif
	}

	return true;
}

ISemaphore::future_t<bool> CAssetConverter::SConvertParams::autoSubmit()
{
	// TODO: transfer first, then compute
	return {};
}

}