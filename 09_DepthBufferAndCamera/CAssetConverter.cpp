// Copyright (C) 2024-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
#include "CAssetConverter.h"

using namespace nbl::core;
using namespace nbl::asset;

namespace nbl::video
{
//
template<asset::Asset AssetType>
struct dep_cache_hasher
{
	inline size_t operator()(const CAssetConverter::input_t<AssetType>& input) const
	{
		return input.hash();
	}
};
template<asset::Asset AssetType>
using dep_cache_t = std::unordered_multimap<CAssetConverter::input_t<AssetType>,CAssetConverter::patch_t<AssetType>,dep_cache_hasher<AssetType>>;

//
void CAssetConverter::CCache<asset::ICPUShader>::lookup_t::hash_impl(blake3_hasher* hasher) const
{
	blake3_hasher_update(hasher,patch.stage);
	const auto* asset = input.asset;
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
	const auto* asset = input.asset;
	for (auto i=0; i<asset::ICPUPipelineLayout::DESCRIPTOR_SET_COUNT; i++)
	{
		// TODO: need the hashes of patched descriptor set layouts!!! FIXME
//		blake3_hasher_update(hasher,asset->getDescriptorSetLayout(i));
	}
}

//
auto CAssetConverter::reserve(const SInputs& inputs) -> SResults
{
	SResults retval = {};
	if (inputs.readCache && inputs.readCache->m_params.device!=m_params.device)
		return retval;
#if 0
	// gather all dependencies (DFS graph search) and patch, this happens top-down
	// do not deduplicate/merge assets at this stage, only patch GPU creation parameters
	{
		std::stack<const IAsset*> dfsStack;
		// This cache stops us adding an asset more than once
		core::tuple_transform_t<dep_cache_t,supported_asset_types> depCache = {};
		// returns true if new element was inserted
		auto cache = [&]<Asset AssetType>(const CCache::key_t<AssetType>& in)->bool
		{
			// skip invalid inputs silently
			if (!in.valid())
				return false;
			
			using cache_t = dep_cache_t<AssetType>;
			auto& cache = std::get<cache_t>(depCache);
			auto found = cache.equal_range(in.asset);
			for (auto it=found.first; it!=found.second; it++)
			{
				auto& cachedPatch = it->second;
				// found a thing, try-combine the patches
				auto combined = cachedPatch.combine(in.patch);
				// check whether the item is creatable after patching
				if (combined.valid())
				{
					cachedPatch = std::move(combined);
					return false;
				}
				// else duplicate/de-alias
			}
			// insert a new entry
			cache.insert(found.first,{in.asset,in.patch});
			if (AssetType::HasDependents)
				dfsStack.push(in.asset.asset);
			return true;
		};
		// initialize stacks
		core::for_each_in_tuple(input.assets,[&]<Asset AssetType>(const SInputs::span_t<AssetType> inputs)->void{
			// dedup the inputs so they API is more forgiving to use
			for (auto& in : inputs)
				cache(in);
		});
		// everything that's not explicit has `uniqueCopyForUser==nullptr` and default patch params
		while (!dfsStack.empty())
		{
			const auto* asset = dfsStack.top();
			dfsStack.pop();
			// everything we popped, has already been cached, now time to go over dependents
			switch (asset->getAssetType())
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
				case ICPUDescriptorSetLayout::AssetType:
				{
					auto layout = static_cast<const ICPUDescriptorSetLayout*>(asset);
					for (const auto& sampler : layout->getImmutableSamplers())
						cache({/*TODO*/});
					break;
				}
#endif
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
					_NBL_TODO();
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
//#if 0
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