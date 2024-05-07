// Copyright (C) 2024-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
#include "CAssetConverter.h"


namespace nbl::video
{


struct dep_gather_hash
{
	template<asset::Asset AssetType>
	inline size_t operator()(const CAssetConverter::root_t<AssetType>& in) const
	{
		return std::hash<const void*>{}(in.asset)^(in.unique ? (~0x0ull):0x0ull);
	}
};
template<asset::Asset AssetType>
using dep_gather_cache_t = core::unordered_multimap<CAssetConverter::root_t<AssetType>,typename asset_traits<AssetType>::patch_t,dep_gather_hash>;

auto CAssetConverter::reserve(const SInput& input) -> SResults
{
	SResults retval = {};
	if (input.readCache->m_params.device!=m_params.device)
		return retval;

	core::tuple_transform_t<dep_gather_cache_t,supported_asset_types> dep_gather_caches;
	// gather all dependencies (DFS graph search) and patch
	// do not deduplicate/merge assets at this stage, only patch GPU creation parameters
	{
		core::stack<const asset::IAsset*> dfsStack;
		auto push = [&]<asset::Asset AssetType>(const CAssetConverter::input_t<AssetType>& in)->void
		{
			using cache_t = dep_gather_cache_t<AssetType>;
			auto& cache = std::get<cache_t>(dep_gather_caches);
			auto found = cache.equal_range(in);
			if (found.first!=found.second)
			{
#if 0
				// found the thing, combine patches
				const auto& cachedPatch = found->patch;
				if (auto combined=in.patch; combined)
					const_cast<asset_traits<AssetType>::patch_t&>(cachedPatch) = combined;
#endif
			}
			// insert a new entry
			cache.insert(found.first,{in,{}/*in.patch*/});
		};
		core::visit([&push]<asset::Asset AssetType>(const SInput::span_t<AssetType> assets)->void{
			for (auto& asset : assets)
				push(asset);
		},input.assets);
/*
		auto pushAll = [&push]()->void
		{
		};
		while (!dfsStack.empty())
		{
			const auto* asset = dfsStack.top();
			dfsStack.pop();

			auto found = std::get<dep_gather_cache_t<asset::ICPUShader>>(dep_gather_caches).find(asset);
			if (!=end())
			{
			}
		}
*/
	}

		// check whether the item is creatable after patching, else duplicate/de-alias

	// now we have a list of gpu creation parameters we want to create resources with

#if 0
	auto stuff = [&]<typename AssetType>(const input_t<AssetType>& key)->void
	{
		//
		CCache::search_t<AssetType> s = key;
		if (!s.patch.has_value())
			s.patch = asset_traits<AssetType>::defaultPatch(key.asset);
		const size_t hash = CCache::Hash{}(s);

		// search by the {asset+patch} in read cache
		if (auto found=input.readCache->find(s,hash); found!=input.readCache->end<AssetType>())
		{
			//
		}

		// reserve in write cache
	};
#endif
	
	// if read cache
		// find deps in read cache
		// mark deps as ready/found
	// if not ready/found
		// find dep in transient cache
		// if not found, insert and recurse

	// see what can be patched/combined

	return retval;
}

bool CAssetConverter::convert(SResults& reservations, SConvertParams& params)
{
	if (!reservations.reserveSuccess())
		return false;

	const auto reqQueueFlags = reservations.getRequiredQueueFlags();
	if (reqQueueFlags.hasFlags(IQueue::FAMILY_FLAGS::TRANSFER_BIT|IQueue::FAMILY_FLAGS::COMPUTE_BIT) && !params.utilities)
		return false;

	auto invalidQueue = [reqQueueFlags](const IQueue::FAMILY_FLAGS flag, IQueue* queue)->bool
	{
		if (!reqQueueFlags.hasFlags(flag))
			return false;
		if (!queue || queue->getFamilyIndex())
			return true;
		return false;
	};
	if (reqQueueFlags.hasFlags(IQueue::FAMILY_FLAGS::TRANSFER_BIT) && (!params.transfer.queue || params.transfer.queue->getFamilyIndex() || !params.utilities))
		return false;
	if (reqQueueFlags.hasFlags(IQueue::FAMILY_FLAGS::COMPUTE_BIT) && (!params.compute.queue || !params.utilities))
		return false;

	const core::string debugPrefix = "Created by Converter "+std::to_string(ptrdiff_t(this))+" with hash ";
	auto setDebugName = [&debugPrefix](IBackendObject* obj, const size_t hashval)->void
	{
		if (obj)
			obj->setObjectDebugName((debugPrefix+std::to_string(hashval)).c_str());
	};

	auto device = m_params.device;
	// create shaders
	{
		ILogicalDevice::SShaderCreationParameters params = {
			.optimizer = m_params.optimizer.get(),
			.readCache = m_params.compilerCache.get(),
			.writeCache = m_params.compilerCache.get()
		};

		{
			params.cpushader = nullptr; // TODO
			reservations.get<asset::ICPUShader>().emplace_back(device->createShader(params));
		}
	}

	return true;
}

}