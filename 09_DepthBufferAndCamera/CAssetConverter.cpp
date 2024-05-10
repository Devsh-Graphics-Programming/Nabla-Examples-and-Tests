// Copyright (C) 2024-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
#include "CAssetConverter.h"


namespace nbl::video
{
using namespace nbl::core;
using namespace nbl::asset;

auto CAssetConverter::reserve(const SInput& input) -> SResults
{
	SResults retval = {};
	if (input.readCache && input.readCache->m_params.device!=m_params.device)
		return retval;

	// gather all dependencies (DFS graph search) and patch, this happens top-down
	// do not deduplicate/merge assets at this stage, only patch GPU creation parameters
	{
		core::stack<const IAsset*> dfsStack;
		// returns true if new element was inserted
		auto cache = [&]<Asset AssetType>(const CAssetConverter::SInput::input_t<AssetType>& in)->bool
		{
			if (!in.key.asset)
				return false;

			using cache_t = SResults::dag_cache_t<AssetType>;
			auto& cache = std::get<cache_t>(retval.m_typedDagNodes);
			auto found = cache.equal_range(in.key);
			if (found.first!=found.second)
			{
#if 0
				// found the thing, combine patches
				const auto& cachedPatch = found->patch;
				if (auto combined=in.patch; combined)
				{
					const_cast<asset_traits<AssetType>::patch_t&>(cachedPatch) = combined;
					return false;
				}
#endif
				// check whether the item is creatable after patching, else duplicate/de-alias
			}
			// insert a new entry
			cache.insert(found.first,{in.key,{.patch=in.patch}});
			return true;
		};
		// initialize stacks
		core::visit([&]<Asset AssetType>(const SInput::span_t<AssetType> inputs)->void{
			for (auto& in : inputs)
			if (cache(in) && AssetType::HasDependents)
				dfsStack.push(in.key.asset);
		},input.assets);
		// everything that's not explicit has `!unique` and default patch params
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
	// now we have a set of implicit gpu creation parameters we want to create resources with
	// and a mapping from (Asset,Patch) -> UniqueAsset

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

		for (auto& shader : std::get<SResults::dag_cache_t<ICPUShader>>(reservations.m_typedDagNodes))
		if (!shader.second.canonical)
		{
			assert(!shader.second.result);
			// custom code start
			params.cpushader = shader.first.asset;
			shader.second.result = device->createShader(params);
		}
	}

	return true;
}

}