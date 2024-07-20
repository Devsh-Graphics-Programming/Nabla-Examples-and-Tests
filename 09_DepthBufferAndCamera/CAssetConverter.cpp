// Copyright (C) 2024-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
#include "CAssetConverter.h"

#include <type_traits>


using namespace nbl::core;
using namespace nbl::asset;


namespace nbl::video
{
CAssetConverter::patch_impl_t<ICPUSampler>::patch_impl_t(
	const ICPUSampler* sampler,
	const SPhysicalDeviceFeatures& features,
	const SPhysicalDeviceLimits& limits
) : anisotropyLevelLog2(sampler->getParams().AnisotropicFilter)
{
	if (anisotropyLevelLog2>limits.maxSamplerAnisotropyLog2)
		anisotropyLevelLog2 = limits.maxSamplerAnisotropyLog2;
}

/** repurpose for pipeline
CAssetConverter::patch_impl_t<ICPUShader>::patch_impl_t(
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
}*/

CAssetConverter::patch_impl_t<ICPUBuffer>::patch_impl_t(
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

CAssetConverter::patch_impl_t<ICPUPipelineLayout>::patch_impl_t(
	const ICPUPipelineLayout* pplnLayout,
	const SPhysicalDeviceFeatures& features,
	const SPhysicalDeviceLimits& limits
) : patch_impl_t()
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

//
void CAssetConverter::CHashCache::eraseStale()
{
	auto rehash = [&]<typename AssetType>() -> void
	{
		auto& container = std::get<container_t<AssetType>>(m_containers);
		core::erase_if(container,[this](const auto& entry)->bool
			{
				// backup because `hash(lookup)` call will update it
				const auto oldHash = entry.second;
				const auto& key = entry.first;
				hash_request_t<AssetType> lookup = {
					{
						.asset = key.asset.get(),
						.patch = &key.patch,
					},
					// can re-use cached hashes for dependants if we start ejecting in the correct order
					/*.cacheMistrustLevel = */1
				};
				return hash(lookup)!=oldHash;
			}
		);
	};
	// to make the process more efficient we start ejecting from "lowest level" assets
	rehash.operator()<ICPUSampler>();
	rehash.operator()<ICPUShader>();
	rehash.operator()<ICPUBuffer>();
//	rehash.operator()<ICPUBufferView>();
	rehash.operator()<ICPUDescriptorSetLayout>();
	rehash.operator()<ICPUPipelineLayout>();
}


template<>
void CAssetConverter::CHashCache::hash_impl<ICPUShader>(::blake3_hasher& hasher, const ICPUShader* asset, const patch_t<ICPUShader>& patch, const uint32_t nextMistrustLevel)
{
	core::blake3_hasher_update(hasher,asset->getContentType());
	const auto* content = asset->getContent();
	// we're not using the buffer directly, just its contents
	core::blake3_hasher_update(hasher,content->getContentHash());
}

template<>
void CAssetConverter::CHashCache::hash_impl<ICPUSampler>(::blake3_hasher& hasher, const ICPUSampler* asset, const patch_t<ICPUSampler>& patch, const uint32_t nextMistrustLevel)
{
	auto patchedParams = asset->getParams();
	patchedParams.AnisotropicFilter = patch.anisotropyLevelLog2;
	core::blake3_hasher_update(hasher,patchedParams);
}

template<>
void CAssetConverter::CHashCache::hash_impl<ICPUBuffer>(::blake3_hasher& hasher, const ICPUBuffer* asset, const patch_t<ICPUBuffer>& patch, const uint32_t nextMistrustLevel)
{
	auto patchedParams = asset->getCreationParams();
	assert(patch.usage.hasFlags(patchedParams.usage));
	patchedParams.usage = patch.usage;
	core::blake3_hasher_update(hasher,patchedParams);
	core::blake3_hasher_update(hasher,asset->getContentHash());
}

template<>
void CAssetConverter::CHashCache::hash_impl<ICPUDescriptorSetLayout>(::blake3_hasher& hasher, const ICPUDescriptorSetLayout* asset, const patch_t<ICPUDescriptorSetLayout>& patch, const uint32_t nextMistrustLevel)
{
//	asset->getImmutableSamplers();
//	core::blake3_hasher_update(hasher,);
}

template<>
void CAssetConverter::CHashCache::hash_impl<ICPUPipelineLayout>(::blake3_hasher& hasher, const ICPUPipelineLayout* asset, const patch_t<ICPUPipelineLayout>& patch, const uint32_t nextMistrustLevel)
{
	core::blake3_hasher_update(hasher,patch.pushConstantBytes);
	for (auto i=0; i<ICPUPipelineLayout::DESCRIPTOR_SET_COUNT; i++)
	{
		hash<ICPUDescriptorSetLayout>({
			{
				.asset = asset->getDescriptorSetLayout(i),
				.patch = {}, // there's nothing to patch
			},
			nextMistrustLevel
		});
	}
}


// question of propagating changes, image view and buffer view
// if image view used as STORAGE, or SAMPLED have to change subUsage
// then need to propagate change of subUsage to usage
// therefore image is now changed
// have to pass the patch around

/*
// need to map (asset,uniqueGroup,patch) -> blake3_hash to avoid re-hashing
struct dedup_entry_t
{
	core::blake3_hash_t patchedHash;
	core::blake3_hash_t unpatchedHash;
};


When to look for assets in the read cache?

Ideally when traversing already because then we can skip DAG subgraph exploration.

But we need to explore the DAG to hash anyway.


How to "amortized hash"?

Grab (asset,group,patch) hash the asset params, group and patch, then update with dependents:
- this requires building or retrieving patches for dependents
- lookup with (dependent,group,patch)
- if lookup fails, proceed to compute full hash and insert it into cache

We know pointer, so can actually trim hashes of stale assets (same pointer, different hash) if we do full recompute.

*/

template<asset::Asset AssetType>
using patch_vector_t = core::vector<CAssetConverter::patch_t<AssetType>>;

// not sure if useful enough to move to core utils
template<typename T, typename TypeList>
struct index_of;
template<typename T>
struct index_of<T,core::type_list<>> : std::integral_constant<size_t,0> {};
template<typename T, typename... Us>
struct index_of<T,core::type_list<T,Us...>> : std::integral_constant<size_t,0> {};
template<typename T, typename U, typename... Us>
struct index_of<T,core::type_list<U,Us...>> : std::integral_constant<size_t,1+index_of<T,core::type_list<Us...>>::value> {};
template<typename T, typename TypeList>
inline constexpr size_t index_of_v = index_of<T,TypeList>::value;

//
struct instance_metadata_t
{
	inline bool operator==(const instance_metadata_t&) const = default;

	size_t uniqueCopyGroupID = 0xdeadbeefBADC0FFEull;
	size_t patchIndex = 0xdeadbeefBADC0FFEull;
};

//
template<asset::Asset AssetType>
struct instance_t
{
	inline bool operator==(const instance_t<AssetType>&) const = default;

	const AssetType* asset = nullptr;
	instance_metadata_t meta = {};
};

//
template<asset::Asset AssetType>
struct unique_conversion_t
{
	instance_t<AssetType> canonical = {};
	asset_cached_t<AssetType> gpuObj = {};
};

// nice and polymorphic
struct dfs_entry_t
{
	inline bool operator==(const dfs_entry_t&) const = default;

	const IAsset* asset;
	instance_metadata_t instance;
};

// This cache stops us traversing an asset with the same user group and patch more than once.
template<asset::Asset AssetType>
// Maps `instance_t` to `patchIndex` and optionally `gpuObj` if it can be retrieved
struct dfs_cache_hash_and_equals
{
	using is_transparent = void;

	static inline dfs_entry_t uniqueToDFS(const unique_conversion_t<AssetType>& entry)
	{
		return dfs_entry_t{.asset=entry.canonical.asset,.instance=entry.canonical.meta};
	}

	inline size_t operator()(const dfs_entry_t& entry) const
	{
		return ptrdiff_t(entry.asset)^entry.instance.uniqueCopyGroupID;
	}

	inline size_t operator()(const unique_conversion_t<AssetType>& entry) const
	{
		// its very important to cast the derived AssetType to IAsset because otherwise pointers won't match
		return operator()(uniqueToDFS(entry));
	}
	
	inline bool operator()(const dfs_entry_t& lhs, const unique_conversion_t<AssetType>& rhs) const
	{
		return lhs==uniqueToDFS(rhs);
	}
	inline bool operator()(const unique_conversion_t<AssetType>& lhs, const dfs_entry_t& rhs) const
	{
		return uniqueToDFS(lhs)==rhs;
	}
	inline bool operator()(const unique_conversion_t<AssetType>& lhs, const unique_conversion_t<AssetType>& rhs) const
	{
		return uniqueToDFS(lhs)==uniqueToDFS(rhs);
	}
};
template<asset::Asset AssetType>
using dfs_cache_t = core::unordered_multiset<unique_conversion_t<AssetType>,dfs_cache_hash_and_equals<AssetType>,dfs_cache_hash_and_equals<AssetType>>;

//
template<asset::Asset AssetType>
using conversions_t = core::unordered_map<core::blake3_hash_t,unique_conversion_t<AssetType>>;

//
auto CAssetConverter::reserve(const SInputs& inputs) -> SResults
{
	auto* const device = m_params.device;
	if (inputs.readCache && inputs.readCache->m_params.device!=m_params.device)
		return {};
	if (inputs.pipelineCache && inputs.pipelineCache->getOriginDevice()!=device)
		return {};

	core::smart_refctd_ptr<CHashCache> hashCache;
	if (inputs.hashCache)
		hashCache = core::smart_refctd_ptr<CHashCache>(inputs.hashCache);
	else
		hashCache = core::make_smart_refctd_ptr<CHashCache>();

	SResults retval = {};
	
	// this will allow us to look up the conversion parameter (actual patch for an asset) and therefore write the GPUObject to the correct place in the return value
	core::vector<instance_metadata_t> inputsMetadata[core::type_list_size_v<supported_asset_types>];
	// stop multiple copies of the patches floating around
	core::tuple_transform_t<patch_vector_t,supported_asset_types> finalPatchStorage = {};

	{
		core::tuple_transform_t<dfs_cache_t,supported_asset_types> dfsCaches = {};
		// gather all dependencies (DFS graph search) and patch, this happens top-down
		// do not deduplicate/merge assets at this stage, only patch GPU creation parameters
		{
			core::stack<dfs_entry_t> dfsStack;
			// returns true if new element was inserted
			auto cache = [&]<Asset AssetType>(const dfs_entry_t& user, const AssetType* asset, patch_t<AssetType>&& patch) -> instance_metadata_t
			{
				assert(asset);
				// skip invalid inputs silently
				if (!patch.valid())
					return {};

				// get unique group and see if we visited already
				dfs_entry_t record = {
					.asset = asset,
					.instance = {
						.uniqueCopyGroupID = inputs.getDependantUniqueCopyGroupID(user.instance.uniqueCopyGroupID,user.asset,asset)
					}
				};

				//
#if 0
				{
					// lets try to find an already converted GPU object corresponding to this
					if (inputs.readCache)
					{
						// first find our content hash (blind find, we're trying to see if this asset ptr+patch has been hashed before)
						const auto contentHashFound = hashCache->find<AssetType>({.asset=asset,.patch=&patch});
						if (contentHashFound!=hashCache->end<AssetType>())
						{
							const auto& conversionCache = inputs.readCache->getCache<AssetType>();
							// found a content hash, lets try to find the converted gpu object
							const auto foundGPUObj = conversionCache.find({contentHashFound->second,record.instance.uniqueCopyGroupID});
							if (foundGPUObj)
							{
								// now return it
							}
						}
					}
				}
#endif
				auto& patchStorage = std::get<patch_vector_t<AssetType>>(finalPatchStorage);

				auto& dfsCache = std::get<dfs_cache_t<AssetType>>(dfsCaches);
				// iterate over all intended EXTRA copies of an asset
				auto found = dfsCache.equal_range(record);
				for (auto it=found.first; it!=found.second; it++)
				{
					const auto patchIndex = it->canonical.meta.patchIndex;
					auto& candidate = patchStorage[patchIndex];
					// found a thing, try-combine the patches
					auto combined = candidate.combine(patch);
					// check whether the item is creatable after patching
					if (combined.valid())
					{
						record.instance.patchIndex = patchIndex;
						candidate = std::move(combined);
						return record.instance;
					}
					// else try the next one
				}
				// Either haven't managed to combine with anything or no entry found
				record.instance.patchIndex = patchStorage.size();
				patchStorage.push_back(std::move(patch));
				if (AssetType::HasDependents)
					dfsStack.emplace(record);
				dfsCache.emplace(instance_t<AssetType>{.asset=asset,.meta=record.instance},asset_cached_t<AssetType>{});
				return record.instance;
			};
			//
			const auto& features = device->getEnabledFeatures();
			const auto& limits = device->getPhysicalDevice()->getLimits();
			// initialize stacks
			auto initialize = [&]<typename AssetType>(const std::span<const AssetType* const> assets)->void
			{
				const auto count = assets.size();
				const auto& patches = std::get<SInputs::patch_span_t<AssetType>>(inputs.patches);
				// size and fill the result array with nullptr
				std::get<SResults::vector_t<AssetType>>(retval.m_gpuObjects).resize(count);
				// size the final patch mapping
				auto& metadata = inputsMetadata[index_of_v<AssetType,supported_asset_types>];
				metadata.resize(count);
				for (size_t i=0; i<count; i++)
				if (auto asset=assets[i]; asset) // skip invalid inputs silently
				{
					// for explicitly given patches we don't try to create implicit patch and merge that with the explicit
					// we trust the implicit patches are correct/feasible
					patch_t<AssetType> patch = i<patches.size() ? patches[i]:patch_t<AssetType>(asset,features,limits);
					metadata[i] = cache({},asset,std::move(patch));
				}
			};
			core::for_each_in_tuple(inputs.assets,initialize);
			// everything that's not explicit has `uniqueCopyForUser==nullptr` and default patch params
			while (!dfsStack.empty())
			{
				const auto entry = dfsStack.top();
				dfsStack.pop();
				// everything we popped, has already been cached, now time to go over dependents
				const auto* user = entry.asset;
				switch (user->getAssetType())
				{
					case ICPUPipelineLayout::AssetType:
					{
						auto pplnLayout = static_cast<const ICPUPipelineLayout*>(user);
						for (auto i=0; i<ICPUPipelineLayout::DESCRIPTOR_SET_COUNT; i++)
						if (auto layout=pplnLayout->getDescriptorSetLayout(i); layout)
							cache.operator()<ICPUDescriptorSetLayout>(entry,layout,{layout,features,limits});
						break;
					}
					case ICPUDescriptorSetLayout::AssetType:
					{
						auto layout = static_cast<const ICPUDescriptorSetLayout*>(user);
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
						auto view = static_cast<const ICPUBufferView*>(user);
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

		// This map contains the assets by-hash, identical asset+patch hash the same.
		core::tuple_transform_t<conversions_t,supported_asset_types> uniqueGpuObjects;
		auto dedup = [&]<Asset AssetType>()->void
		{
			// go over all `dfsCache` entries
			
			// hash them

			// insert them

			// if insertion prevented ++ the counter of copies
		};
		// Lets see if we can collapse any of the (Asset Content) into the same thing,
		// to correctly de-dup we need to go bottom-up!!!
		dedup.operator()<ICPUShader>();
	}
	// Now we have a set of implicit gpu creation parameters we want to create resources with.
	// If there's a readCache we need to look for an item there first.
	// We can't look up "near misses" (supersets) because they'd have different hashes
	// and we can't afford to split hairs like finding overlapping buffer ranges, etc.
	// Stuff like that would require a completely different hashing/lookup strategy (or multiple fake entries).



// Shader, DSLayout, PipelineLayout, Compute Pipeline
// Renderpass, Graphics Pipeline
// Buffer, BufferView, Sampler, Image, Image View, Bottom Level AS, Top Level AS, Descriptor Set, Framebuffer  
// Buffer -> SRange, patched usage, owner(s)
// BufferView -> SRange, promoted format
// Sampler -> Clamped Params (only aniso, really)
// Image -> this, patched usage, promoted format
// Image View -> ref to patched Image, patched usage, promoted format
// Descriptor Set -> unique layout, 
	
	// its a very good idea to set debug names on everything!
	auto setDebugName = [this](IBackendObject* obj, const core::blake3_hash_t& hashval)->void
	{
		if (obj)
		{
			std::ostringstream debugName;
			debugName << "Created by Converter ";
			debugName << std::hex;
			debugName << this;
			debugName << " from Asset with hash ";
			for (const auto& byte : hashval.data)
				debugName << uint32_t(byte) << " ";
			obj->setObjectDebugName(debugName.str().c_str());
		}
	};

	// create shaders
	{
		ILogicalDevice::SShaderCreationParameters createParams = {
			.optimizer = m_params.optimizer.get(),
			.readCache = inputs.readShaderCache,
			.writeCache = inputs.writeShaderCache
		};
#if 0
		for (auto& entry : shadersToCreate)
		{
			assert(!shader.second.result);
			// custom code start
			params.cpushader = shader.first.asset;
			shader.second.result = device->createShader(params);
		}
#endif
	}

	core::unordered_map<core::blake3_hash_t,patch_t<ICPUBuffer>> buffersToCreate;
	// create IGPUBuffers
	for (auto& entry : buffersToCreate)
	{
		IGPUBuffer::SCreationParams params = {};
		params.size = 0x45;
		params.usage = entry.second.usage;
		// TODO: make this configurable
		params.queueFamilyIndexCount = 0;
		params.queueFamilyIndices = nullptr;
		auto buffer = device->createBuffer(std::move(params));
	}

	// create IGPUImages

	// gather memory reqs
	// allocate device memory
//	device->allocate(reqs,false);

	// create IGPUBufferViews

	// create IGPUImageViews


	// insert whatever assets don't need conversion to the cache
//	core::for_each_in_tuple(uniqueGpuObjects,cacheInsert);

	// write out results
	auto finalize = [&]<typename AssetType>(const std::span<const AssetType* const> assets)->void
	{
		const auto count = assets.size();
		//
		const auto& metadata = inputsMetadata[index_of_v<AssetType,supported_asset_types>];
		const auto& patchStorage = std::get<patch_vector_t<AssetType>>(finalPatchStorage);
		const auto& staging = std::get<CCache<AssetType>>(retval.m_stagingCache);
		auto& results = std::get<SResults::vector_t<AssetType>>(retval.m_gpuObjects);
		//
		for (size_t i=0; i<count; i++)
		if (auto asset=assets[i]; asset && metadata[i].patchIndex<patchStorage.size())
		{
			const auto* patch = patchStorage.data()+metadata[i].patchIndex;
			if (!patch->valid())
				continue;
			const core::blake3_hash_t hash = hashCache->hash<AssetType>({{
				.asset=asset,
				.patch=patch
			}});
			// lookup the hash in the local cache (it must be present)
			const auto found = staging.find({hash,metadata[i].uniqueCopyGroupID});
			// write it out to the results
			results[i] = *found;
		}
	};
	core::for_each_in_tuple(inputs.assets,finalize);

	return retval;
}

//
bool CAssetConverter::convert(SResults&& reservations, SConvertParams& params)
{
	const auto reqQueueFlags = reservations.getRequiredQueueFlags();
	// nothing to do!
	if (reqQueueFlags.value==IQueue::FAMILY_FLAGS::NONE)
		return true;

	auto device = m_params.device;
	if (reqQueueFlags.hasFlags(IQueue::FAMILY_FLAGS::TRANSFER_BIT) && (!params.utilities || params.utilities->getLogicalDevice()!=device))
		return false;

	auto invalidQueue = [reqQueueFlags,device,&params](const IQueue::FAMILY_FLAGS flag, IQueue* queue)->bool
	{
		if (!reqQueueFlags.hasFlags(flag))
			return false;
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

	return true;
}

ISemaphore::future_t<bool> CAssetConverter::SConvertParams::autoSubmit()
{
	// TODO: transfer first, then compute
	return {};
}

}