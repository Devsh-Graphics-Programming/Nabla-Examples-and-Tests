// Copyright (C) 2024-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
#include "CAssetConverter.h"

#include <type_traits>


using namespace nbl::core;
using namespace nbl::asset;


// if you end up specializing `patch_t` for any type because its non trivial and starts needing weird stuff done with memory, you need to spec this as well
namespace nbl
{
template<asset::Asset AssetType, typename Dummy>
struct core::blake3_hasher::update_impl<video::CAssetConverter::patch_t<AssetType>,Dummy>
{
	static inline void __call(blake3_hasher& hasher, const video::CAssetConverter::patch_t<AssetType>& input)
	{
		hasher.update(&input,sizeof(input));
	}
};

namespace video
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
}

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
	rehash.operator()<ICPUDescriptorSetLayout>();
	rehash.operator()<ICPUPipelineLayout>();
	// shaders and images depend on buffers for data sourcing
	rehash.operator()<ICPUBuffer>();
//	rehash.operator()<ICPUBufferView>();
//	rehash.operator()<ICPUImage>();
//	rehash.operator()<ICPUImageView>();
//	rehash.operator()<ICPUBottomLevelAccelerationStructure>();
//	rehash.operator()<ICPUTopLevelAccelerationStructure>();
	// only once all the descriptor types have been hashed, we can hash sets
//	rehash.operator()<ICPUDescriptorSet>();
	// naturally any pipeline depends on shaders and pipeline cache
	rehash.operator()<ICPUShader>();
	rehash.operator()<ICPUPipelineCache>();
	rehash.operator()<ICPUComputePipeline>();
	// graphics pipeline needs a renderpass
//	rehash.operator()<ICPURenderpass>();
//	rehash.operator()<ICPUGraphicsPipeline>();
//	rehash.operator()<ICPUFramebuffer>();
}


template<>
void CAssetConverter::CHashCache::hash_impl<ICPUSampler>(core::blake3_hasher& hasher, const ICPUSampler* asset, const patch_t<ICPUSampler>& patch, const uint32_t nextMistrustLevel)
{
	auto patchedParams = asset->getParams();
	patchedParams.AnisotropicFilter = patch.anisotropyLevelLog2;
	hasher.update(&patchedParams,sizeof(patchedParams));
}

template<>
void CAssetConverter::CHashCache::hash_impl<ICPUShader>(core::blake3_hasher& hasher, const ICPUShader* asset, const patch_t<ICPUShader>& patch, const uint32_t nextMistrustLevel)
{
	const auto type = asset->getContentType();
	hasher << type;
	// if not SPIR-V then own path matters
	if (type!=ICPUShader::E_CONTENT_TYPE::ECT_SPIRV)
		hasher << asset->getFilepathHint();
	const auto* content = asset->getContent();
	// we're not using the buffer directly, just its contents
	hasher << content->getContentHash();
}

template<>
void CAssetConverter::CHashCache::hash_impl<ICPUBuffer>(core::blake3_hasher& hasher, const ICPUBuffer* asset, const patch_t<ICPUBuffer>& patch, const uint32_t nextMistrustLevel)
{
	auto patchedParams = asset->getCreationParams();
	assert(patch.usage.hasFlags(patchedParams.usage));
	patchedParams.usage = patch.usage;
	hasher.update(&patchedParams,sizeof(patchedParams)) << asset->getContentHash();
}

template<>
void CAssetConverter::CHashCache::hash_impl<ICPUDescriptorSetLayout>(core::blake3_hasher& hasher, const ICPUDescriptorSetLayout* asset, const patch_t<ICPUDescriptorSetLayout>& patch, const uint32_t nextMistrustLevel)
{
	const auto& immutableSamplerRedirect = asset->getImmutableSamplerRedirect();
	const auto count = immutableSamplerRedirect.getBindingCount();
	for (auto i=0; i<count; i++)
	{
		const ICPUDescriptorSetLayout::CBindingRedirect::storage_range_index_t storageRangeIx(i);
		const auto binding = immutableSamplerRedirect.getBinding(storageRangeIx);
		// need to hash not only the sampler state, but the slots they're supposed to go into
		hasher << binding.data;
		// now the sampler itself
		hasher << hash<ICPUSampler>({
			{
				.asset = asset->getImmutableSamplers()[i].get(),
				.patch = {}, // TODO: FIND IT !
			},
			nextMistrustLevel
		});
	}
}

template<>
void CAssetConverter::CHashCache::hash_impl<ICPUPipelineLayout>(core::blake3_hasher& hasher, const ICPUPipelineLayout* asset, const patch_t<ICPUPipelineLayout>& patch, const uint32_t nextMistrustLevel)
{
	hasher << std::span(patch.pushConstantBytes);
	for (auto i=0; i<ICPUPipelineLayout::DESCRIPTOR_SET_COUNT; i++)
	{
		hasher << hash<ICPUDescriptorSetLayout>({
			{
				.asset = asset->getDescriptorSetLayout(i),
				.patch = {}, // there's nothing to patch in a DS Layout
			},
			nextMistrustLevel
		});
	}
}

template<>
void CAssetConverter::CHashCache::hash_impl<ICPUPipelineCache>(core::blake3_hasher& hasher, const ICPUPipelineCache* asset, const patch_t<ICPUPipelineCache>& patch, const uint32_t nextMistrustLevel)
{
	for (const auto& entry : asset->getEntries())
	{
		hasher << entry.first.deviceAndDriverUUID;
		if (entry.first.meta)
			hasher.update(entry.first.meta->data(),entry.first.meta->size());
	}
	hasher << asset->getContentHash();
}

template<>
void CAssetConverter::CHashCache::hash_impl<ICPUComputePipeline>(core::blake3_hasher& hasher, const ICPUComputePipeline* asset, const patch_t<ICPUComputePipeline>& patch, const uint32_t nextMistrustLevel)
{
	hash<ICPUPipelineLayout>({
		{
			.asset = asset->getLayout(),
			.patch = {}, // TODO: how to get!?
		},
		nextMistrustLevel
	});
	hash<ICPUShader>({
		{
			.asset = asset->getSpecInfo().shader,
			.patch = {}, // TODO: how to get!?
		},
		nextMistrustLevel
	});
}


// question of propagating changes, image view and buffer view
// if image view used as STORAGE, or SAMPLED have to change subUsage
// then need to propagate change of subUsage to usage
// therefore image is now changed
// have to pass the patch around


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
	inline bool operator!=(const instance_metadata_t&) const = default;

	explicit inline operator bool() const {return operator!=({}); }

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
struct dfs_result_t
{
	instance_t<AssetType> canonical = {};
	mutable core::blake3_hash_t contentHash = {};
	mutable asset_cached_t<AssetType> gpuObj = {};
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

	static inline dfs_entry_t uniqueToDFS(const dfs_result_t<AssetType>& entry)
	{
		return dfs_entry_t{.asset=entry.canonical.asset,.instance=entry.canonical.meta};
	}

	inline size_t operator()(const dfs_entry_t& entry) const
	{
		return ptrdiff_t(entry.asset)^entry.instance.uniqueCopyGroupID;
	}

	inline size_t operator()(const dfs_result_t<AssetType>& entry) const
	{
		// its very important to cast the derived AssetType to IAsset because otherwise pointers won't match
		return operator()(uniqueToDFS(entry));
	}
	
	inline bool operator()(const dfs_entry_t& lhs, const dfs_result_t<AssetType>& rhs) const
	{
		return lhs==uniqueToDFS(rhs);
	}
	inline bool operator()(const dfs_result_t<AssetType>& lhs, const dfs_entry_t& rhs) const
	{
		return uniqueToDFS(lhs)==rhs;
	}
	inline bool operator()(const dfs_result_t<AssetType>& lhs, const dfs_result_t<AssetType>& rhs) const
	{
		return uniqueToDFS(lhs)==uniqueToDFS(rhs);
	}
};
template<asset::Asset AssetType>
using dfs_cache_t = core::unordered_multiset<dfs_result_t<AssetType>,dfs_cache_hash_and_equals<AssetType>,dfs_cache_hash_and_equals<AssetType>>;

//
template<asset::Asset AssetType>
struct unique_conversion_t
{
	const AssetType* canonicalAsset = nullptr;
	size_t patchIndex = 0xdeadbeefBAull;
	size_t firstCopyIx : 40 = 0u;
	size_t copyCount : 24 = 1u;
};

// Map from ContentHash to canonical asset & patch and the list of uniqueCopyGroupIDs
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

	SResults retval = {};

	// No asset has a 0 length input to the hash function
	const auto NoContentHash = static_cast<core::blake3_hash_t>(core::blake3_hasher());
	
	// this will allow us to look up the conversion parameter (actual patch for an asset) and therefore write the GPUObject to the correct place in the return value
	core::vector<instance_metadata_t> inputsMetadata[core::type_list_size_v<supported_asset_types>];
	// stop multiple copies of the patches floating around
	core::tuple_transform_t<patch_vector_t,supported_asset_types> finalPatchStorage = {};
	// One would think that we first need an (AssetPtr,Patch) -> ContentHash map and then a ContentHash -> GPUObj map to
	// save ourselves iterating over redundant assets. The truth is that we going from a ContentHash to GPUObj is blazing fast.
	core::tuple_transform_t<dfs_cache_t,supported_asset_types> dfsCaches = {};

	{
		// gather all dependencies (DFS graph search) and patch, this happens top-down
		// do not deduplicate/merge assets at this stage, only patch GPU creation parameters
		{
			// to do GPU Object lookups during DFS phase, we need a reliable (IAsset*,patch)->ContentHash cache first, otherwise we'd be doing O(N^2) during hashing
			const bool lookupGPUObjDuringDFS = inputs.hashCache && inputs.readCache;
			core::stack<dfs_entry_t> dfsStack;
			// returns `instance_metadata_t` which you can `bool(instance_metadata_t)` to find out if a new element was inserted
			auto cache = [&]<Asset AssetType>(const dfs_entry_t& user, const AssetType* asset, patch_t<AssetType>&& patch) -> instance_metadata_t
			{
				assert(asset);
				// skip invalid inputs silently
				if (!patch.valid())
					return {};

				// get unique group
				dfs_entry_t record = {
					.asset = asset,
					.instance = {
						.uniqueCopyGroupID = inputs.getDependantUniqueCopyGroupID(user.instance.uniqueCopyGroupID,user.asset,asset)
					}
				};

				// all entries refer to patch by index, so its stable against vector growth
				// NOTE: There's a 1:1 correspondence between `dfsCache` entries and `finalPatchStorage` entries!
				auto& patchStorage = std::get<patch_vector_t<AssetType>>(finalPatchStorage);

				// now see if we visited already
				auto& dfsCache = std::get<dfs_cache_t<AssetType>>(dfsCaches);
				// get all the existing patches for the same (AssetType*,UniqueCopyGroupID)
				auto dfsCachedRange = dfsCache.equal_range(record);

				// if we will do GPU object lookups we need to back up the first mergable entry in case we find nothing
				typename dfs_cache_t<AssetType>::const_iterator mergableIt = dfsCachedRange.second;
				// we may need the first combined patch candidate later if we can `lookupGPUObjDuringDFS` but don't manage to find anything
				patch_t<AssetType> combined;
				// iterate over all intended EXTRA copies of an asset
				for (auto it=dfsCachedRange.first; it!=dfsCachedRange.second; it++)
				{
					// get our candidate patch
					const auto patchIndex = it->canonical.meta.patchIndex;
					const auto& candidate = patchStorage[patchIndex];
					// found a thing, try-combine the patches
					combined = candidate.combine(patch);
					// check whether the item is creatable after patching
					if (combined.valid())
					{
						if (lookupGPUObjDuringDFS)
						{
							// Check if a GPU object was already found for the cache before, but we can only use this GPU object if its patch is a superset of ours,
							// i.e. the candidate cannot change! But we already looked up a GPU object for it, treat same as Not Found if would have to change the patch.
							if (it->gpuObj && combined==candidate)
							{
								// no patch std::move needed because it needs to stay the same
								record.instance.patchIndex = patchIndex;
								return record.instance;
							}
							// We don't do a lookup of a GPU object in the cache because it should have already looked up when emplaced as a new entry into the dfsCache.
							// But we back up the entry that's mergable, so we can merge and do a lookup later.
							if (mergableIt!=dfsCachedRange.second)
							{
								record.instance.patchIndex = patchIndex;
								mergableIt = it;
							}
							// but we continue iterating because we want to keep looking for compatible GPU objects in the current equal range
						}
						else // no GPU lookups to do, can return right away
						{
							// change the patch to a combined version
							patchStorage[patchIndex] = std::move(combined);
							record.instance.patchIndex = patchIndex;
							return record.instance;
						}
					}
					// else try the next one
				}

				// Handle, no GPU objects with compatible patches found in the dfsCache
				if (lookupGPUObjDuringDFS)
				{
					// now we'll look in the read-only conversion cache (from this or another CAssetConverter)
					const auto& conversionCache = inputs.readCache->getCache<AssetType>();
					// lambda for doing lookups, returns a pointer to a cached type if successful
					auto findGPUObj = [&](const patch_t<AssetType>* _patch)->std::pair<core::blake3_hash_t,const asset_cached_t<AssetType>*>
					{
						// Find our content hash (blind find, we're trying to see if this asset ptr+patch has been hashed before)
						// lookup Hash Cache using asset+patch, using find to not trigger hash recompute until finished DFS
						// NOTE: its super important that the `input.hashCache` entries for `asset*.patch` are not stale!
						const auto contentHashFound = inputs.hashCache->find<AssetType>({.asset=asset,.patch=_patch});
						if (contentHashFound!=inputs.hashCache->end<AssetType>())
						{
							// found a content hash, lets try to find the converted gpu object
							return {contentHashFound->second,conversionCache.find({contentHashFound->second,record.instance.uniqueCopyGroupID})};
						}
						return {NoContentHash,nullptr};
					};
					// First look for unpatched/unmerged self.
					if (const auto [contentHash,found] = findGPUObj(&patch); found)
					{
						// this is a new never before seen patch, we need to push it
						record.instance.patchIndex = patchStorage.size();
						patchStorage.push_back(std::move(patch));
						// obviously not in the DFS cache so need to insert
						dfsCache.emplace(instance_t<AssetType>{.asset=asset,.meta=record.instance},contentHash,*found);
						return record.instance;
					}
					else if (mergableIt!=dfsCachedRange.second) // if there's a valid merge candidate
					{
						// we can now do same as the `!lookupGPUObjDuringDFS` case during merging and make the merge take effect
						const auto& effective = (patchStorage[record.instance.patchIndex] = std::move(combined));
						// then look for self with that combined patch
						const auto [contentHash,found] = findGPUObj(&patch);
						// even if we haven't found a GPU object we'll store the content hash for this patch
						mergableIt->contentHash = contentHash;
						// instead of emplacing a new entry in DFS cache, update it, now the assigned patch (and contentHash) will not mutate
						if (found)
							mergableIt->gpuObj = *found;
						// NOTE: if we don't find a GPU object, the contentHash will be stale if the patch changes!
						// there's no other patch to try and found GPU object, but at least we were able to merge this oen
						return record.instance;
					}
					// proceed onto the no compatible patch and no compatible GPU object found case
				}

				// Either haven't managed to combine with anything, or no GPU object found
				record.instance.patchIndex = patchStorage.size();
				// push a new patch
				patchStorage.push_back(std::move(patch));
				// Only when we cannot find a cached GPU object or a compatible patch entry do we carry on with the DFS
				if (asset_traits<AssetType>::HasChildren)
					dfsStack.emplace(record);
				// emplace without a valid GPU object, this will not change unless the cache entry's
				// patch gets combined and that is findable in the gpu conversion cache
				dfsCache.emplace(instance_t<AssetType>{.asset=asset,.meta=record.instance},NoContentHash,asset_cached_t<AssetType>{});
				return record.instance;
			};
			// Need to look at ENABLED features and not Physical Device's AVAILABLE features.
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

			// Perform Depth First Search of the Asset Graph
			while (!dfsStack.empty())
			{
				const auto entry = dfsStack.top();
				dfsStack.pop();
				// everything we popped has already been cached in dfsCache, now time to go over dependents
				const auto* user = entry.asset;
				switch (user->getAssetType())
				{
					case ICPUDescriptorSetLayout::AssetType:
					{
						auto layout = static_cast<const ICPUDescriptorSetLayout*>(user);
						for (const auto& sampler : layout->getImmutableSamplers())
						if (sampler)
							cache.operator()<ICPUSampler>(entry,sampler.get(),{sampler.get(),features,limits});
						break;
					}
					case ICPUPipelineLayout::AssetType:
					{
						auto pplnLayout = static_cast<const ICPUPipelineLayout*>(user);
						for (auto i=0; i<ICPUPipelineLayout::DESCRIPTOR_SET_COUNT; i++)
						if (auto layout=pplnLayout->getDescriptorSetLayout(i); layout)
							cache.operator()<ICPUDescriptorSetLayout>(entry,layout,{layout,features,limits});
						break;
					}
					case ICPUComputePipeline::AssetType:
					{
						auto compPpln = static_cast<const ICPUComputePipeline*>(user);
						const auto* layout = compPpln->getLayout();
						cache.operator()<ICPUPipelineLayout>(entry,layout,{layout,features,limits});
						const auto* shader = compPpln->getSpecInfo().shader;
						patch_t<ICPUShader> patch = {shader,features,limits};
						patch.stage = IGPUShader::E_SHADER_STAGE::ESS_COMPUTE;
						cache.operator()<ICPUShader>(entry,shader,std::move(patch));
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
							patch.usage |= IGPUBuffer::EUF_STORAGE_TEXEL_BUFFER_BIT|IGPUBuffer::EUF_UNIFORM_TEXEL_BUFFER_BIT;
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
		//! `inputsMetadata` is now constant!
		//! `finalPatchStorage` is now constant!

		// can now spawn our own hash cache if one wasn't provided
		core::smart_refctd_ptr<CHashCache> hashCache;
		if (inputs.hashCache)
			hashCache = core::smart_refctd_ptr<CHashCache>(inputs.hashCache);
		else
			hashCache = core::make_smart_refctd_ptr<CHashCache>();

		// Deduplication, Creation and Propagation
		auto dedupCreateProp = [&]<Asset AssetType>()->void
		{
			auto& dfsCache = std::get<dfs_cache_t<AssetType>>(dfsCaches);
			const auto* pPatches = std::get<patch_vector_t<AssetType>>(finalPatchStorage).data();
			// This map contains the assets by-hash, identical asset+patch hash the same.
			conversions_t<AssetType> conversionRequests;

			// We now go through the dfsCache and work out each entry's content hashes, so that we can carry out unique conversions.
			const CCache<AssetType>* readCache = inputs.readCache ? (&std::get<CCache<AssetType>>(inputs.readCache->m_caches)):nullptr;
			for (auto& entry : dfsCache)
			if (!entry.gpuObj)
			{
				const auto& canonical = entry.canonical;
				if (!canonical.asset)
					continue;
				const auto patchIx = canonical.meta.patchIndex;
				// compute the hash or look it up if it exists
				entry.contentHash = hashCache->hash<AssetType>({
					{.asset=canonical.asset,.patch=pPatches+patchIx},
					// We mistrust every dependency such that the eject/update if needed.
					// Its really important that the Deduplication gets performed Bottom-Up
					/*.mistrustLevel = */1
				});
				// if we have a read cache, lets retry looking the item up!
				if (readCache)
				{
					// We can't look up "near misses" (supersets of patches) because they'd have different hashes
					// and we can't afford to split hairs like finding overlapping buffer ranges, etc.
					// Stuff like that would require a completely different hashing/lookup strategy (or multiple fake entries).
					const auto found = readCache->find({entry.contentHash,canonical.meta.uniqueCopyGroupID});
					if (found)
					{
						entry.gpuObj = *found;
						// no conversion needed
						continue;
					}
				}
				// The conversion request we insert needs a canonical asset without missing contents
				if (IPreHashed::anyDependantDiscardedContents(canonical.asset))
					continue;
				// then de-duplicate the conversions needed
				auto [inSetIt,inserted] = conversionRequests.emplace(entry.contentHash,unique_conversion_t<AssetType>{.canonicalAsset=canonical.asset,.patchIndex=patchIx});
				if (!inserted)
				{
					// If an element prevented insertion, the patch must be identical!
					// Because the conversions don't care about groupIDs, the patches may be identical but not the same object in memory.
					assert(inSetIt->second.patchIndex==patchIx || pPatches[inSetIt->second.patchIndex]==pPatches[patchIx]);
					inSetIt->second.copyCount++;
				}
			}
			
			// work out mapping of `conversionRequests` to multiple GPU objects and their copy groups via counting sort
			auto exclScanConvReqs = [&]()->size_t
			{
				size_t sum = 0;
				for (auto& entry : conversionRequests)
				{
					entry.second.firstCopyIx = sum;
					sum += entry.second.copyCount;
				}
				return sum;
			};
			const auto gpuObjUniqueCopyGroupIDs = [&]()->core::vector<size_t>
			{
				core::vector<size_t> retval;
				// now assign storage offsets via exclusive scan and put the `uniqueGroupID` mappings in sorted order
				retval.resize(exclScanConvReqs());
				for (auto& entry : dfsCache)
				if (entry.canonical.asset && !entry.gpuObj)
				{
					auto found = conversionRequests.find(entry.contentHash);
					// may not find things because of unconverted dummy deps
					if (found!=conversionRequests.end())
						retval[found->second.firstCopyIx++] = entry.canonical.meta.uniqueCopyGroupID;
				}
				// `{conversionRequests}.firstCopyIx` needs to be brought back down to exclusive scan form
				exclScanConvReqs();
				return retval;
			}();
			core::vector<asset_cached_t<AssetType>> gpuObjects(gpuObjUniqueCopyGroupIDs.size());
			
			// small utility
			auto getDependant = [&]<Asset DepAssetType, typename Pred>(const size_t usersCopyGroupID, const AssetType* user, const DepAssetType* depAsset, Pred pred, bool& failed)->asset_cached_t<DepAssetType>::type
			{
				if (!depAsset)
					return {};
				const auto candidates = std::get<dfs_cache_t<DepAssetType>>(dfsCaches).equal_range(
					dfs_entry_t
					{
						.asset = depAsset,
						.instance = {
							.uniqueCopyGroupID = inputs.getDependantUniqueCopyGroupID(usersCopyGroupID,user,depAsset)
						}
					}
				);
				const auto chosen = std::find_if(candidates.first,candidates.second,pred);
				if (chosen!=candidates.second)
					return chosen->gpuObj.value;
				failed = true;
				return {};
			};
			// for all the asset types which don't have a patch possible, or its irrelavant for the user asset
			auto firstPatchMatch = [](const auto& candidate)->bool{return true;};

			// Only warn once to reduce log spam
			auto assign = [&]<bool GPUObjectWhollyImmutable=false>(const core::blake3_hash_t& contentHash, const size_t baseIx, const size_t copyIx, asset_cached_t<AssetType>::type&& gpuObj)->bool
			{
				const auto hashAsU64 = reinterpret_cast<const uint64_t*>(contentHash.data);
				if constexpr (GPUObjectWhollyImmutable) // including any deps!
				if (copyIx==1)
					inputs.logger.log(
						"Why are you creating multiple Objects for asset content %8llx%8llx%8llx%8llx, when they are a readonly GPU Object Type with no dependants!?",
						system::ILogger::ELL_PERFORMANCE,hashAsU64[0],hashAsU64[1],hashAsU64[2],hashAsU64[3]
					);
				//
				if (!gpuObj)
				{
					inputs.logger.log(
						"Failed to create GPU Object for asset content %8llx%8llx%8llx%8llx",
						system::ILogger::ELL_ERROR,hashAsU64[0],hashAsU64[1],hashAsU64[2],hashAsU64[3]
					);
					return false;
				}
				gpuObjects[copyIx+baseIx].value = std::move(gpuObj);
				return true;
			};
			
			// Dispatch to correct creation of GPU objects
			if constexpr (std::is_same_v<AssetType,ICPUSampler>)
			{
				for (auto& entry : conversionRequests)
				for (auto i=0ull; i<entry.second.copyCount; i++)
					assign.operator()<true>(entry.first,entry.second.firstCopyIx,i,device->createSampler(entry.second.canonicalAsset->getParams()));
			}
			if constexpr (std::is_same_v<AssetType,ICPUBuffer>)
			{
				for (auto& entry : conversionRequests)
				for (auto i=0ull; i<entry.second.copyCount; i++)
				{
					IGPUBuffer::SCreationParams params = {};
					params.size = entry.second.canonicalAsset->getSize();
					params.usage = pPatches[entry.second.patchIndex].usage;
					// TODO: make this configurable
					params.queueFamilyIndexCount = 0;
					params.queueFamilyIndices = nullptr;
					// if creation successful, we 
					if (assign(entry.first,entry.second.firstCopyIx,i,device->createBuffer(std::move(params))))
						retval.m_queueFlags |= IQueue::FAMILY_FLAGS::TRANSFER_BIT;
				}
			}
			if constexpr (std::is_same_v<AssetType,ICPUShader>)
			{
				ILogicalDevice::SShaderCreationParameters createParams = {
					.optimizer = m_params.optimizer.get(),
					.readCache = inputs.readShaderCache,
					.writeCache = inputs.writeShaderCache
				};
				for (auto& entry : conversionRequests)
				for (auto i=0ull; i<entry.second.copyCount; i++)
				{
					createParams.cpushader = entry.second.canonicalAsset;
					assign(entry.first,entry.second.firstCopyIx,i,device->createShader(createParams));
				}
			}
			if constexpr (std::is_same_v<AssetType,ICPUDescriptorSetLayout>)
			{
				for (auto& entry : conversionRequests)
				{
					const ICPUDescriptorSetLayout* asset = entry.second.canonicalAsset;
					// there is no patching possible for this asset
					using storage_range_index_t = ICPUDescriptorSetLayout::CBindingRedirect::storage_range_index_t;
					// rebuild bindings from CPU info
					core::vector<IGPUDescriptorSetLayout::SBinding> bindings;
					bindings.reserve(asset->getTotalBindingCount());
					for (uint32_t t=0u; t<static_cast<uint32_t>(IDescriptor::E_TYPE::ET_COUNT); t++)
					{
						const auto type = static_cast<IDescriptor::E_TYPE>(t);
						const auto& redirect = asset->getDescriptorRedirect(type);
						const auto count = redirect.getBindingCount();
						for (auto i=0u; i<count; i++)
						{
							const storage_range_index_t storageRangeIx(i);
							const auto binding = redirect.getBinding(storageRangeIx);
							bindings.push_back(IGPUDescriptorSetLayout::SBinding{
								.binding = binding.data,
								.type = type,
								.createFlags = redirect.getCreateFlags(storageRangeIx),
								.stageFlags = redirect.getStageFlags(storageRangeIx),
								.count = redirect.getCount(storageRangeIx),
								.immutableSamplers = nullptr
							});
						}
					}
					// get the immutable sampler info
					const auto samplerAssets = asset->getImmutableSamplers();
					core::vector<core::smart_refctd_ptr<IGPUSampler>> immutableSamplers(samplerAssets.size());
					// to let us know what binding has immutables
					const auto& immutableSamplerRedirects = asset->getImmutableSamplerRedirect();
					for (auto i=0ull; i<entry.second.copyCount; i++)
					{
						const auto outIx = i+entry.second.firstCopyIx;
						const auto uniqueCopyGroupID = gpuObjUniqueCopyGroupIDs[outIx];
						// go over the immutables, can't be factored out because depending on groupID the dependant might change
						bool notAllDepsFound = false;
						{
							const auto count = immutableSamplerRedirects.getBindingCount();
							auto outImmutableSamplers = immutableSamplers.data();
							for (auto j=0u; j<count; j++)
							{
								const storage_range_index_t storageRangeIx(j);
								// assuming the asset was validly created, the binding must exist
								const auto binding = immutableSamplerRedirects.getBinding(storageRangeIx);
								auto inSamplerAsset = samplerAssets.data()+immutableSamplerRedirects.getStorageOffset(storageRangeIx).data;
								// TODO: optimize this, the `bindings` are sorted within a given type
								auto outBinding = std::find_if(bindings.begin(),bindings.end(),[=](const IGPUDescriptorSetLayout::SBinding& item)->bool{return item.binding==binding.data;});
								// the binding must be findable, otherwise above code logic is wrong
								assert(outBinding!=bindings.end());
								outBinding->immutableSamplers = outImmutableSamplers;
								//
								const auto end = outImmutableSamplers+outBinding->count;
								while (outImmutableSamplers!=end)
								{
									// make the first sampler found match, there shouldn't be multiple copies anyway (warning will be logged)
									auto found = getDependant(uniqueCopyGroupID,asset,(inSamplerAsset++)->get(),firstPatchMatch,notAllDepsFound);
									// if we cannot find a dep, we fail whole gpu object creation
									if (notAllDepsFound)
										break;
									*(outImmutableSamplers++) = std::move(found);
								}
								// early out
								if (notAllDepsFound)
									break;
							}
							if (notAllDepsFound)
								continue;
						}
						assign(entry.first,entry.second.firstCopyIx,i,device->createDescriptorSetLayout(bindings));
					}
				}
			}
			if constexpr (std::is_same_v<AssetType,ICPUPipelineLayout>)
			{
				core::vector<asset::SPushConstantRange> pcRanges;
				pcRanges.reserve(CSPIRVIntrospector::MaxPushConstantsSize);
				for (auto& entry : conversionRequests)
				{
					const ICPUPipelineLayout* asset = entry.second.canonicalAsset;
					const auto& patch = std::get<patch_vector_t<ICPUPipelineLayout>>(finalPatchStorage)[entry.second.patchIndex];
					// time for some RLE
					{
						pcRanges.resize(0);
						asset::SPushConstantRange prev = {
							.stageFlags = IGPUShader::ESS_UNKNOWN,
							.offset = 0,
							.size = 0
						};
						for (auto byte=0u; byte<patch.pushConstantBytes.size(); byte++)
						{
							const auto current = patch.pushConstantBytes[byte].value;
							if (current!=prev.stageFlags)
							{
								if (prev.stageFlags)
								{
									prev.size = byte-prev.offset;
									pcRanges.push_back(prev);
								}
								prev.stageFlags = current;
								prev.offset = byte;
							}
						}
						if (prev.stageFlags)
						{
							prev.size = CSPIRVIntrospector::MaxPushConstantsSize-prev.offset;
							pcRanges.push_back(prev);
						}
					}
					for (auto i=0ull; i<entry.second.copyCount; i++)
					{
						const auto outIx = i+entry.second.firstCopyIx;
						const auto uniqueCopyGroupID = gpuObjUniqueCopyGroupIDs[outIx];
						asset_cached_t<ICPUDescriptorSetLayout>::type dsLayouts[4];
						{
							bool notAllDepsFound = false;
							for (auto j=0u; j<4; j++)
								dsLayouts[j] = getDependant(uniqueCopyGroupID,asset,asset->getDescriptorSetLayout(j),firstPatchMatch,notAllDepsFound);
							if (notAllDepsFound)
								continue;
						}
						assign(entry.first,entry.second.firstCopyIx,i,device->createPipelineLayout(pcRanges,std::move(dsLayouts[0]),std::move(dsLayouts[1]),std::move(dsLayouts[2]),std::move(dsLayouts[3])));
					}
				}
			}
			if constexpr (std::is_same_v<AssetType,ICPUPipelineCache>)
			{
				for (auto& entry : conversionRequests)
				{
					const ICPUPipelineCache* asset = entry.second.canonicalAsset;
					// there is no patching possible for this asset
					for (auto i=0ull; i<entry.second.copyCount; i++)
					{
						// since we don't have dependants we don't care about our group ID
						// we create threadsafe pipeline caches, because we have no idea how they may be used
						assign.operator()<true>(entry.first,entry.second.firstCopyIx,i,device->createPipelineCache(asset,false));
					}
				}
			}
			if constexpr (std::is_same_v<AssetType,ICPUComputePipeline>)
			{
				for (auto& entry : conversionRequests)
				{
					const ICPUComputePipeline* asset = entry.second.canonicalAsset;
					const auto& assetSpecShader = asset->getSpecInfo();
					const IGPUShader::SSpecInfo specShaderWithoutDep = {
						.entryPoint = assetSpecShader.entryPoint,
						.shader = nullptr, // to get later in the loop
						.entries = assetSpecShader.entries,
						.requiredSubgroupSize = assetSpecShader.requiredSubgroupSize,
						.requireFullSubgroups = assetSpecShader.requireFullSubgroups
					};
					// there is no patching possible for this asset
					for (auto i=0ull; i<entry.second.copyCount; i++)
					{
						const auto outIx = i+entry.second.firstCopyIx;
						const auto uniqueCopyGroupID = gpuObjUniqueCopyGroupIDs[outIx];
						// ILogicalDevice::createComputePipelines is rather aggressive on the spec constant validation, so we create one pipeline at a time
						{
							// no derivatives, special flags, etc.
							IGPUComputePipeline::SCreationParams params = {};
							params.shader = specShaderWithoutDep;
							bool depNotFound = false;
							{
								// we choose whatever patch, because there should really only ever be one (all pipeline layouts merge their PC ranges seamlessly)
								params.layout = getDependant(uniqueCopyGroupID,asset,asset->getLayout(),firstPatchMatch,depNotFound).get();
								// while there are patches possible for shaders, the only patch which can happen here is changing a stage from UNKNOWN to COMPUTE
								params.shader.shader = getDependant(uniqueCopyGroupID,asset,assetSpecShader.shader,firstPatchMatch,depNotFound).get();
							}
							if (depNotFound)
								continue;
							core::smart_refctd_ptr<IGPUComputePipeline> ppln;
							device->createComputePipelines(inputs.pipelineCache,{&params,1},&ppln);
							assign(entry.first,entry.second.firstCopyIx,i,std::move(ppln));
						}
					}
				}
			}

			// Propagate the results back, since the dfsCache has the original asset pointers as keys, we map in reverse
			auto& stagingCache = std::get<CCache<AssetType>>(retval.m_stagingCaches);
			for (auto& entry : dfsCache)
			if (entry.canonical.asset && !entry.gpuObj)
			{
				auto found = conversionRequests.find(entry.contentHash);
				const auto uniqueCopyGroupID = entry.canonical.meta.uniqueCopyGroupID;

				// can happen if deps were unconverted dummies
				if (found==conversionRequests.end())
				{
					const auto hashAsU64 = reinterpret_cast<const uint64_t*>(entry.contentHash.data);
					inputs.logger.log(
						"Could not find GPU Object for Asset %p in group %ull with Content Hash %8llx%8llx%8llx%8llx",
						system::ILogger::ELL_ERROR,entry.canonical.asset,uniqueCopyGroupID,hashAsU64[0],hashAsU64[1],hashAsU64[2],hashAsU64[3]
					);
					continue;
				}

				const auto copyIx = found->second.firstCopyIx++;
				// the counting sort was stable
				assert(uniqueCopyGroupID==gpuObjUniqueCopyGroupIDs[copyIx]);

				auto& gpuObj = gpuObjects[copyIx];
				// set debug names on everything!
				{
					std::ostringstream debugName;
					debugName << "Created by Converter ";
					debugName << std::hex;
					debugName << this;
					debugName << " from Asset with hash ";
					for (const auto& byte : entry.contentHash.data)
						debugName << uint32_t(byte) << " ";
					debugName << "for Group " << uniqueCopyGroupID;
					gpuObj.get()->setObjectDebugName(debugName.str().c_str());
				}
				// insert into staging cache
				stagingCache.insert({entry.contentHash,uniqueCopyGroupID},gpuObj);
				// propagate back to dfsCache
				entry.gpuObj = std::move(gpuObj);
			}
		};
		// The order of these calls is super important to go BOTTOM UP in terms of hashing and conversion dependants.
		// Both so we can hash in O(Depth) and not O(Depth^2) but also so we have all the possible dependants ready.
		// If two Asset chains are independent then we order them from most catastrophic failure to least.
		dedupCreateProp.operator()<ICPUBuffer>();
//		dedupCreateProp.operator()<ICPUImage>();
		// Allocate Memory
		{
			// gather memory reqs
			// allocate device memory
		//	device->allocate(reqs,false);
			// now bind it
			// if fail, need to wipe the GPU Obj as a failure
		}
//		dedupCreateProp.operator()<ICPUBottomLevelAccelerationStructure>();
//		dedupCreateProp.operator()<ICPUTopLevelAccelerationStructure>();
//		dedupCreateProp.operator()<ICPUBufferView>();
		dedupCreateProp.operator()<ICPUShader>();
		dedupCreateProp.operator()<ICPUSampler>();
		dedupCreateProp.operator()<ICPUDescriptorSetLayout>();
		dedupCreateProp.operator()<ICPUPipelineLayout>();
		dedupCreateProp.operator()<ICPUPipelineCache>();
		dedupCreateProp.operator()<ICPUComputePipeline>();
//		dedupCreateProp.operator()<ICPURenderpass>();
//		dedupCreateProp.operator()<ICPUGraphicsPipeline>();
//		dedupCreateProp.operator()<ICPUDescriptorSet>();
//		dedupCreateProp.operator()<ICPUFramebuffer>();


// TODO:
// how to get dependant while converting?
	// - (A*,G) of dependant easily supplied, the group is done by calling `inputs` again
	// - how to find patch? first compatible!
		// + compatible with what? derived patch? asset usage?
			// * Maybe lets store original patch in YAC?
	}

	// write out results
	auto finalize = [&]<typename AssetType>(const std::span<const AssetType* const> assets)->void
	{
		const auto count = assets.size();
		//
		const auto& metadata = inputsMetadata[index_of_v<AssetType,supported_asset_types>];
		const auto& dfsCache = std::get<dfs_cache_t<AssetType>>(dfsCaches);
		const auto& stagingCache = std::get<CCache<AssetType>>(retval.m_stagingCaches);
		auto& results = std::get<SResults::vector_t<AssetType>>(retval.m_gpuObjects);
		//
		const auto patchCount = std::get<patch_vector_t<AssetType>>(finalPatchStorage).size();
		for (size_t i=0; i<count; i++)
		if (auto asset=assets[i]; asset && metadata[i].patchIndex<patchCount)
		{
			// The Content Hash and GPU object are in the dfsCache
			auto range = dfsCache.equal_range(dfs_entry_t{.asset=asset,.instance=metadata[i]});
			// we'll find the correct patch from metadata
			auto found = std::find_if(range.first,range.second,[&](const auto& entry)->bool{return entry.canonical.meta.patchIndex==metadata[i].patchIndex;});
			// unless ofc the patch was invalid
			const auto uniqueCopyGroupID = metadata[i].uniqueCopyGroupID;
			if (found==range.second)
			{
				inputs.logger.log("No valid patch could be created for Asset %p in group %d",system::ILogger::ELL_ERROR,asset,uniqueCopyGroupID);
				continue;
			}
			// write it out to the results
			if (found->gpuObj) // found from the `input.readCache`
			{
				results[i] = found->gpuObj;
				// if something with this content hash is in the stagingCache, then it must match the `found->gpuObj`
				if (auto pGpuObj=stagingCache.find({found->contentHash,metadata[i].uniqueCopyGroupID}); pGpuObj)
				{
					assert(pGpuObj->get()==found->gpuObj.get());
				}
			}
			else
				inputs.logger.log("No valid patch could be created for Asset %p in group %d",system::ILogger::ELL_ERROR,asset,uniqueCopyGroupID);
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
}