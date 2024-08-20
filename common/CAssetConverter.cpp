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
// No asset has a 0 length input to the hash function
const core::blake3_hash_t CAssetConverter::CHashCache::NoContentHash = static_cast<core::blake3_hash_t>(core::blake3_hasher());

CAssetConverter::patch_impl_t<ICPUSampler>::patch_impl_t(const ICPUSampler* sampler) : anisotropyLevelLog2(sampler->getParams().AnisotropicFilter) {}
bool CAssetConverter::patch_impl_t<ICPUSampler>::valid(const SPhysicalDeviceFeatures& features, const SPhysicalDeviceLimits& limits)
{
	if (anisotropyLevelLog2>5) // unititialized
		return false;
	if (anisotropyLevelLog2>limits.maxSamplerAnisotropyLog2)
		anisotropyLevelLog2 = limits.maxSamplerAnisotropyLog2;
	return true;
}

CAssetConverter::patch_impl_t<ICPUShader>::patch_impl_t(const ICPUShader* shader) : stage(shader->getStage()) {}
bool CAssetConverter::patch_impl_t<ICPUShader>::valid(const SPhysicalDeviceFeatures& features, const SPhysicalDeviceLimits& limits)
{
	switch (stage)
	{
		// supported always
		case IGPUShader::E_SHADER_STAGE::ESS_VERTEX:
		case IGPUShader::E_SHADER_STAGE::ESS_FRAGMENT:
		case IGPUShader::E_SHADER_STAGE::ESS_COMPUTE:
			return true;
			break;
		case IGPUShader::E_SHADER_STAGE::ESS_TESSELLATION_CONTROL:
		case IGPUShader::E_SHADER_STAGE::ESS_TESSELLATION_EVALUATION:
			if (features.tessellationShader)
				return true;
			break;
		case IGPUShader::E_SHADER_STAGE::ESS_GEOMETRY:
			if (features.geometryShader)
				return true;
			break;
		case IGPUShader::E_SHADER_STAGE::ESS_TASK:
//			if (features.taskShader)
//				return true;
			break;
		case IGPUShader::E_SHADER_STAGE::ESS_MESH:
//			if (features.meshShader)
//				return true;
			break;
		case IGPUShader::E_SHADER_STAGE::ESS_RAYGEN:
		case IGPUShader::E_SHADER_STAGE::ESS_ANY_HIT:
		case IGPUShader::E_SHADER_STAGE::ESS_CLOSEST_HIT:
		case IGPUShader::E_SHADER_STAGE::ESS_MISS:
		case IGPUShader::E_SHADER_STAGE::ESS_INTERSECTION:
		case IGPUShader::E_SHADER_STAGE::ESS_CALLABLE:
			if (features.rayTracingPipeline)
				return true;
			break;
		default:
			break;
	}
	return false;
}

CAssetConverter::patch_impl_t<ICPUBuffer>::patch_impl_t(const ICPUBuffer* buffer) : usage(buffer->getUsageFlags()) {}
bool CAssetConverter::patch_impl_t<ICPUBuffer>::valid(const SPhysicalDeviceFeatures& features, const SPhysicalDeviceLimits& limits)
{
	if (usage.hasFlags(usage_flags_t::EUF_CONDITIONAL_RENDERING_BIT_EXT) && !features.conditionalRendering)
		return false;
	if ((usage.hasFlags(usage_flags_t::EUF_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT)||usage.hasFlags(usage_flags_t::EUF_ACCELERATION_STRUCTURE_STORAGE_BIT)) && !features.accelerationStructure)
		return false;
	if (usage.hasFlags(usage_flags_t::EUF_SHADER_BINDING_TABLE_BIT) && !features.rayTracingPipeline)
		return false;
	// good default
	usage |= usage_flags_t::EUF_INLINE_UPDATE_VIA_CMDBUF;
	return true;
}

CAssetConverter::patch_impl_t<ICPUPipelineLayout>::patch_impl_t(const ICPUPipelineLayout* pplnLayout) : patch_impl_t()
{
	const auto pc = pplnLayout->getPushConstantRanges();
	for (auto it=pc.begin(); it!=pc.end(); it++)
	if (it->stageFlags!=shader_stage_t::ESS_UNKNOWN)
	{
		if (it->offset>=pushConstantBytes.size())
			return;
		const auto end = it->offset+it->size;
		if (end<it->offset || end>pushConstantBytes.size())
			return;
		for (auto byte=it->offset; byte<end; byte++)
			pushConstantBytes[byte] = it->stageFlags;
	}
	invalid = false;
}
bool CAssetConverter::patch_impl_t<ICPUPipelineLayout>::valid(const SPhysicalDeviceFeatures& features, const SPhysicalDeviceLimits& limits)
{
	for (auto byte=limits.maxPushConstantsSize; byte<pushConstantBytes.size(); byte++)
	if (pushConstantBytes[byte]!=shader_stage_t::ESS_UNKNOWN)
		return false;
	return true;
}


// question of propagating changes, image view and buffer view
// if image view used as STORAGE, or SAMPLED have to change subUsage
// then need to propagate change of subUsage to usage
// therefore image is now changed
// have to pass the patch around



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
struct patch_index_t
{
	inline bool operator==(const patch_index_t&) const = default;
	inline bool operator!=(const patch_index_t&) const = default;

	explicit inline operator bool() const {return operator!=({});}

	uint64_t value = 0xdeadbeefBADC0FFEull;
};

//
struct input_metadata_t
{
	inline bool operator==(const input_metadata_t&) const = default;
	inline bool operator!=(const input_metadata_t&) const = default;

	explicit inline operator bool() const {return operator!=({});}

	size_t uniqueCopyGroupID = 0xdeadbeefBADC0FFEull;
	patch_index_t patchIndex = {};
};

//
template<Asset AssetType>
struct instance_t
{
	inline operator instance_t<IAsset>() const
	{
		return instance_t<IAsset>{.asset=asset,.uniqueCopyGroupID=uniqueCopyGroupID};
	}

	inline bool operator==(const instance_t<AssetType>&) const = default;

	//
	const AssetType* asset = nullptr;
	size_t uniqueCopyGroupID = 0xdeadbeefBADC0FFEull;
};

// This cache stops us traversing an asset with the same user group and patch more than once.
template<asset::Asset AssetType>
struct dfs_cache
{
	// Maps `instance_t` to `patchIndex`, makes sure the find can handle polymorphism of assets
	struct HashEquals
	{
		using is_transparent = void;

		inline size_t operator()(const instance_t<IAsset>& entry) const
		{
			return ptrdiff_t(entry.asset)^entry.uniqueCopyGroupID;
		}

		inline size_t operator()(const instance_t<AssetType>& entry) const
		{
			// its very important to cast the derived AssetType to IAsset because otherwise pointers won't match
			return operator()(instance_t<IAsset>(entry));
		}
	
		inline bool operator()(const instance_t<IAsset>& lhs, const instance_t<AssetType>& rhs) const
		{
			return lhs==instance_t<IAsset>(rhs);
		}
		inline bool operator()(const instance_t<AssetType>& lhs, const instance_t<IAsset>& rhs) const
		{
			return instance_t<IAsset>(lhs)==rhs;
		}
		inline bool operator()(const instance_t<AssetType>& lhs, const instance_t<AssetType>& rhs) const
		{
			return instance_t<IAsset>(lhs)==instance_t<IAsset>(rhs);
		}
	};
	using key_map_t = core::unordered_map<instance_t<AssetType>,patch_index_t,HashEquals,HashEquals>;

	// Find the first node for an instance with a compatible patch
	// For performance reasons you may want to defer the patch construction/deduction till after you know a matching instance exists
	template<typename DeferredPatchGet>
	inline std::pair<typename key_map_t::const_iterator,patch_index_t> find(const instance_t<AssetType>& instance, DeferredPatchGet patchGet) const
	{
		// get all the existing patches for the same (AssetType*,UniqueCopyGroupID)
		auto found = instances.find(instance);
		if (found!=instances.end())
		{
			const auto& requiredSubset = patchGet();
			// we don't want to pass the device features and limits to this function, it just assumes the patch will be valid without touch-ups
			//assert(requiredSubset.valid(deviceFeatures,deviceLimits));
			auto createdIndex = found->second;
			while (createdIndex)
			{
				const auto& candidate = nodes[createdIndex.value];
				if (std::get<bool>(candidate.patch.combine(requiredSubset)))
					break;
				createdIndex = candidate.next;
			}
			return {found,createdIndex};
		}
		return {found,{}};
	}

	template<typename What>
	inline void for_each(What what)
	{
		for (auto& entry : instances)
		{
			const auto& instance = entry.first;
			auto patchIx = entry.second;
			assert(instance.asset || !patchIx);
			for (; patchIx; patchIx=nodes[patchIx.value].next)
				what(instance,nodes[patchIx.value]);
		}
	}

	// not a multi-map anymore because order of insertion into an equal range needs to be stable, so I just make it a linked list explicitly
	key_map_t instances;
	// node struct
	struct created_t
	{
		CAssetConverter::patch_t<AssetType> patch = {};
		core::blake3_hash_t contentHash = {};
		asset_cached_t<AssetType> gpuObj = {};
		patch_index_t next = {};
	};
	// all entries refer to patch by index, so its stable against vector growth
	core::vector<created_t> nodes;
};

//
struct PatchGetter
{
	template<asset::Asset AssetType>
	static inline CAssetConverter::patch_t<AssetType> ConstructPatch(const IAsset* user, const AssetType* asset)
	{
		CAssetConverter::patch_t<AssetType> patch(asset);
		if (user) // only patch if there's a user
		switch (user->getAssetType())
		{
			case ICPUDescriptorSetLayout::AssetType:
			{
				const auto* typedUser = static_cast<const ICPUDescriptorSetLayout*>(user);
				if constexpr(std::is_same_v<AssetType,ICPUSampler>)
				{
					// nothing special to do
					return patch;
				}
				break;
			}
			case ICPUPipelineLayout::AssetType:
			{
				const auto* typedUser = static_cast<const ICPUPipelineLayout*>(user);
				if constexpr(std::is_same_v<AssetType,ICPUDescriptorSetLayout>)
				{
					// nothing special to do either
					return patch;
				}
				break;
			}
			case ICPUComputePipeline::AssetType:
			{
				const auto* typedUser = static_cast<const ICPUComputePipeline*>(user);
				if constexpr(std::is_same_v<AssetType,ICPUPipelineLayout>)
				{
					// nothing special to do
					return patch;
				}
				if constexpr(std::is_same_v<AssetType,ICPUShader>)
				{
					patch.stage = IGPUShader::E_SHADER_STAGE::ESS_COMPUTE;
					return patch;
				}
				break;
			}
			case ICPUBufferView::AssetType:
			{
				const auto* typedUser = static_cast<const ICPUBufferView*>(user);
				if constexpr(std::is_same_v<AssetType,ICPUBuffer>)
				{
					// we have no clue how this will be used, so we mark both usages
					patch.usage |= IGPUBuffer::EUF_STORAGE_TEXEL_BUFFER_BIT|IGPUBuffer::EUF_UNIFORM_TEXEL_BUFFER_BIT;
					return patch;
				}
				break;
			}
			default:
				break;
		}
		return {};
	}

	template<asset::Asset AssetType>
	inline patch_index_t impl(const AssetType* asset)
	{
		// don't advance `uniqueCopyGroupID` or `user` on purpose
		if (!asset)
			return {};
		uniqueCopyGroupID = p_inputs->getDependantUniqueCopyGroupID(uniqueCopyGroupID,user,asset);
		const auto& dfsCache = std::get<dfs_cache<AssetType>>(*p_dfsCaches);
		// Returning the first compatible patch is correct, as back when building the dfsCache you merge with the first compatible patch.
		// (assuming insertion order into the same bucket is stable)
		const auto found = dfsCache.find(
			{.asset=asset,.uniqueCopyGroupID=uniqueCopyGroupID},
			[&]()->CAssetConverter::patch_t<AssetType>{return ConstructPatch<AssetType>(user,asset);}
		);
		user = asset;
		if (const auto foundIx=std::get<patch_index_t>(found); foundIx)
			return foundIx;
		// some dependency is not in DFS cache - wasn't explored, probably because it was unpatchable/uncreatable
		return {};
	}
	
	template<asset::Asset AssetType>
	inline const CAssetConverter::patch_t<AssetType>* operator()(const AssetType* asset)
	{
		const patch_index_t foundIx = impl<AssetType>(asset);
		return foundIx ? &(std::get<dfs_cache<AssetType>>(*p_dfsCaches).nodes[foundIx.value].patch) : nullptr;
	}

	const SPhysicalDeviceFeatures* pFeatures;
	const SPhysicalDeviceLimits* pLimits;
	const CAssetConverter::SInputs* const p_inputs;
	const core::tuple_transform_t<dfs_cache,CAssetConverter::supported_asset_types>* const p_dfsCaches;
	// progressive state
	size_t uniqueCopyGroupID = 0xdeadbeefBADC0FFEull;
	const IAsset* user = nullptr;
};

//
template<asset::Asset AssetType>
struct unique_conversion_t
{
	const AssetType* canonicalAsset = nullptr;
	patch_index_t patchIndex = {};
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
	{
		inputs.logger.log("Read Cache's owning device %p not compatible with this cache's owning device %p.",system::ILogger::ELL_ERROR,inputs.readCache->m_params.device,m_params.device);
		return {};
	}
	if (inputs.pipelineCache && inputs.pipelineCache->getOriginDevice()!=device)
	{
		inputs.logger.log("Pipeline Cache's owning device %p not compatible with this cache's owning device %p.",system::ILogger::ELL_ERROR,inputs.pipelineCache->getOriginDevice(),m_params.device);
		return {};
	}

	SResults retval = {};
	
	// this will allow us to look up the conversion parameter (actual patch for an asset) and therefore write the GPUObject to the correct place in the return value
	core::vector<input_metadata_t> inputsMetadata[core::type_list_size_v<supported_asset_types>];
	// One would think that we first need an (AssetPtr,Patch) -> ContentHash map and then a ContentHash -> GPUObj map to
	// save ourselves iterating over redundant assets. The truth is that we going from a ContentHash to GPUObj is blazing fast.
	core::tuple_transform_t<dfs_cache,supported_asset_types> dfsCaches = {};

	{
		// Need to look at ENABLED features and not Physical Device's AVAILABLE features.
		const auto& features = device->getEnabledFeatures();
		const auto& limits = device->getPhysicalDevice()->getLimits();

		// gather all dependencies (DFS graph search) and patch, this happens top-down
		// do not deduplicate/merge assets at this stage, only patch GPU creation parameters
		{
			// stack is nice an polymorphic
			core::stack<instance_t<IAsset>> dfsStack;
			// returns `input_metadata_t` which you can `bool(input_metadata_t)` to find out if a new element was inserted
			auto cache = [&]<Asset AssetType>(const instance_t<IAsset>& user, const AssetType* asset, patch_t<AssetType>&& patch) -> input_metadata_t
			{
				assert(asset);
				// skip invalid inputs silently
				if (!patch.valid(features,limits))
					return {};
				//
				if constexpr (std::is_same_v<AssetType,ICPUShader>)
				if (asset->getContentType()==ICPUShader::E_CONTENT_TYPE::ECT_GLSL)
				{
					inputs.logger.log("Asset Converter doesn't support converting GLSL shaders! Asset %p won't be converted (GLSL is deprecated in Nabla)",system::ILogger::ELL_ERROR,asset);
					return {};
				}

				// get unique group
				instance_t<AssetType> record = {
					.asset = asset,
					.uniqueCopyGroupID = inputs.getDependantUniqueCopyGroupID(user.uniqueCopyGroupID,user.asset,asset)
				};

				// now see if we visited already
				auto& dfsCache = std::get<dfs_cache<AssetType>>(dfsCaches);
				//
				const patch_index_t newPatchIndex = {dfsCache.nodes.size()};
				// get all the existing patches for the same (AssetType*,UniqueCopyGroupID)
				auto found = dfsCache.instances.find(record);
				if (found!=dfsCache.instances.end())
				{
					// iterate over linked list
					patch_index_t* pIndex = &found->second;
					while (*pIndex)
					{
						// get our candidate
						auto& candidate = dfsCache.nodes[pIndex->value];
						// found a thing, try-combine the patches
						auto [success, combined] = candidate.patch.combine(patch);
						// check whether the item is creatable after patching
						if (success)
						{
							// change the patch to a combined version
							candidate.patch = std::move(combined);
							return { .uniqueCopyGroupID = record.uniqueCopyGroupID,.patchIndex=*pIndex};
						}
						// else try the next one
						pIndex = &candidate.next;
					}
					// nothing mergable found, make old TAIL point to new node about to be inserted
					*pIndex = newPatchIndex;
				}
				else
				{
					// there isn't even a linked list head for this entry
					dfsCache.instances.emplace_hint(found,record,newPatchIndex);
				}
				// Haven't managed to combine with anything, so push a new patch
				dfsCache.nodes.emplace_back(std::move(patch),CHashCache::NoContentHash);
				// Only when we don't find a compatible patch entry do we carry on with the DFS
				if (asset_traits<AssetType>::HasChildren)
					dfsStack.emplace(record);
				return {.uniqueCopyGroupID=record.uniqueCopyGroupID,.patchIndex=newPatchIndex};
			};
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
					patch_t<AssetType> patch(asset);
					if (i<patches.size())
					{
						if (!patch.valid(features,limits))
							continue;
						auto overidepatch = patches[i];
						if (!overidepatch.valid(features,limits))
							continue;
						bool combineSuccess;
						std::tie(combineSuccess,patch) = patch.combine(overidepatch);
						if (!combineSuccess)
							continue;
					}
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
							cache.operator()<ICPUSampler>(entry,sampler.get(),{sampler.get()});
						break;
					}
					case ICPUPipelineLayout::AssetType:
					{
						auto pplnLayout = static_cast<const ICPUPipelineLayout*>(user);
						for (auto i=0; i<ICPUPipelineLayout::DESCRIPTOR_SET_COUNT; i++)
						if (auto layout=pplnLayout->getDescriptorSetLayout(i); layout)
							cache.operator()<ICPUDescriptorSetLayout>(entry,layout,{layout});
						break;
					}
					case ICPUComputePipeline::AssetType:
					{
						auto compPpln = static_cast<const ICPUComputePipeline*>(user);
						const auto* layout = compPpln->getLayout();
						cache.operator()<ICPUPipelineLayout>(entry,layout,{layout});
						const auto* shader = compPpln->getSpecInfo().shader;
						patch_t<ICPUShader> patch = {shader};
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
							patch_t<ICPUBuffer> patch = {buffer};
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
		//! `dfsCache` keys are now constant!

		// can now spawn our own hash cache
		retval.m_hashCache = core::make_smart_refctd_ptr<CHashCache>();

		// Deduplication, Creation and Propagation
		auto dedupCreateProp = [&]<Asset AssetType>()->void
		{
			auto& dfsCache = std::get<dfs_cache<AssetType>>(dfsCaches);
			// This map contains the assets by-hash, identical asset+patch hash the same.
			conversions_t<AssetType> conversionRequests;

			// We now go through the dfsCache and work out each entry's content hashes, so that we can carry out unique conversions.
			const CCache<AssetType>* readCache = inputs.readCache ? (&std::get<CCache<AssetType>>(inputs.readCache->m_caches)):nullptr;
			dfsCache.for_each([&](const instance_t<AssetType>& instance, dfs_cache<AssetType>::created_t& created)->void
				{
					// compute the hash or look it up if it exists
					// We mistrust every dependency such that the eject/update if needed.
					// Its really important that the Deduplication gets performed Bottom-Up
					auto& contentHash = created.contentHash;
					contentHash = retval.getHashCache()->hash<AssetType>(
						instance.asset,
						PatchGetter{
							&features,
							&limits,
							&inputs,
							&dfsCaches
						},
						/*.mistrustLevel = */1
					);
					// failed to hash all together (only possible reason is failure of `PatchGetter` to provide a valid patch)
					if (contentHash==CHashCache::NoContentHash)
					{
						inputs.logger.log("Could not compute hash for asset %p in group %d, maybe an IPreHashed dependant's content hash is missing?",system::ILogger::ELL_ERROR,instance.asset,instance.uniqueCopyGroupID);
						return;
					}
					// if we have a read cache, lets retry looking the item up!
					if (readCache)
					{
						// We can't look up "near misses" (supersets of patches) because they'd have different hashes
						// and we can't afford to split hairs like finding overlapping buffer ranges, etc.
						// Stuff like that would require a completely different hashing/lookup strategy (or multiple fake entries).
						const auto found = readCache->find({contentHash,instance.uniqueCopyGroupID});
						if (found)
						{
							created.gpuObj = *found;
							// no conversion needed
							return;
						}
					}
					// The conversion request we insert needs an instance asset without missing contents
					if (IPreHashed::anyDependantDiscardedContents(instance.asset))
						return;
					// then de-duplicate the conversions needed
					const patch_index_t patchIx = {static_cast<uint64_t>(std::distance(dfsCache.nodes.data(),&created))};
					auto [inSetIt,inserted] = conversionRequests.emplace(contentHash,unique_conversion_t<AssetType>{.canonicalAsset=instance.asset,.patchIndex=patchIx});
					if (!inserted)
					{
						// If an element prevented insertion, the patch must be identical!
						// Because the conversions don't care about groupIDs, the patches may be identical but not the same object in memory.
						assert(inSetIt->second.patchIndex==patchIx || dfsCache.nodes[inSetIt->second.patchIndex.value].patch==dfsCache.nodes[patchIx.value].patch);
						inSetIt->second.copyCount++;
					}
				}
			);
			
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
				//
				dfsCache.for_each([&inputs,&retval,&conversionRequests](const instance_t<AssetType>& instance, dfs_cache<AssetType>::created_t& created)->void
					{
						if (!created.gpuObj)
							return;
						auto found = conversionRequests.find(created.contentHash);
						// may not find things because of unconverted dummy deps
						if (found!=conversionRequests.end())
							retval[found->second.firstCopyIx++] = instance.uniqueCopyGroupID;
						else
						{
							inputs.logger.log(
								"No conversion request made for Asset %p in group %d, its impossible to convert.",
								system::ILogger::ELL_ERROR,instance.asset,instance.uniqueCopyGroupID
							);
						}
					}
				);
				// `{conversionRequests}.firstCopyIx` needs to be brought back down to exclusive scan form
				exclScanConvReqs();
				return retval;
			}();
			core::vector<asset_cached_t<AssetType>> gpuObjects(gpuObjUniqueCopyGroupIDs.size());
			
			// small utility
			auto getDependant = [&]<Asset DepAssetType, typename Pred>(const size_t usersCopyGroupID, const AssetType* user, const DepAssetType* depAsset, Pred pred, bool& failed)->asset_cached_t<DepAssetType>::type
			{
				const patch_index_t found = PatchGetter{&features,&limits,&inputs,&dfsCaches,usersCopyGroupID,user}.impl<DepAssetType>(depAsset);
				if (found)
					return std::get<dfs_cache<DepAssetType>>(dfsCaches).nodes[found.value].gpuObj.value;
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
					params.usage = dfsCache.nodes[entry.second.patchIndex.value].patch.usage;
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
					const auto& patch = dfsCache.nodes[entry.second.patchIndex.value].patch;
					// time for some RLE
					{
						pcRanges.resize(0);
						asset::SPushConstantRange prev = {
							.stageFlags = IGPUShader::E_SHADER_STAGE::ESS_UNKNOWN,
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
			dfsCache.for_each([&](const instance_t<AssetType>& instance, dfs_cache<AssetType>::created_t& created)->void
				{
					// already found in read cache and not converted
					if (created.gpuObj)
						return;

					const auto& contentHash = created.contentHash;
					auto found = conversionRequests.find(contentHash);

					const auto uniqueCopyGroupID = instance.uniqueCopyGroupID;
					// can happen if deps were unconverted dummies
					if (found==conversionRequests.end())
					{
						const auto hashAsU64 = reinterpret_cast<const uint64_t*>(contentHash.data);
						if (contentHash!=CHashCache::NoContentHash)
							inputs.logger.log(
								"Could not find GPU Object for Asset %p in group %ull with Content Hash %8llx%8llx%8llx%8llx",
								system::ILogger::ELL_ERROR,instance.asset,uniqueCopyGroupID,hashAsU64[0],hashAsU64[1],hashAsU64[2],hashAsU64[3]
							);
						return;
					}
					// unhashables were not supposed to be added to conversion requests
					assert(contentHash!=CHashCache::NoContentHash);

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
						for (const auto& byte : contentHash.data)
							debugName << uint32_t(byte) << " ";
						debugName << "for Group " << uniqueCopyGroupID;
						gpuObj.get()->setObjectDebugName(debugName.str().c_str());
					}
					// insert into staging cache
					stagingCache.insert({contentHash,uniqueCopyGroupID},gpuObj);
					// propagate back to dfsCache
					created.gpuObj = std::move(gpuObj);
				}
			);
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
	}

	// write out results
	auto finalize = [&]<typename AssetType>(const std::span<const AssetType* const> assets)->void
	{
		const auto count = assets.size();
		//
		const auto& metadata = inputsMetadata[index_of_v<AssetType,supported_asset_types>];
		const auto& dfsCache = std::get<dfs_cache<AssetType>>(dfsCaches);
		const auto& stagingCache = std::get<CCache<AssetType>>(retval.m_stagingCaches);
		auto& results = std::get<SResults::vector_t<AssetType>>(retval.m_gpuObjects);
		for (size_t i=0; i<count; i++)
		if (auto asset=assets[i]; asset)
		{
			const auto uniqueCopyGroupID = metadata[i].uniqueCopyGroupID;
			// simple and easy to find all the associated items
			if (!metadata[i].patchIndex)
			{
				inputs.logger.log("No valid patch could be created for Asset %p in group %d",system::ILogger::ELL_ERROR,asset,uniqueCopyGroupID);
				continue;
			}
			const auto& found = dfsCache.nodes[metadata[i].patchIndex.value];
			// write it out to the results
			if (const auto& gpuObj=found.gpuObj; gpuObj) // found from the `input.readCache`
			{
				results[i] = gpuObj;
				// if something with this content hash is in the stagingCache, then it must match the `found->gpuObj`
				if (auto pGpuObj=stagingCache.find({found.contentHash,uniqueCopyGroupID}); pGpuObj)
				{
					assert(pGpuObj->get()==gpuObj.get());
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

	// insert items into cache if overflows handled fine and commandbuffers ready to be recorded
	// tag any failures
	// TODO: rescan all the GPU objects and find out if they depend on anything that failed, if so add them to the failure set

	return true;
}

ISemaphore::future_t<bool> CAssetConverter::SConvertParams::autoSubmit()
{
	// TODO: transfer first, then compute

	return {};
}

}
}