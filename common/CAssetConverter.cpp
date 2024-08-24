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

// nothing in the compute pipeline can be patched

// renderpass won't check for formats, resolve modes, sampler counts being supported because we don't patch them
// most we could patch would be sample counts, but nobody will really profit from this right now

// graphics pipeline can't really be patched just as a compute pipeline cannot be patched


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
			case ICPUGraphicsPipeline::AssetType:
			{
				const auto* typedUser = static_cast<const ICPUGraphicsPipeline*>(user);
				if constexpr (std::is_same_v<AssetType,ICPUPipelineLayout>)
				{
					// nothing special to do
					return patch;
				}
				if constexpr (std::is_same_v<AssetType,ICPUShader>)
				{
					using stage_t = ICPUShader::E_SHADER_STAGE;
					for (stage_t stage : {stage_t::ESS_VERTEX,stage_t::ESS_TESSELLATION_CONTROL,stage_t::ESS_TESSELLATION_EVALUATION,stage_t::ESS_GEOMETRY,stage_t::ESS_FRAGMENT})
					if (typedUser->getSpecInfo(stage).shader==asset)
					{
						patch.stage = stage;
						break;
					}
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
				return {};
		}
		return patch;
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
template<typename T, typename... U>
constexpr bool has_type(core::type_list<U...>) {return (std::is_same_v<T,U> || ...);}

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
				// special checks
				if constexpr (std::is_same_v<AssetType,ICPUShader>)
				if (asset->getContentType()==ICPUShader::E_CONTENT_TYPE::ECT_GLSL)
				{
					inputs.logger.log("Asset Converter doesn't support converting GLSL shaders! Asset %p won't be converted (GLSL is deprecated in Nabla)",system::ILogger::ELL_ERROR,asset);
					return {};
				}
				if constexpr (std::is_same_v<AssetType,ICPUBuffer>)
				if (asset->getSize()>limits.maxBufferSize)
				{
					inputs.logger.log("Requested buffer size %zu is larger than the Physical Device's maxBufferSize Limit! Asset %p won't be converted",system::ILogger::ELL_ERROR,asset->getSize(),asset);
					return {};
				}

				// get unique group
				instance_t<AssetType> record = {
					.asset = asset,
					.uniqueCopyGroupID = inputs.getDependantUniqueCopyGroupID(user.uniqueCopyGroupID,user.asset,asset)
				};
				inputs.logger.log("Asset (%p,%d) is used by (%p,%d)",system::ILogger::ELL_DEBUG,record.asset,record.uniqueCopyGroupID,user.asset,user.uniqueCopyGroupID);

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
					case ICPUGraphicsPipeline::AssetType:
					{
						auto gfxPpln = static_cast<const ICPUGraphicsPipeline*>(user);
						const auto* layout = gfxPpln->getLayout();
						cache.operator()<ICPUPipelineLayout>(entry,layout,{layout});
						const auto* rpass = gfxPpln->getRenderpass();
						cache.operator()<ICPURenderpass>(entry,rpass,{rpass});
						using stage_t = ICPUShader::E_SHADER_STAGE;
						for (stage_t stage : {stage_t::ESS_VERTEX,stage_t::ESS_TESSELLATION_CONTROL,stage_t::ESS_TESSELLATION_EVALUATION,stage_t::ESS_GEOMETRY,stage_t::ESS_FRAGMENT})
						{
							const auto* shader = gfxPpln->getSpecInfo(stage).shader;
							if (!shader)
								continue;
							patch_t<ICPUShader> patch = {shader};
							patch.stage = stage;
							cache.operator()<ICPUShader>(entry,shader,std::move(patch));
						}
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
		
		// a somewhat structured uint64_t
		struct MemoryRequirementBin
		{
			inline bool operator==(const MemoryRequirementBin&) const = default;

			// We order our requirement bins from those that can be allocated from the most memory types to those that can only be allocated from one
			inline bool operator<(const MemoryRequirementBin& other) const
			{
				if (needsDeviceAddress!=other.needsDeviceAddress)
					return needsDeviceAddress;
				return hlsl::bitCount(compatibileMemoryTypeBits)<hlsl::bitCount(other.compatibileMemoryTypeBits);
			}

			uint64_t compatibileMemoryTypeBits : 32 = 0;
			uint64_t needsDeviceAddress : 1 = 0;
		};
		// Because we store node pointer we can both get the `IDeviceMemoryBacked*` to bind to, and also zero out the cache entry if allocation unsuccessful
		using memory_backed_ptr_variant_t = std::variant<asset_cached_t<ICPUBuffer>*,asset_cached_t<ICPUImage>*>;
		core::map<MemoryRequirementBin,core::vector<memory_backed_ptr_variant_t>> allocationRequests;
		// for this we require that the data storage for the dfsCaches' nodes does not change
		auto requestAllocation = [&inputs,device,&allocationRequests]<DeviceMemoryBacked DeviceMemoryBackedType>(asset_cached_t<DeviceMemoryBackedType>* pGpuObj)->bool
		{
			const auto* gpuObj = pGpuObj->get();
			const IDeviceMemoryBacked::SDeviceMemoryRequirements& memReqs = gpuObj->getMemoryReqs();
			// this shouldn't be possible
			assert(memReqs.memoryTypeBits);
			// allocate right away those that need their own allocation
			if (memReqs.requiresDedicatedAllocation)
			{
				// allocate and bind right away
				auto allocation = device->allocate(memReqs,gpuObj);
				if (!allocation)
				{
					inputs.logger.log("Failed to allocate and bind dedicated memory for %s",system::ILogger::ELL_ERROR,gpuObj->getObjectDebugName());
					return false;
				}
			}
			else
			{
				// make the creation conditional upon allocation success
				MemoryRequirementBin reqBin = {
					.compatibileMemoryTypeBits = memReqs.memoryTypeBits,
					// we ignore this for now, because we can't know how many `DeviceMemory` objects we have left to make, so just join everything by default
					//.refersDedicatedAllocation = memReqs.prefersDedicatedAllocation
				};
				if constexpr (std::is_same_v<DeviceMemoryBackedType,IGPUBuffer>)
					reqBin.needsDeviceAddress = gpuObj->getCreationParams().usage.hasFlags(IGPUBuffer::E_USAGE_FLAGS::EUF_SHADER_DEVICE_ADDRESS_BIT);
				allocationRequests[reqBin].emplace_back(pGpuObj);
			}
			return true;
		};

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
					const auto hashAsU64 = reinterpret_cast<const uint64_t*>(contentHash.data);
					{
						inputs.logger.log("Asset (%p,%d) has hash %8llx%8llx%8llx%8llx",system::ILogger::ELL_DEBUG,instance.asset,instance.uniqueCopyGroupID,hashAsU64[0],hashAsU64[1],hashAsU64[2],hashAsU64[3]);
					}
					// if we have a read cache, lets retry looking the item up!
					if (readCache)
					{
						// We can't look up "near misses" (supersets of patches) because they'd have different hashes
						// and we can't afford to split hairs like finding overlapping buffer ranges, etc.
						// Stuff like that would require a completely different hashing/lookup strategy (or multiple fake entries).
						const auto found = readCache->find({contentHash,instance.uniqueCopyGroupID});
						if (found!=readCache->forwardMapEnd())
						{
							created.gpuObj = found->second;
							inputs.logger.log(
								"Asset (%p,%d) with hash %8llx%8llx%8llx%8llx found its GPU Object in Read Cache",system::ILogger::ELL_DEBUG,
								instance.asset,instance.uniqueCopyGroupID,hashAsU64[0],hashAsU64[1],hashAsU64[2],hashAsU64[3]
							);
							return;
						}
					}
					// The conversion request we insert needs an instance asset whose unconverted dependencies don't have missing content
					// SUPER SIMPLIFICATION: because we hash and search for readCache items bottom up (BFS), we don't need a stack (DFS) here!
					// Any dependant that's not getting a GPU object due to missing content or GPU cache object for its cache, will show up later during `getDependant`
					// An additional optimization would be to improve the `PatchGetter` to check dependants (only deps) during hashing for missing dfs cache gpu Object (no read cache) and no conversion request.
					auto* isPrehashed = dynamic_cast<const IPreHashed*>(instance.asset);
					if (isPrehashed && isPrehashed->missingContent())
					{
						inputs.logger.log(
							"PreHashed Asset (%p,%d) with hash %8llx%8llx%8llx%8llx has missing content and no GPU Object in Read Cache!",system::ILogger::ELL_ERROR,
							instance.asset,instance.uniqueCopyGroupID,hashAsU64[0],hashAsU64[1],hashAsU64[2],hashAsU64[3]
						);
						return;
					}
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
				{
					const auto& gpuObj = std::get<dfs_cache<DepAssetType>>(dfsCaches).nodes[found.value].gpuObj;
					if (gpuObj.value)
						return gpuObj.value;
				}
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
							if (auto dsLayout=asset->getDescriptorSetLayout(j); dsLayout) // remember layouts are optional
								dsLayouts[j] = getDependant(uniqueCopyGroupID,asset,dsLayout,firstPatchMatch,notAllDepsFound);
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
			if constexpr (std::is_same_v<AssetType,ICPURenderpass>)
			{
				for (auto& entry : conversionRequests)
				{
					const ICPURenderpass* asset = entry.second.canonicalAsset;
					// there is no patching possible for this asset
					for (auto i=0ull; i<entry.second.copyCount; i++)
					{
						// since we don't have dependants we don't care about our group ID
						// we create threadsafe pipeline caches, because we have no idea how they may be used
						assign.operator()<true>(entry.first,entry.second.firstCopyIx,i,device->createRenderpass(asset->getCreationParameters()));
					}
				}
			}
			if constexpr (std::is_same_v<AssetType,ICPUGraphicsPipeline>)
			{
				core::vector<IGPUShader::SSpecInfo> tmpSpecInfo;
				tmpSpecInfo.reserve(5);
				for (auto& entry : conversionRequests)
				{
					const ICPUGraphicsPipeline* asset = entry.second.canonicalAsset;
					// there is no patching possible for this asset
					for (auto i=0ull; i<entry.second.copyCount; i++)
					{
						const auto outIx = i+entry.second.firstCopyIx;
						const auto uniqueCopyGroupID = gpuObjUniqueCopyGroupIDs[outIx];
						// ILogicalDevice::createComputePipelines is rather aggressive on the spec constant validation, so we create one pipeline at a time
						{
							// no derivatives, special flags, etc.
							IGPUGraphicsPipeline::SCreationParams params = {};
							bool depNotFound = false;
							{
								// we choose whatever patch, because there should really only ever be one (all pipeline layouts merge their PC ranges seamlessly)
								params.layout = getDependant(uniqueCopyGroupID,asset,asset->getLayout(),firstPatchMatch,depNotFound).get();
								// we choose whatever patch, because there should only ever be one (users don't influence patching)
								params.renderpass = getDependant(uniqueCopyGroupID,asset,asset->getRenderpass(),firstPatchMatch,depNotFound).get();
								// while there are patches possible for shaders, the only patch which can happen here is changing a stage from UNKNOWN to match the slot here
								tmpSpecInfo.clear();
								using stage_t = ICPUShader::E_SHADER_STAGE;
								for (stage_t stage : {stage_t::ESS_VERTEX,stage_t::ESS_TESSELLATION_CONTROL,stage_t::ESS_TESSELLATION_EVALUATION,stage_t::ESS_GEOMETRY,stage_t::ESS_FRAGMENT})
								{
									const auto& info = asset->getSpecInfo(stage);
									if (info.shader)
										tmpSpecInfo.push_back({
											.entryPoint = info.entryPoint,
											.shader = getDependant(uniqueCopyGroupID,asset,info.shader,firstPatchMatch,depNotFound).get(),
											.entries = info.entries,
											.requiredSubgroupSize = info.requiredSubgroupSize
										});
								}
								params.shaders = tmpSpecInfo;
							}
							if (depNotFound)
								continue;
							params.cached = asset->getCachedCreationParams();
							core::smart_refctd_ptr<IGPUGraphicsPipeline> ppln;
							device->createGraphicsPipelines(inputs.pipelineCache,{&params,1},&ppln);
							assign(entry.first,entry.second.firstCopyIx,i,std::move(ppln));
						}
					}
				}
			}

			// Propagate the results back, since the dfsCache has the original asset pointers as keys, we map in reverse
			auto& stagingCache = std::get<SResults::staging_cache_t<AssetType>>(retval.m_stagingCaches);
			dfsCache.for_each([&](const instance_t<AssetType>& instance, dfs_cache<AssetType>::created_t& created)->void
				{
					// already found in read cache and not converted
					if (created.gpuObj)
						return;

					const auto& contentHash = created.contentHash;
					auto found = conversionRequests.find(contentHash);

					const auto uniqueCopyGroupID = instance.uniqueCopyGroupID;

					const auto hashAsU64 = reinterpret_cast<const uint64_t*>(contentHash.data);
					// can happen if deps were unconverted dummies
					if (found==conversionRequests.end())
					{
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
					if (!gpuObj)
					{
						inputs.logger.log(
							"Conversion for Content Hash %8llx%8llx%8llx%8llx Copy Index %d from Canonical Asset %p Failed.",
							system::ILogger::ELL_ERROR,hashAsU64[0],hashAsU64[1],hashAsU64[2],hashAsU64[3],copyIx,found->second.canonicalAsset
						);
						return;
					}
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
					stagingCache.emplace(gpuObj.get(),CCache<AssetType>::key_t(contentHash,uniqueCopyGroupID));
					// propagate back to dfsCache
					created.gpuObj = std::move(gpuObj);
					// record if a device memory allocation will be needed
					if constexpr (std::is_base_of_v<IDeviceMemoryBacked,AssetType>)
					{
						if (!requestAllocation(&created.gpuObj))
						{
							created.gpuObj = nullptr;
							continue;
						}
					}
					//
					if constexpr (has_type<AssetType>(SResults::convertible_asset_types{}))
					{
						auto& requests = std::get<SResults::conversion_requests_t<ICPUBuffer>>(retval.m_conversionRequests);
						requests.emplace_back(core::smart_refctd_ptr<const AssetType>(instance.asset),created.gpuObj.get());
					}
				}
			);
		};
		// The order of these calls is super important to go BOTTOM UP in terms of hashing and conversion dependants.
		// Both so we can hash in O(Depth) and not O(Depth^2) but also so we have all the possible dependants ready.
		// If two Asset chains are independent then we order them from most catastrophic failure to least.
		dedupCreateProp.operator()<ICPUBuffer>();
//		dedupCreateProp.operator()<ICPUImage>();
// TODO: add backing buffers (not assets) for BLAS and TLAS builds
		// Allocate Memory
		{
			auto getAsBase = [](const memory_backed_ptr_variant_t& var) -> const IDeviceMemoryBacked*
			{
				switch (var.index())
				{
					case 0:
						return std::get<asset_cached_t<ICPUBuffer>*>(var)->get();
					case 1:
						return std::get<asset_cached_t<ICPUImage>*>(var)->get();
					default:
						assert(false);
						break;
				}
				return nullptr;
			};
			// sort each bucket by size from largest to smallest with pessimized allocation size due to alignment
			for (auto& bin : allocationRequests)
				std::sort(bin.second.begin(),bin.second.end(),[getAsBase](const memory_backed_ptr_variant_t& lhs, const memory_backed_ptr_variant_t& rhs)->bool
					{
						const auto& lhsReqs = getAsBase(lhs)->getMemoryReqs();
						const auto& rhsReqs = getAsBase(rhs)->getMemoryReqs();
						const size_t lhsWorstSize = lhsReqs.size+(0x1ull<<lhsReqs.alignmentLog2)-1;
						const size_t rhsWorstSize = rhsReqs.size+(0x1ull<<rhsReqs.alignmentLog2)-1;
						return lhsWorstSize<rhsWorstSize;
					}
				);

			// lets define our order of memory type usage
			const auto& memoryProps = device->getPhysicalDevice()->getMemoryProperties();
			core::vector<uint32_t> memoryTypePreference(memoryProps.memoryTypeCount);
			std::iota(memoryTypePreference.begin(),memoryTypePreference.end(),0);
			std::sort(memoryTypePreference.begin(),memoryTypePreference.end(),
				[&memoryProps](const uint32_t leftIx, const uint32_t rightIx)->bool
				{
					const auto& leftType = memoryProps.memoryTypes[leftIx];
					const auto& rightType = memoryProps.memoryTypes[rightIx];

					using flags_t = IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS;
					const auto& leftTypeFlags = leftType.propertyFlags;
					const auto& rightTypeFlags = rightType.propertyFlags;

					// we want to try types that device local first, then non-device local
					const bool leftDeviceLocal = leftTypeFlags.hasFlags(flags_t::EMPF_DEVICE_LOCAL_BIT);
					const bool rightDeviceLocal = rightTypeFlags.hasFlags(flags_t::EMPF_DEVICE_LOCAL_BIT);
					if (leftDeviceLocal!=rightDeviceLocal)
						return leftDeviceLocal;

					// then we want to allocate from largest heap to smallest
					// TODO: actually query the amount of free memory using VK_EXT_memory_budget
					const size_t leftHeapSize = memoryProps.memoryHeaps[leftType.heapIndex].size;
					const size_t rightHeapSize = memoryProps.memoryHeaps[rightType.heapIndex].size;
					if (leftHeapSize<rightHeapSize)
						return true;
					else if (leftHeapSize!=rightHeapSize)
						return false;

					// within those types we want to first do non-mappable
					const bool leftMappable = leftTypeFlags.value&(flags_t::EMPF_HOST_READABLE_BIT|flags_t::EMPF_HOST_WRITABLE_BIT);
					const bool rightMappable = rightTypeFlags.value&(flags_t::EMPF_HOST_READABLE_BIT|flags_t::EMPF_HOST_WRITABLE_BIT);
					if (leftMappable!=rightMappable)
						return rightMappable;

					// then non-coherent
					const bool leftCoherent = leftTypeFlags.hasFlags(flags_t::EMPF_HOST_COHERENT_BIT);
					const bool rightCoherent = rightTypeFlags.hasFlags(flags_t::EMPF_HOST_COHERENT_BIT);
					if (leftCoherent!=rightCoherent)
						return rightCoherent;

					// then non-cached
					const bool leftCached = leftTypeFlags.hasFlags(flags_t::EMPF_HOST_CACHED_BIT);
					const bool rightCached = rightTypeFlags.hasFlags(flags_t::EMPF_HOST_CACHED_BIT);
					if (leftCached!=rightCached)
						return rightCached;

					// otherwise equal
					return false;
				}
			);
			
			// go over our preferred memory types and try to service allocations from them
			core::vector<size_t> offsetsTmp;
			for (const auto memTypeIx : memoryTypePreference)
			{
				// we could try to service multiple requirements with the same allocation, but we probably don't need to try so hard
				for (auto& reqBin : allocationRequests)
				if (reqBin.first.compatibileMemoryTypeBits&(0x1<<memTypeIx))
				{
					using allocate_flags_t = IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS;
					IDeviceMemoryAllocator::SAllocateInfo info = {
						.size = offsetsTmp.back(), // we have one more item in the array
						.flags = reqBin.first.needsDeviceAddress ? allocate_flags_t::EMAF_DEVICE_ADDRESS_BIT:allocate_flags_t::EMAF_NONE,
						.memoryTypeIndex = memTypeIx,
						.dedication = nullptr
					};

					auto& binItems = reqBin.second;
					const auto binItemCount = reqBin.second.size();
					// the `std::exclusive_scan` syntax is more effort for this
					{
						offsetsTmp.resize(binItemCount+1);
						offsetsTmp[0] = 0;
						for (size_t i=0; i<binItemCount;)
						{
							const auto* memBacked = getAsBase(binItems[i]);
							const auto& memReqs = memBacked->getMemoryReqs();
							// round up the offset to get the correct alignment
							offsetsTmp[++i] = core::roundUp(offsetsTmp[i],0x1ull<<memReqs.alignmentLog2)+memReqs.size;
						}
					}
					// allocate in progression of combined allocations, while trying allocate as much as possible in a single allocation
					auto itemsBegin = binItems.begin();
					for (auto firstOffsetIt=offsetsTmp.begin(); firstOffsetIt!=offsetsTmp.end(); )
					for (auto nextOffsetIt=offsetsTmp.end(); nextOffsetIt>firstOffsetIt; nextOffsetIt--)
					{
						const size_t combinedCount = std::distance(firstOffsetIt,nextOffsetIt);
						const size_t lastIx = combinedCount-1;
						// if we take `combinedCount` starting at `firstItem` their allocation would need this size
						info.size = (firstOffsetIt[lastIx]-*firstOffsetIt)+getAsBase(itemsBegin[lastIx])->getMemoryReqs().size;
						auto allocation = device->allocate(info);
						if (allocation.isValid())
						{
							// bind everything
							for (auto i=0; i<combinedCount; i++)
							{
								const auto& toBind = itemsBegin[i];
								bool bindSuccess = false;
								const IDeviceMemoryBacked::SMemoryBinding binding = {
									.memory = allocation.memory.get(),
									// base allocation offset, plus relative offset for this batch
									.offset = allocation.offset+firstOffsetIt[i]-*firstOffsetIt
								};
								switch (toBind.index())
								{
									case 0:
										{
											const ILogicalDevice::SBindBufferMemoryInfo info =
											{
												.buffer = std::get<asset_cached_t<ICPUBuffer>*>(toBind)->get(),
												.binding = binding
											};
											bindSuccess = device->bindBufferMemory(1,&info);
										}
										break;
									case 1:
										{
											const ILogicalDevice::SBindImageMemoryInfo info =
											{
												.image = std::get<asset_cached_t<ICPUImage>*>(toBind)->get(),
												.binding = binding
											};
											bindSuccess = device->bindImageMemory(1,&info);
										}
										break;
									default:
										break;
								}
								assert(bindSuccess);
							}
							// erase `combinedCount` items from bin
							itemsBegin = binItems.erase(itemsBegin,itemsBegin+combinedCount);
							// move onto next batch
							firstOffsetIt = nextOffsetIt;
							break;
						}
						// we're unable to allocate even for a single item with a dedicated allocation, skip trying then
						else if ((nextOffsetIt-1)==firstOffsetIt)
						{
							firstOffsetIt = nextOffsetIt;
							itemsBegin++;
							break;
						}
					}
				}
			}

			// If we failed to allocate and bind memory from any heap, need to wipe the GPU Obj as a failure
			for (const auto& reqBin : allocationRequests)
			for (auto& req : reqBin.second)
			{
				const auto asBacked = getAsBase(req);
				if (asBacked->getBoundMemory().isValid())
					continue;
				switch (req.index())
				{
					case 0:
						*std::get<asset_cached_t<ICPUBuffer>*>(req) = {};
						break;
					case 1:
						*std::get<asset_cached_t<ICPUImage>*>(req) = {};
						break;
					default:
						assert(false);
						break;
				}
				inputs.logger.log("Allocation and Binding of Device Memory for \"%\" failed, deleting GPU object.",system::ILogger::ELL_ERROR,asBacked->getObjectDebugName());
			}
			allocationRequests.clear();
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
		dedupCreateProp.operator()<ICPURenderpass>();
		dedupCreateProp.operator()<ICPUGraphicsPipeline>();
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
		const auto& stagingCache = std::get<SResults::staging_cache_t<AssetType>>(retval.m_stagingCaches);
		auto& results = std::get<SResults::vector_t<AssetType>>(retval.m_gpuObjects);
		for (size_t i=0; i<count; i++)
		if (auto asset=assets[i]; asset)
		{
			const auto uniqueCopyGroupID = metadata[i].uniqueCopyGroupID;
			// simple and easy to find all the associated items
			if (!metadata[i].patchIndex)
			{
				inputs.logger.log("No valid patch could be created for Root Asset %p in group %d",system::ILogger::ELL_ERROR,asset,uniqueCopyGroupID);
				continue;
			}
			const auto& found = dfsCache.nodes[metadata[i].patchIndex.value];
			// write it out to the results
			if (const auto& gpuObj=found.gpuObj; gpuObj) // found from the `input.readCache`
			{
				results[i] = gpuObj;
				// if something with this content hash is in the stagingCache, then it must match the `found->gpuObj`
				if (auto finalCacheIt=stagingCache.find(gpuObj.get()); finalCacheIt!=stagingCache.end())
				{
					const bool matches = finalCacheIt->second==CCache<AssetType>::key_t(found.contentHash,uniqueCopyGroupID);
					assert(matches);
				}
			}
			else
				inputs.logger.log("No GPU Object could be found or created for Root Asset %p in group %d",system::ILogger::ELL_ERROR,asset,uniqueCopyGroupID);
		}
	};
	core::for_each_in_tuple(inputs.assets,finalize);

	retval.m_converter = core::smart_refctd_ptr<CAssetConverter>(this);
	retval.m_logger = system::logger_opt_smart_ptr(core::smart_refctd_ptr<system::ILogger>(inputs.logger.get()));
	return retval;
}

//
bool CAssetConverter::convert_impl(SResults&& reservations, SConvertParams& params)
{
	if (!reservations.m_converter)
	{
		reservations.m_logger.log("Cannot call convert on an unsuccessful reserve result!");
		return false;
	}
	assert(reservations.m_converter.get()==this);

	const auto reqQueueFlags = reservations.getRequiredQueueFlags();
	// Anything to do?
	if (reqQueueFlags.value!=IQueue::FAMILY_FLAGS::NONE)
	{
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

		// wipe gpu item in staging cache (this may drop it as well if it was made for only a root asset == no users)
		auto makFailureInStaging = [&]<Asset AssetType>(asset_traits<AssetType>::video_t* gpuObj)->void
		{
			auto& stagingCache = std::get<SResults::staging_cache_t<AssetType>>(reservations.m_stagingCaches);
			const auto found = stagingCache.find(gpuObj);
			assert(found!=stagingCache.end());
			// change the content hash on the reverse map to a NoContentHash
			const_cast<core::blake3_hash_t&>(found->second.value) = {};
		};

		// upload Buffers
		auto& buffersToUpload = std::get<SResults::conversion_requests_t<ICPUBuffer>>(reservations.m_conversionRequests);
		{
			core::vector<IGPUCommandBuffer::SBufferMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier>> ownershipTransfers;
			ownershipTransfers.reserve(buffersToUpload.size());
			// do the uploads
			for (auto& item : buffersToUpload)
			{
				auto found = std::get<SResults::staging_cache_t<ICPUBuffer>>(reservations.m_stagingCaches).find(item.gpuObj);
				const SBufferRange<IGPUBuffer> range = {
					.offset = 0,
					.size = item.gpuObj->getCreationParams().size,
					.buffer = core::smart_refctd_ptr<IGPUBuffer>(item.gpuObj)
				};
				const bool success = params.utilities->updateBufferRangeViaStagingBuffer(params.transfer,range,item.canonical->getPointer());
				// let go of canonical asset (may free RAM)
				item.canonical = nullptr;
				if (!success)
				{
					reservations.m_logger.log("Data upload failed for \"%s\"",system::ILogger::ELL_ERROR,item.gpuObj->getObjectDebugName());
					makFailureInStaging.operator()<ICPUBuffer>(item.gpuObj);
					continue;
				}
				// enqueue ownership release if necessary
				if (const auto ownerQueueFamily=params.getFinalOwnerQueueFamily(item.gpuObj,{}); ownerQueueFamily!=IQueue::FamilyIgnored && params.transfer.queue->getFamilyIndex()!=ownerQueueFamily)
				{
					ownershipTransfers.push_back({
						.barrier = {
							.dep = {
								.srcStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT,
								.srcAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT
								// leave rest empty, we can release whenever after the copies and before the semaphore signal
							},
							.ownershipOp = IGPUCommandBuffer::SOwnershipTransferBarrier::OWNERSHIP_OP::RELEASE,
							.otherQueueFamilyIndex = ownerQueueFamily
						},
						.range = range
					});
				}
			}
			buffersToUpload.clear();
			// release ownership
			if (!params.transfer.getScratchCommandBuffer()->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE,{.memBarriers={},.bufBarriers=ownershipTransfers}))
				reservations.m_logger.log("Ownership Releases of Buffers Failed",system::ILogger::ELL_ERROR);
		}

#if 0
		auto& imagesToUpload = std::get<SResults::conversion_requests_t<ICPUImage>>(reservations.m_conversionRequests);
		{
			core::vector<IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier>> layoutTransitions;
			layoutTransitions.reserve(imagesToUpload.size());
			// first transition all images to dst-optimal layout
			for (const auto& item : imagesToUpload)
			{
				const auto& creationParams = item.gpuObj->getCreationParameters();
				layoutTransitions.push_back({
						.barrier = {
							.dep = {
								// first usage doesn't need to sync against anything, so leave src default
								.dstStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT,
								.dstAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT
							}
							// plain usage just acquires ownership (without content preservation though
						},
						.image = item.gpuObj,
						.subresourceRange = {
							.aspectMask = core::bitflag(IGPUImage::E_ASPECT_FLAGS::EAF_COLOR_BIT),
							.baseMipLevel = 0u,
							.levelCount = creationParams.mipLevels,
							.baseArrayLayer = 0u,
							.layerCount = creationParams.arrayLayers
						},
						.oldLayout = IGPUImage::LAYOUT::UNDEFINED,
						.newLayout = IGPUImage::LAYOUT::TRANSFER_DST_OPTIMAL
				});
			}
			if (!params.transfer.getScratchCommandBuffer()->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE,{.memBarriers={},.bufBarriers={},.imgBarriers=layoutTransitions}))
			{
				reservations.m_logger.log("Layout Transition to TRANSFER_DST_OPTIMAL failed for all images",system::ILogger::ELL_ERROR);
				for (const auto& item : imagesToUpload) // wipe everything
					makFailureInStaging.operator()<ICPUImage>(item.gpuObj);
			}
			else // upload Images
			{
				for (const auto& item : imagesToUpload)
				{
					const bool success = params.utilities->updateImageViaStagingBuffer(
						params.transfer,
						item.canonical->getBuffer()->getPointer(),
						item.canonical->getCreationParameters().format,
						item.gpuObj,
						IGPUImage::LAYOUT::TRANSFER_DST_OPTIMAL,
						item.canonical->getRegions()
					);
					if (!success)
					{
						reservations.m_logger.log("Data upload failed for \"%s\"",system::ILogger::ELL_ERROR,item.gpuObj->getObjectDebugName());
						makFailureInStaging.operator()<ICPUImage>(item.gpuObj);
						continue;
					}
					// TODO: enqueue ownership release or layout transition if necessary
				}
				// TODO: layout transitions etc.
			}
		}
#endif

		// TODO: build BLASes and TLASes
	}

	// want to check if deps successfully exist
	auto missingDependent = [&reservations]<Asset AssetType>(const asset_traits<AssetType>::video_t* dep)->bool
	{
		auto& stagingCache = std::get<SResults::staging_cache_t<AssetType>>(reservations.m_stagingCaches);
		auto found = stagingCache.find(const_cast<asset_traits<AssetType>::video_t*>(dep));
		if (found!=stagingCache.end() && found->second.value==core::blake3_hash_t{})
			return true;
		// dependent might be in readCache of one or more converters, so if in doubt assume its okay
		return false;
	};
	// insert items into cache if overflows handled fine and commandbuffers ready to be recorded
	auto mergeCache = [&]<Asset AssetType>()->void
	{
		auto& stagingCache = std::get<SResults::staging_cache_t<AssetType>>(reservations.m_stagingCaches);
		auto& cache = std::get<CCache<AssetType>>(m_caches);
		cache.m_forwardMap.reserve(cache.m_forwardMap.size()+stagingCache.size());
		cache.m_reverseMap.reserve(cache.m_reverseMap.size()+stagingCache.size());
		for (auto& item : stagingCache)
		if (item.second.value!=core::blake3_hash_t{}) // didn't get wiped
		{
			// rescan all the GPU objects and find out if they depend on anything that failed, if so add to failure set
			bool depsMissing = false;
			// only go over types we could actually break via missing upload/build (i.e. pipelines are unbreakable)
//			if constexpr (std::is_same_v<AssetType,ICPUBufferView>)
//				depMissing = missingDependent.operator()<ICPUBuffer>(item.first->getBuffer());
//			if constexpr (std::is_same_v<AssetType,ICPUImageView>)
//				depMissing = missingDependent.operator()<ICPUImage>(item.first->getCreationParams().image);
			if constexpr (std::is_same_v<AssetType,ICPUDescriptorSet>)
			{
				// TODO
			}
			if (depsMissing)
			{
				// wipe self, to let users know
				item.second.value = {};
				continue;
			}
			asset_cached_t<AssetType> cached;
			cached.value = core::smart_refctd_ptr<typename asset_traits<AssetType>::video_t>(item.first);
			cache.m_forwardMap.emplace(item.second,std::move(cached));
		}
	};
	// again, need to go bottom up so we can check dependencies being successes
	mergeCache.operator()<ICPUBuffer>();
//	mergeCache.operator()<ICPUImage>();
//	mergeCache.operator()<ICPUBottomLevelAccelerationStructure>();
//	mergeCache.operator()<ICPUTopLevelAccelerationStructure>();
//	mergeCache.operator()<ICPUBufferView>();
	mergeCache.operator()<ICPUShader>();
	mergeCache.operator()<ICPUSampler>();
	mergeCache.operator()<ICPUDescriptorSetLayout>();
	mergeCache.operator()<ICPUPipelineLayout>();
	mergeCache.operator()<ICPUPipelineCache>();
	mergeCache.operator()<ICPUComputePipeline>();
	mergeCache.operator()<ICPURenderpass>();
	mergeCache.operator()<ICPUGraphicsPipeline>();
//	mergeCache.operator()<ICPUDescriptorSet>();
//	mergeCache.operator()<ICPUFramebuffer>();

	return true;
}

ISemaphore::future_t<bool> CAssetConverter::SConvertParams::autoSubmit()
{
	// TODO: transfer first, then compute

	return {};
}

}
}