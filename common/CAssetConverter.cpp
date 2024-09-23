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
		// empty classes are still sizeof==1
		if (std::is_empty_v<video::CAssetConverter::patch_impl_t<AssetType>>)
			return;
		hasher.update(&input,sizeof(input));
	}
};

namespace video
{
// No asset has a 0 length input to the hash function
const core::blake3_hash_t CAssetConverter::CHashCache::NoContentHash = static_cast<core::blake3_hash_t>(core::blake3_hasher());

CAssetConverter::patch_impl_t<ICPUSampler>::patch_impl_t(const ICPUSampler* sampler) : anisotropyLevelLog2(sampler->getParams().AnisotropicFilter) {}
bool CAssetConverter::patch_impl_t<ICPUSampler>::valid(const ILogicalDevice* device)
{
	if (anisotropyLevelLog2>5) // unititialized
		return false;
	const auto& limits = device->getPhysicalDevice()->getLimits();
	if (anisotropyLevelLog2>limits.maxSamplerAnisotropyLog2)
		anisotropyLevelLog2 = limits.maxSamplerAnisotropyLog2;
	return true;
}

CAssetConverter::patch_impl_t<ICPUShader>::patch_impl_t(const ICPUShader* shader) : stage(shader->getStage()) {}
bool CAssetConverter::patch_impl_t<ICPUShader>::valid(const ILogicalDevice* device)
{
	const auto& features = device->getEnabledFeatures();
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
bool CAssetConverter::patch_impl_t<ICPUBuffer>::valid(const ILogicalDevice* device)
{
	const auto& features = device->getEnabledFeatures();
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

CAssetConverter::patch_impl_t<ICPUImage>::patch_impl_t(const ICPUImage* image)
{
	const auto& params = image->getCreationParameters();
	format = params.format;
	usageFlags = params.usage;
	if (isDepthOrStencilFormat(format))
	{
		const bool hasStencil = !isDepthOnlyFormat(format);
		if (hasStencil)
			stencilUsage = params.actualStencilUsage();
		const bool hasDepth = !isStencilOnlyFormat(format);
		bool foundDepthReg = !hasDepth;
		bool foundStencilReg = !hasStencil;
		for (const auto& region : image->getRegions())
		{
			foundDepthReg = foundDepthReg || region.imageSubresource.aspectMask.hasFlags(IGPUImage::EAF_DEPTH_BIT);
			foundStencilReg = foundStencilReg || region.imageSubresource.aspectMask.hasFlags(IGPUImage::EAF_STENCIL_BIT);
			if (foundDepthReg && foundStencilReg)
				break;
		}
	}
	else if(!image->getRegions().empty())
		usageFlags |= usage_flags_t::EUF_TRANSFER_DST_BIT;

	using create_flags_t = IGPUImage::E_CREATE_FLAGS;
	mutableFormat = params.flags.hasFlags(create_flags_t::ECF_MUTABLE_FORMAT_BIT);
	cubeCompatible = params.flags.hasFlags(create_flags_t::ECF_CUBE_COMPATIBLE_BIT);
	mutableFormat = params.flags.hasFlags(create_flags_t::ECF_2D_ARRAY_COMPATIBLE_BIT);
	mutableFormat = params.flags.hasFlags(create_flags_t::ECF_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT);
	mipLevels = params.mipLevels;
}
bool CAssetConverter::patch_impl_t<ICPUImage>::valid(const ILogicalDevice* device)
{
	const auto& features = device->getEnabledFeatures();
	// usages we don't have features for
	if (usageFlags.hasFlags(usage_flags_t::EUF_SHADING_RATE_ATTACHMENT_BIT))// && !features.shadingRate)
		return false;
	if (usageFlags.hasFlags(usage_flags_t::EUF_FRAGMENT_DENSITY_MAP_BIT) && !features.fragmentDensityMap)
		return false;
	// create flags
// mutableFormat
// cube compatible
// 2d array comp
// block texel view compat (compressed as uncompressed)
// extended usage
	return true;
}

CAssetConverter::patch_impl_t<ICPUBufferView>::patch_impl_t(const ICPUBufferView* view) {}
bool CAssetConverter::patch_impl_t<ICPUBufferView>::valid(const ILogicalDevice* device)
{
	// note that we don't check the validity of things we don't patch, so offset alignment, size and format
	// we could check if the format and usage make sense, but it will be checked by the driver anyway
	return true;
}

CAssetConverter::patch_impl_t<ICPUImageView>::patch_impl_t(const ICPUImageView* view, const core::bitflag<usage_flags_t> extraSubUsages) :
	subUsages(view->getCreationParameters().actualUsages()|extraSubUsages)
{
	const auto& params = view->getCreationParameters();
	originalFormat = params.format;
	// Impossible to deduce this without knowing all the sampler & view combos that will be used,
	// because under descriptor indexing we can use any sampler with any view in SPIR-V.
	// Also because we don't know what Descriptor Sets will be used with what Pipelines!
	if (subUsages.hasFlags(usage_flags_t::EUF_SAMPLED_BIT))
	{
		if (isDepthOrStencilFormat(originalFormat))
		{
			if (params.subresourceRange.aspectMask.hasFlags(IGPUImage::E_ASPECT_FLAGS::EAF_DEPTH_BIT))
				depthCompareSampledImage = true;
		}
		// Also have no info about any blit cmd that could use this view's image as source
		else if (!asset::isIntegerFormat(originalFormat))
			linearlySampled = true;
// TODO: full mip chain?
	}
	// same stuff for storage images
	if (subUsages.hasFlags(usage_flags_t::EUF_STORAGE_BIT))
	{
		// Deducing this in another way would seriously hinder our ability to do format promotions.
		// To ensure the view doesn't get promoted away from device-feature dependant atomic storage capable formats, use explicit patches upon input! 
		storageAtomic = originalFormat ==EF_R32_UINT;
	}
	//
	if (subUsages.hasFlags(usage_flags_t::EUF_RENDER_ATTACHMENT_BIT))
	{
		// The only way you'd be able to deduce this is by watching for any graphics pipeline with a blend state being created
		// for a subpass in renderpass thats also referenced by a framebuffer that uses this image view. However there are also
		// subpass compatibility rules which could allow someone to use a compatible subpass graphics pipeline.
		// And finally dynamic rendering has render infos, so all connection between pipelinesm, images and renderpasses is lost.
		//if (!isIntegerFormat(format) && !isDepthOrStencilFormat(format))
			//attachmentBlend = true;
		// actually this deduction would promote RGB9E5 to RGBA16F even if RGB9E5 supported by the device
	}
	else if (originalFormat==params.image->getCreationParameters().format)
		originalFormat = EF_UNKNOWN; // format is the same as the base image and we are are not using it for renderpass attachments, allow to mutate with base image's
}
bool CAssetConverter::patch_impl_t<ICPUImageView>::valid(const ILogicalDevice* device)
{
	invalid = true;
	// check exotic usages against device caps
	const auto& features = device->getEnabledFeatures();
	if (subUsages.hasFlags(usage_flags_t::EUF_SHADING_RATE_ATTACHMENT_BIT))// && !features.fragmentDensityMap)
		return false;
	if (subUsages.hasFlags(usage_flags_t::EUF_FRAGMENT_DENSITY_MAP_BIT) && !features.fragmentDensityMap)
		return false;
	const auto* physDev = device->getPhysicalDevice();
	if (storageImageLoadWithoutFormat && !physDev->getLimits().shaderStorageImageReadWithoutFormat)
		return false;
	// now check for the format (if the format is immutable)
	if (originalFormat!=EF_UNKNOWN)
	{
		// normally we wouldn't check for usages being valid, but combine needs to know about validity before combining
		IPhysicalDevice::SFormatImageUsages::SUsage usages = {subUsages};
		usages.linearlySampledImage = linearlySampled;
		if (usages.storageImage) // we require this anyway
			usages.storageImageStoreWithoutFormat = true;
		usages.storageImageAtomic = storageAtomic;
		usages.attachmentBlend = attachmentBlend;
		usages.storageImageLoadWithoutFormat = storageImageLoadWithoutFormat;
		usages.depthCompareSampledImage = depthCompareSampledImage;
		invalid = (physDev->getImageFormatUsagesOptimalTiling()[originalFormat]&usages)!=usages || (physDev->getImageFormatUsagesLinearTiling()[originalFormat]&usages)!=usages;
	}
	else
		invalid = false;
	return !invalid;
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
bool CAssetConverter::patch_impl_t<ICPUPipelineLayout>::valid(const ILogicalDevice* device)
{
	const auto& limits = device->getPhysicalDevice()->getLimits();
	for (auto byte=limits.maxPushConstantsSize; byte<pushConstantBytes.size(); byte++)
	if (pushConstantBytes[byte]!=shader_stage_t::ESS_UNKNOWN)
		return false;
	return !invalid;
}



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

//
template<typename CRTP>
class AssetVisitor : public CRTP
{
	public:
		using AssetType = CRTP::AssetType;

		bool operator()()
		{
			if (!instance.asset)
				return false;
			return impl(instance,patch);
		}

		const instance_t<AssetType> instance;
		const CAssetConverter::patch_t<AssetType>& patch;

	protected:
		template<Asset DepType, typename... ExtraArgs>
		bool descend(const DepType* dep, CAssetConverter::patch_t<DepType>&& candidatePatch, ExtraArgs&&... extraArgs)
		{
			assert(dep);
			return bool(
				CRTP::descend_impl(
					instance,patch,
					{dep,CRTP::getDependantUniqueCopyGroupID(instance.uniqueCopyGroupID,instance.asset,dep)},
					std::move(candidatePatch),
					std::forward<ExtraArgs>(extraArgs)...
				)
			);
		}

	private:
		inline bool impl(const instance_t<ICPUBufferView>& instance, const CAssetConverter::patch_t<ICPUBufferView>& userPatch)
		{
			const auto* dep = instance.asset->getUnderlyingBuffer();
			if (!dep)
				return false;
			CAssetConverter::patch_t<ICPUBuffer> patch = {dep};
			if (userPatch.utbo)
				patch.usage |= IGPUBuffer::E_USAGE_FLAGS::EUF_UNIFORM_TEXEL_BUFFER_BIT;
			if (userPatch.stbo)
				patch.usage |= IGPUBuffer::E_USAGE_FLAGS::EUF_STORAGE_TEXEL_BUFFER_BIT;
			return descend<ICPUBuffer>(dep,std::move(patch));
		}
		inline bool impl(const instance_t<ICPUImageView>& instance, const CAssetConverter::patch_t<ICPUImageView>& userPatch)
		{
			const auto& params = instance.asset->getCreationParameters();
			const auto* dep = params.image;
			if (!dep)
				return false;
			CAssetConverter::patch_t<ICPUImage> patch = {dep};
			patch.usage |= userPatch.subUsages;
			// figure out create flags
			// mipLevels
			// recomputeMips
			// format
			return descend<ICPUImage>(dep,std::move(patch));
		}
		inline bool impl(const instance_t<ICPUDescriptorSetLayout>& instance, const CAssetConverter::patch_t<ICPUDescriptorSetLayout>& userPatch)
		{
			const auto samplers = instance.asset->getImmutableSamplers();
			for (size_t i=0; i<samplers.size(); i++)
			{
				const auto sampler = samplers[i].get();
				if (!sampler || !descend(sampler,{sampler},i))
					return false;
			}
			return true;
		}
		inline bool impl(const instance_t<ICPUPipelineLayout>& instance, const CAssetConverter::patch_t<ICPUPipelineLayout>& userPatch)
		{
			// individual DS layouts are optional
			for (auto i=0; i<ICPUPipelineLayout::DESCRIPTOR_SET_COUNT; i++)
			{
				if (auto layout=instance.asset->getDescriptorSetLayout(i); layout)
				{
					if (!descend(layout,{layout},i))
						return false;
				}
				else
					CRTP::template nullOptional<ICPUDescriptorSetLayout>();
			}
			return true;
		}
		inline bool impl(const instance_t<ICPUComputePipeline>& instance, const CAssetConverter::patch_t<ICPUComputePipeline>& userPatch)
		{
			const auto* asset = instance.asset;
			const auto* layout = asset->getLayout();
			if (!layout || !descend(layout,{layout}))
				return false;
			const auto& specInfo = asset->getSpecInfo();
			const auto* shader = specInfo.shader;
			if (!shader)
				return false;
			CAssetConverter::patch_t<ICPUShader> patch = {shader};
			constexpr auto stage = IGPUShader::E_SHADER_STAGE::ESS_COMPUTE;
			patch.stage = stage;
			if (!descend(shader,std::move(patch),stage,specInfo))
				return false;
			return true;
		}
		inline bool impl(const instance_t<ICPUGraphicsPipeline>& instance, const CAssetConverter::patch_t<ICPUGraphicsPipeline>& userPatch)
		{
			const auto* asset = instance.asset;
			const auto* layout = asset->getLayout();
			if (!layout || !descend(layout,{layout}))
				return false;
			const auto* rpass = asset->getRenderpass();
			if (!rpass || !descend(rpass,{rpass}))
				return false;
			using stage_t = ICPUShader::E_SHADER_STAGE;
			for (stage_t stage : {stage_t::ESS_VERTEX,stage_t::ESS_TESSELLATION_CONTROL,stage_t::ESS_TESSELLATION_EVALUATION,stage_t::ESS_GEOMETRY,stage_t::ESS_FRAGMENT})
			{
				const auto& specInfo = asset->getSpecInfo(stage);
				const auto* shader = specInfo.shader;
				if (!shader)
				{
					if (stage==stage_t::ESS_VERTEX) // required
						return false;
					CRTP::template nullOptional<ICPUShader>();
					continue;
				}
				CAssetConverter::patch_t<ICPUShader> patch = {shader};
				patch.stage = stage;
				if (!descend(shader,std::move(patch),stage,specInfo))
					return false;
			}
			return true;
		}
		inline bool impl(const instance_t<ICPUDescriptorSet>& instance, const CAssetConverter::patch_t<ICPUDescriptorSet>& userPatch)
		{
			const auto* asset = instance.asset;
			const auto* layout = asset->getLayout();
			if (!layout || !descend(layout,{layout}))
				return false;
			for (auto i=0u; i<static_cast<uint32_t>(IDescriptor::E_TYPE::ET_COUNT); i++)
			{
				const auto type = static_cast<IDescriptor::E_TYPE>(i);
				const auto allInfos = asset->getDescriptorInfoStorage(type);
				if (allInfos.empty())
					continue;
				const auto& redirect = layout->getDescriptorRedirect(type);
				const auto bindingCount = redirect.getBindingCount();
				// go over every binding
				for (auto j=0; j<bindingCount; j++)
				{
					const IDescriptorSetLayoutBase::CBindingRedirect::storage_range_index_t storageRangeIx(j);
					const auto binding = redirect.getBinding(storageRangeIx);
					const uint32_t count = redirect.getCount(storageRangeIx);
					// this is where the descriptors have their flattened place in a unified array 
					const auto* infos = allInfos.data()+redirect.getStorageOffset(storageRangeIx).data;
					for (uint32_t el=0u; el<count; el++)
					{
						const auto& info = infos[el];
						if (auto untypedDesc=info.desc.get(); untypedDesc) // written descriptors are optional
						switch (IDescriptor::GetTypeCategory(type))
						{
							case IDescriptor::EC_BUFFER:
							{
								auto buffer = static_cast<const ICPUBuffer*>(untypedDesc);
								CAssetConverter::patch_t<ICPUBuffer> patch = {buffer};
								switch(type)
								{
									case IDescriptor::E_TYPE::ET_UNIFORM_BUFFER:
									case IDescriptor::E_TYPE::ET_UNIFORM_BUFFER_DYNAMIC:
										patch.usage |= IGPUBuffer::E_USAGE_FLAGS::EUF_UNIFORM_BUFFER_BIT;
										break;
									case IDescriptor::E_TYPE::ET_STORAGE_BUFFER:
									case IDescriptor::E_TYPE::ET_STORAGE_BUFFER_DYNAMIC:
										patch.usage |= IGPUBuffer::E_USAGE_FLAGS::EUF_STORAGE_BUFFER_BIT;
										break;
									default:
										assert(false);
										return false;
								}
								if (!descend(buffer,std::move(patch),type,binding,el,info.info.buffer))
									return false;
								break;
							}
							case IDescriptor::EC_SAMPLER:
							{
								auto sampler = static_cast<const ICPUSampler*>(untypedDesc);
								if (!descend(sampler,{sampler},type,binding,el))
									return false;
								break;
							}
							case IDescriptor::EC_IMAGE:
							{
								auto imageView = static_cast<const ICPUImageView*>(untypedDesc);
								IGPUImage::E_USAGE_FLAGS usage;
								switch (type)
								{
									case IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER:
										{
											const auto* sampler = info.info.combinedImageSampler.sampler.get();
											if (!descend(sampler,{sampler},type,binding,el))
												return false;
										}
										[[fallthrough]];
									case IDescriptor::E_TYPE::ET_SAMPLED_IMAGE:
										usage = IGPUImage::E_USAGE_FLAGS::EUF_SAMPLED_BIT;
										break;
									case IDescriptor::E_TYPE::ET_STORAGE_IMAGE:
										usage = IGPUImage::E_USAGE_FLAGS::EUF_STORAGE_BIT;
										break;
									case IDescriptor::E_TYPE::ET_INPUT_ATTACHMENT:
										usage = IGPUImage::E_USAGE_FLAGS::EUF_INPUT_ATTACHMENT_BIT;
										break;
									default:
										assert(false);
										break;
								}
								if (!descend(imageView,{imageView,usage},type,binding,el,info.info.image.imageLayout))
									return false;
								break;
							}
							case IDescriptor::EC_BUFFER_VIEW:
							{
								auto bufferView = static_cast<const ICPUBufferView*>(untypedDesc);
								CAssetConverter::patch_t<ICPUBufferView> patch = {bufferView};
								switch (type)
								{
									case IDescriptor::E_TYPE::ET_UNIFORM_TEXEL_BUFFER:
										patch.utbo = true;
										break;
									case IDescriptor::E_TYPE::ET_STORAGE_TEXEL_BUFFER:
										patch.stbo = true;
										break;
									default:
										assert(false);
										return false;
								}
								if (!descend(bufferView,std::move(patch),type,binding,el))
									return false;
								break;
							}
							case IDescriptor::EC_ACCELERATION_STRUCTURE:
							{
								_NBL_TODO();
								[[fallthrough]];
							}
							default:
								assert(false);
								return false;
						}
						CRTP::template nullOptional<IDescriptor>();
					}
				}
			}
			return true;
		}
};


//
struct patch_index_t
{
	inline bool operator==(const patch_index_t&) const = default;
	inline bool operator!=(const patch_index_t&) const = default;

	explicit inline operator bool() const {return operator!=({});}

	uint64_t value = 0xdeadbeefBADC0FFEull;
};
// This cache stops us traversing an asset with the same user group and patch more than once.
template<asset::Asset AssetType>
class dfs_cache
{
	public:
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

	private:
		// Implement both finding the first node for an instance with a compatible patch, and merging patches plus inserting new nodes if nothing can be merged
		template<bool ReplaceWithCombined>
		inline patch_index_t* impl(const instance_t<AssetType>& instance, const CAssetConverter::patch_t<AssetType>& soloPatch)
		{
			// get all the existing patches for the same (AssetType*,UniqueCopyGroupID)
			auto found = instances.find(instance);
			if (found==instances.end())
				return nullptr;
			// we don't want to pass the device to this function, it just assumes the patch will be valid without touch-ups
			//assert(soloPatch.valid(device));
			patch_index_t* pIndex = &found->second;
			// iterate over linked list
			while (*pIndex)
			{
				// get our candidate
				auto& candidate = nodes[pIndex->value];
				// found a thing, try-combine the patches
				auto [success,combined] = candidate.patch.combine(soloPatch);
				if (success)
				{
					if constexpr (ReplaceWithCombined)
					{
						// change the patch to a combined version
						candidate.patch = std::move(combined);
						break;
					}
					else
					{
						// to truly "find" a patched entry, the candidate needs to equal combined
						if (candidate.patch==combined)
							break;
					}
				}
				// else try the next one
				pIndex = &candidate.next;
			}
			return pIndex;
		}

	public:
		// Find the first node for an instance with a compatible patch and return its index
		inline std::pair<patch_index_t,bool> insert(const instance_t<AssetType>& instance, CAssetConverter::patch_t<AssetType>&& soloPatch)
		{
			const patch_index_t newPatchIndex = {nodes.size()};
			auto pIndex = impl<true>(instance,soloPatch);
			// found a linked list for this entry
			if (pIndex)
			{
				// found a patch and merged it
				if (*pIndex)
					return {*pIndex,false};
				// nothing mergable found, make old TAIL point to new node about to be inserted
				*pIndex = newPatchIndex;
			}
			else // there isn't even a linked list head for this entry
			{
				// phmap's unordered map doesn't even care about the found hint
				instances.emplace(instance,newPatchIndex);
			}
			// both non-mergable and completely not found cases fall through here
			nodes.emplace_back(std::move(soloPatch),CAssetConverter::CHashCache::NoContentHash);
			return {newPatchIndex,true};
		}
		// Find the first node for an instance with a compatible patch and return its index
		inline patch_index_t find(const instance_t<AssetType>& instance, const CAssetConverter::patch_t<AssetType>& soloPatch) const
		{
			if (const patch_index_t* pIndex=const_cast<dfs_cache<AssetType>*>(this)->impl<false>(instance,soloPatch); pIndex)
				return *pIndex;
			return {};
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
struct input_metadata_t
{
	inline bool operator==(const input_metadata_t&) const = default;
	inline bool operator!=(const input_metadata_t&) const = default;

	explicit inline operator bool() const {return operator!=({});}

	size_t uniqueCopyGroupID = 0xdeadbeefBADC0FFEull;
	patch_index_t patchIndex = {};
};
// polymorphic
struct patched_instance_t
{
	instance_t<IAsset> instance;
	patch_index_t patchIx;
};
//
template<Asset AssetT>
class DFSVisitor
{
	protected:
		using AssetType = AssetT;

		template<Asset DepType>
		inline size_t getDependantUniqueCopyGroupID(const size_t usersGroupCopyID, const AssetType* user, const DepType* dep) const
		{
			return inputs.getDependantUniqueCopyGroupID(usersGroupCopyID,user,dep);
		}
		
		// impl
		template<Asset DepType, typename... IgnoredArgs>
		bool descend_impl(
			const instance_t<AssetType>& user, const CAssetConverter::patch_t<AssetType>& userPatch,
			const instance_t<DepType>& dep, CAssetConverter::patch_t<DepType>&& soloPatch,
			IgnoredArgs&&... ignoredArgs // DFS doesn't need to know
		)
		{
			return bool(descend_impl_impl<DepType>({user.asset,user.uniqueCopyGroupID},dep,std::move(soloPatch)));
		}

		// do nothing
		template<typename T>
		inline void nullOptional() {}

	public:
		// returns `input_metadata_t` which you can `bool(input_metadata_t)` to find out if patch was valid and merge was successful
		template<Asset DepType>
		input_metadata_t descend_impl_impl(const instance_t<IAsset>& user, const instance_t<DepType>& dep, CAssetConverter::patch_t<DepType>&& soloPatch)
		{
			// skip invalid inputs silently
			if (!soloPatch.valid(device))
			{
				inputs.logger.log(
					"Asset %p used by %p in group %d has an invalid initial patch and won't be converted!",
					system::ILogger::ELL_ERROR,dep.asset,user.asset,user.uniqueCopyGroupID
				);
				return {};
			}
			// special checks (normally the GPU object creation will fail, but these are common pitfall paths, so issue errors earlier for select problems)
			if constexpr (std::is_same_v<DepType,ICPUShader>)
			if (dep.asset->getContentType()==ICPUShader::E_CONTENT_TYPE::ECT_GLSL)
			{
				inputs.logger.log("Asset Converter doesn't support converting GLSL shaders! Asset %p won't be converted (GLSL is deprecated in Nabla)",system::ILogger::ELL_ERROR,dep.asset);
				return {};
			}
			if constexpr (std::is_same_v<DepType,ICPUBuffer>)
			if (dep.asset->getSize()>device->getPhysicalDevice()->getLimits().maxBufferSize)
			{
				inputs.logger.log(
					"Requested buffer size %zu is larger than the Physical Device's maxBufferSize Limit! Asset %p won't be converted",
					system::ILogger::ELL_ERROR,dep.asset->getSize(),dep.asset
				);
				return {};
			}
			// debug print
			inputs.logger.log("Asset (%p,%d) is used by (%p,%d)",system::ILogger::ELL_DEBUG,dep.asset,dep.uniqueCopyGroupID,user.asset,user.uniqueCopyGroupID);

			// now see if we visited already
			auto& dfsCache = std::get<dfs_cache<DepType>>(dfsCaches);
			// try to insert a new instance with a new patch, we'll be told if something mergable already existed
			auto [patchIndex,inserted] = dfsCache.insert(dep,std::move(soloPatch));
			// Only when we don't find a compatible patch entry do we carry on with the DFS
			if constexpr (asset_traits<DepType>::HasChildren)
			if (inserted)
				stack.emplace(instance_t<IAsset>{dep.asset,dep.uniqueCopyGroupID},patchIndex);
			// return the metadata
			return {.uniqueCopyGroupID=dep.uniqueCopyGroupID,.patchIndex=patchIndex};
		}

		const CAssetConverter::SInputs& inputs;
		ILogicalDevice* device;
		core::tuple_transform_t<dfs_cache,CAssetConverter::supported_asset_types>& dfsCaches;
		core::stack<patched_instance_t>& stack;
};

// go forth and find first patch that matches
class PatchOverride final : public CAssetConverter::CHashCache::IPatchOverride
{
		template<Asset AssetType>
		using lookup_t = CAssetConverter::CHashCache::lookup_t<AssetType>;
		template<Asset AssetType>
		using patch_t = CAssetConverter::patch_t<AssetType>;

		template<Asset AssetType>
		inline const patch_t<AssetType>* impl(const lookup_t<AssetType>& lookup) const
		{
			const auto& dfsCache = std::get<dfs_cache<AssetType>>(dfsCaches);
			const auto patchIx = dfsCache.find({lookup.asset,uniqueCopyGroupID},*lookup.patch);
			if (patchIx)
				return &dfsCache.nodes[patchIx.value].patch;
			assert(false); // really shouldn't happen because DFS descent should have explored ALL!
			return lookup.patch;
		}

	public:
		const CAssetConverter::SInputs& inputs;
		core::tuple_transform_t<dfs_cache,CAssetConverter::supported_asset_types>& dfsCaches;
		mutable size_t uniqueCopyGroupID;

		inline explicit PatchOverride(decltype(inputs) _inputs, decltype(dfsCaches) _dfsCaches, const size_t _uniqueCopyGroupID)
			: inputs(_inputs), dfsCaches(_dfsCaches), uniqueCopyGroupID(_uniqueCopyGroupID)
		{}


		inline const patch_t<ICPUSampler>* operator()(const lookup_t<ICPUSampler>& lookup) const override {return impl(lookup);}
		inline const patch_t<ICPUShader>* operator()(const lookup_t<ICPUShader>& lookup) const override {return impl(lookup);}
		inline const patch_t<ICPUBuffer>* operator()(const lookup_t<ICPUBuffer>& lookup) const override {return impl(lookup);}
		inline const patch_t<ICPUImage>* operator()(const lookup_t<ICPUImage>& lookup) const override {return impl(lookup);}
		inline const patch_t<ICPUBufferView>* operator()(const lookup_t<ICPUBufferView>& lookup) const override {return impl(lookup);}
		inline const patch_t<ICPUImageView>* operator()(const lookup_t<ICPUImageView>& lookup) const override {return impl(lookup);}
		inline const patch_t<ICPUPipelineLayout>* operator()(const lookup_t<ICPUPipelineLayout>& lookup) const override {return impl(lookup);}
};

template<Asset AssetT>
class HashVisit : public CAssetConverter::CHashCache::hash_impl_base
{
	public:
		using AssetType = AssetT;

	protected:
		template<Asset DepType>
		inline size_t getDependantUniqueCopyGroupID(const size_t usersGroupCopyID, const AssetType* user, const DepType* dep) const
		{
			return static_cast<const PatchOverride*>(patchOverride)->inputs.getDependantUniqueCopyGroupID(usersGroupCopyID,user,dep);
		}

		template<Asset DepType, typename... ExtraArgs>
		inline bool descend_impl(
			const instance_t<AssetType>& user, const CAssetConverter::patch_t<AssetType>& userPatch, // unused for this visit type
			const instance_t<DepType>& dep, const CAssetConverter::patch_t<DepType>& soloPatch,
			ExtraArgs&&... extraArgs // its just easier to hash most of those in `hash_impl::operator()`
		)
		{
			assert(hashCache && patchOverride);
			// find dependency compatible patch
			const CAssetConverter::patch_t<DepType>* found = patchOverride->operator()({dep.asset,&soloPatch});
			if (!found)
				return false;
			// hash dep
			static_cast<const PatchOverride*>(patchOverride)->uniqueCopyGroupID = dep.uniqueCopyGroupID;
			const auto depHash = hashCache->hash<DepType>({dep.asset,found},patchOverride,nextMistrustLevel);
			// check if hash failed
			if (depHash==CAssetConverter::CHashCache::NoContentHash)
				return false;
			// add dep hash to own
			hasher << depHash;
			// handle the few things we want to handle here
			if constexpr (sizeof...(extraArgs)>0)
			{
				auto argTuple = std::tuple<const ExtraArgs&...>(extraArgs...);
				const auto& arg0 = std::get<0>(argTuple);
				if constexpr (sizeof...(extraArgs)>1)
				{
					const auto& arg1 = std::get<1>(argTuple);
					// hash the spec info
					if constexpr (std::is_same_v<decltype(arg1),const ICPUShader::SSpecInfo&>)
					{
						hasher << arg1.entryPoint;
						for (const auto& specConstant : *arg1.entries)
						{
							hasher << specConstant.first;
							hasher.update(specConstant.second.data,specConstant.second.size);
						}
						hasher << arg1.requiredSubgroupSize;
						switch (arg0)
						{
							case IShader::E_SHADER_STAGE::ESS_COMPUTE:
								hasher << arg1.requireFullSubgroups;
								break;
							default:
								break;
						}
					}
				}
			}
			return true;
		}

		template<typename T>
		inline void nullOptional()
		{
			// just put something in the hash to denote that there's a null node in our graph
			hasher << '\0';
		}
};
bool CAssetConverter::CHashCache::hash_impl::operator()(lookup_t<ICPUSampler> lookup)
{
	auto patchedParams = lookup.asset->getParams();
	patchedParams.AnisotropicFilter = lookup.patch->anisotropyLevelLog2;
	hasher.update(&patchedParams,sizeof(patchedParams));
	return true;
}
bool CAssetConverter::CHashCache::hash_impl::operator()(lookup_t<ICPUShader> lookup)
{
	const auto* asset = lookup.asset;

	hasher << lookup.patch->stage;
	const auto type = asset->getContentType();
	hasher << type;
	// if not SPIR-V then own path matters
	if (type!=ICPUShader::E_CONTENT_TYPE::ECT_SPIRV)
		hasher << asset->getFilepathHint();
	const auto* content = asset->getContent();
	if (!content || content->getContentHash()==NoContentHash)
		return false;
	// we're not using the buffer directly, just its contents
	hasher << content->getContentHash();
	return true;
}
bool CAssetConverter::CHashCache::hash_impl::operator()(lookup_t<ICPUBuffer> lookup)
{
	auto patchedParams = lookup.asset->getCreationParams();
	assert(lookup.patch->usage.hasFlags(patchedParams.usage));
	patchedParams.usage = lookup.patch->usage;
	hasher.update(&patchedParams,sizeof(patchedParams)) << lookup.asset->getContentHash();
	return true;
}
bool CAssetConverter::CHashCache::hash_impl::operator()(lookup_t<ICPUImage> lookup)
{
	auto patchedParams = lookup.asset->getCreationParameters();
	assert(lookup.patch->usageFlags.hasFlags(patchedParams.depthUsage));
	assert(lookup.patch->stencilUsage.hasFlags(patchedParams.actualStencilUsage()));
	patchedParams.usage = lookup.patch->usageFlags;
	static_assert(false,"Need to decide how to promote formats first!");
	patchedParams.usage = lookup.patch->usageFlags;
	hasher.update(&patchedParams,sizeof(patchedParams)) << lookup.asset->getContentHash();
	return true;
}
bool CAssetConverter::CHashCache::hash_impl::operator()(lookup_t<ICPUImageView> lookup)
{
	const auto* asset = lookup.asset;
	AssetVisitor<HashVisit<ICPUImageView>> visitor = {
		*this,
		{asset,static_cast<const PatchOverride*>(patchOverride)->uniqueCopyGroupID},
		*lookup.patch
	};
	if (!visitor())
		return false;
	// NOTE: We don't hash the usage metada from the patch! Because it doesn't matter.
	// The view usage in the patch helps us propagate and patch during DFS, but no more.
	hasher << lookup.patch->subUsages;
	auto params = asset->getCreationParameters();
	hasher << params.viewType;
	if (!lookup.patch->mutatesImageFormat())
		params.format = EF_UNKNOWN;
	hasher << params.format;
	hasher.update(&params.components,sizeof(params.components));
	if (lookup.patch->fullMipChain)
		params.subresourceRange.levelCount = IGPUImageView::remaining_mip_levels;
	hasher.update(&params.subresourceRange,sizeof(params.subresourceRange));
	return true;
}
bool CAssetConverter::CHashCache::hash_impl::operator()(lookup_t<ICPUDescriptorSetLayout> lookup)
{
	const auto* asset = lookup.asset;
	// visit and hash all the immutable samplers
	AssetVisitor<HashVisit<ICPUDescriptorSetLayout>> visitor = {
		*this,
		{asset,static_cast<const PatchOverride*>(patchOverride)->uniqueCopyGroupID},
		*lookup.patch
	};
	if (!visitor())
		return false;

	using storage_range_index_t = ICPUDescriptorSetLayout::CBindingRedirect::storage_range_index_t;
	// but also need to hash the binding they are at
	const auto& immutableSamplerRedirects = asset->getImmutableSamplerRedirect();
	const auto count = immutableSamplerRedirects.getBindingCount();
	for (auto i=0u; i<count; i++)
	{
		const storage_range_index_t storageRangeIx(i);
		hasher << immutableSamplerRedirects.getBinding(storageRangeIx).data;
	}
	// then all the other bindings
	for (uint32_t t=0u; t<static_cast<uint32_t>(IDescriptor::E_TYPE::ET_COUNT); t++)
	{
		const auto type = static_cast<IDescriptor::E_TYPE>(t);
		hasher << type;
		const auto& redirect = asset->getDescriptorRedirect(type);
		const auto count = redirect.getBindingCount();
		for (auto i=0u; i<count; i++)
		{
			const storage_range_index_t storageRangeIx(i);
			hasher << redirect.getBinding(storageRangeIx).data;
			hasher << redirect.getCreateFlags(storageRangeIx);
			hasher << redirect.getStageFlags(storageRangeIx);
			hasher << redirect.getCount(storageRangeIx);
		}
	}
	return true;
}
bool CAssetConverter::CHashCache::hash_impl::operator()(lookup_t<ICPUPipelineLayout> lookup)
{
	const auto* asset = lookup.asset;
	// visit and hash all the set layouts
	AssetVisitor<HashVisit<ICPUPipelineLayout>> visitor = {
		*this,
		{asset,static_cast<const PatchOverride*>(patchOverride)->uniqueCopyGroupID},
		*lookup.patch
	};
	if (!visitor())
		return false;
	// the pc byte ranges
	hasher << std::span(lookup.patch->pushConstantBytes);
	return true;
}
bool CAssetConverter::CHashCache::hash_impl::operator()(lookup_t<ICPUPipelineCache> lookup)
{
	for (const auto& entry : lookup.asset->getEntries())
	{
		hasher << entry.first.deviceAndDriverUUID;
		if (entry.first.meta)
			hasher.update(entry.first.meta->data(),entry.first.meta->size());
	}
	hasher << lookup.asset->getContentHash();
	return true;
}
bool CAssetConverter::CHashCache::hash_impl::operator()(lookup_t<ICPUComputePipeline> lookup)
{
	const auto* asset = lookup.asset;
	//
	AssetVisitor<HashVisit<ICPUComputePipeline>> visitor = {
		*this,
		{asset,static_cast<const PatchOverride*>(patchOverride)->uniqueCopyGroupID},
		*lookup.patch
	};
	if (!visitor())
		return false;
	return true;
}
bool CAssetConverter::CHashCache::hash_impl::operator()(lookup_t<ICPURenderpass> lookup)
{
	const auto* asset = lookup.asset;

	hasher << asset->getDepthStencilAttachmentCount();
	hasher << asset->getColorAttachmentCount();
	hasher << asset->getSubpassCount();
	hasher << asset->getDependencyCount();
	hasher << asset->getViewMaskMSB();
	const ICPURenderpass::SCreationParams& params = asset->getCreationParameters();
	{
		auto hashLayout = [&](const E_FORMAT format, const IImage::SDepthStencilLayout& layout)->void
		{
			if (!isStencilOnlyFormat(format))
				hasher << layout.depth;
			if (!isDepthOnlyFormat(format))
				hasher << layout.stencil;
		};

		for (auto i=0; i<asset->getDepthStencilAttachmentCount(); i++)
		{
			auto entry = params.depthStencilAttachments[i];
			if (!entry.valid())
				return false;
			hasher << entry.format;
			hasher << entry.samples;
			hasher << entry.mayAlias;
			auto hashOp = [&](const auto& op)->void
			{
				if (!isStencilOnlyFormat(entry.format))
					hasher << op.depth;
				if (!isDepthOnlyFormat(entry.format))
					hasher << op.actualStencilOp();
			};
			hashOp(entry.loadOp);
			hashOp(entry.storeOp);
			hashLayout(entry.format,entry.initialLayout);
			hashLayout(entry.format,entry.finalLayout);
		}
		for (auto i=0; i<asset->getColorAttachmentCount(); i++)
		{
			const auto& entry = params.colorAttachments[i];
			if (!entry.valid())
				return false;
			hasher.update(&entry,sizeof(entry));
		}
		// subpasses
		using SubpassDesc = ICPURenderpass::SCreationParams::SSubpassDescription;
		auto hashDepthStencilAttachmentRef = [&](const SubpassDesc::SDepthStencilAttachmentRef& ref)->void
		{
			hasher << ref.attachmentIndex;
			hashLayout(params.depthStencilAttachments[ref.attachmentIndex].format,ref.layout);
		};
		for (auto i=0; i<asset->getSubpassCount(); i++)
		{
			const auto& entry = params.subpasses[i];
			const auto depthStencilRenderAtt = entry.depthStencilAttachment.render;
			if (depthStencilRenderAtt.used())
			{
				hashDepthStencilAttachmentRef(depthStencilRenderAtt);
				if (entry.depthStencilAttachment.resolve.used())
				{
					hashDepthStencilAttachmentRef(entry.depthStencilAttachment.resolve);
					hasher.update(&entry.depthStencilAttachment.resolveMode,sizeof(entry.depthStencilAttachment.resolveMode));
				}
			}
			else // hash needs to care about which slots go unused
				hasher << false;
			// color attachments
			for (const auto& colorAttachment : std::span(entry.colorAttachments))
			{
				if (colorAttachment.render.used())
				{
					hasher.update(&colorAttachment.render,sizeof(colorAttachment.render));
					if (colorAttachment.resolve.used())
						hasher.update(&colorAttachment.resolve,sizeof(colorAttachment.resolve));
				}
				else // hash needs to care about which slots go unused
					hasher << false;
			}
			// input attachments
			for (auto inputIt=entry.inputAttachments; *inputIt!=SubpassDesc::InputAttachmentsEnd; inputIt++)
			{
				if (inputIt->used())
				{
					hasher << inputIt->aspectMask;
                    if (inputIt->aspectMask==IImage::EAF_COLOR_BIT)
						hashDepthStencilAttachmentRef(inputIt->asDepthStencil);
					else
						hasher.update(&inputIt->asColor,sizeof(inputIt->asColor));
				}
				else
					hasher << false;
			}
			// preserve attachments
			for (auto preserveIt=entry.preserveAttachments; *preserveIt!=SubpassDesc::PreserveAttachmentsEnd; preserveIt++)
				hasher.update(preserveIt,sizeof(SubpassDesc::SPreserveAttachmentRef));
			hasher << entry.viewMask;
			hasher << entry.flags;
		}
		// TODO: we could sort these before hashing (and creating GPU objects)
		hasher.update(params.dependencies,sizeof(ICPURenderpass::SCreationParams::SSubpassDependency)*asset->getDependencyCount());
	}
	hasher.update(params.viewCorrelationGroup,sizeof(params.viewCorrelationGroup));

	return true;
}
bool CAssetConverter::CHashCache::hash_impl::operator()(lookup_t<ICPUGraphicsPipeline> lookup)
{
	const auto* asset = lookup.asset;
	//
	AssetVisitor<HashVisit<ICPUGraphicsPipeline>> visitor = {
		*this,
		{asset,static_cast<const PatchOverride*>(patchOverride)->uniqueCopyGroupID},
		*lookup.patch
	};
	if (!visitor())
		return false;

	const auto& params = asset->getCachedCreationParams();
	{
		for (auto i=0; i<SVertexInputParams::MAX_VERTEX_ATTRIB_COUNT; i++)
		if (params.vertexInput.enabledAttribFlags&(0x1u<<i))
		{
			const auto& attribute = params.vertexInput.attributes[i];
			hasher.update(&attribute,sizeof(SVertexInputAttribParams));
			hasher.update(&params.vertexInput.bindings+attribute.binding,sizeof(SVertexInputBindingParams));
		}
		const auto& ass = params.primitiveAssembly;
		hasher << ass.primitiveType;
		hasher << ass.primitiveRestartEnable;
		if (ass.primitiveType==E_PRIMITIVE_TOPOLOGY::EPT_PATCH_LIST)
			hasher << ass.tessPatchVertCount;
		const auto& raster = params.rasterization;
		if (!raster.rasterizerDiscard)
		{
			hasher << raster.viewportCount;
			hasher << raster.samplesLog2;
			hasher << raster.polygonMode;
			//if (raster.polygonMode==E_POLYGON_MODE::EPM_FILL) // do wireframes and point draw with face culling?
			{
				hasher << raster.faceCullingMode;
				hasher << raster.frontFaceIsCCW;
			}
			const auto& rpassParam = asset->getRenderpass()->getCreationParameters();
			const auto& depthStencilRef = rpassParam.subpasses[params.subpassIx].depthStencilAttachment.render;
			if (depthStencilRef.used())
			{
				const auto attFormat = rpassParam.depthStencilAttachments[depthStencilRef.attachmentIndex].format;
				if (!isStencilOnlyFormat(attFormat))
				{
					hasher << raster.depthCompareOp;
					hasher << raster.depthWriteEnable;
					if (raster.depthTestEnable())
					{
						hasher << raster.depthClampEnable;
						hasher << raster.depthBiasEnable;
						hasher << raster.depthBoundsTestEnable;
					}
				}
				if (raster.stencilTestEnable() && !isDepthOnlyFormat(attFormat))
				{
					if ((raster.faceCullingMode&E_FACE_CULL_MODE::EFCM_FRONT_BIT)==0)
						hasher << raster.frontStencilOps;
					if ((raster.faceCullingMode&E_FACE_CULL_MODE::EFCM_BACK_BIT)==0)
						hasher << raster.backStencilOps;
				}
			}
			hasher << raster.alphaToCoverageEnable;
			hasher << raster.alphaToOneEnable;
			if (raster.samplesLog2)
			{
				hasher << raster.minSampleShadingUnorm;
				hasher << (reinterpret_cast<const uint64_t&>(raster.sampleMask)&((0x1ull<<raster.samplesLog2)-1));
			}
		}
		for (const auto& blend : std::span(params.blend.blendParams))
		{
			if (blend.blendEnabled())
				hasher.update(&blend,sizeof(blend));
			else
				hasher << blend.colorWriteMask;
		}
		hasher << params.blend.logicOp;
	}
	hasher << params.subpassIx;
	return true;
}
bool CAssetConverter::CHashCache::hash_impl::operator()(lookup_t<ICPUDescriptorSet> lookup)
{
	const auto* asset = lookup.asset;
	//
	AssetVisitor<HashVisit<ICPUDescriptorSet>> visitor = {
		*this,
		{asset,static_cast<const PatchOverride*>(patchOverride)->uniqueCopyGroupID},
		*lookup.patch
	};
	if (!visitor())
		return false;
	//
	for (auto i=0u; i<static_cast<uint32_t>(IDescriptor::E_TYPE::ET_COUNT); i++)
	{
		const auto type = static_cast<IDescriptor::E_TYPE>(i);
		const auto infos = asset->getDescriptorInfoStorage(type);
		if (infos.empty())
			continue;
		for (const auto& info : infos)
		if (const auto* untypedDesc=info.desc.get(); untypedDesc)
		{
			core::blake3_hash_t descHash = NoContentHash;
			switch (IDescriptor::GetTypeCategory(type))
			{
				case IDescriptor::EC_BUFFER:
					hasher.update(&info.info.buffer,sizeof(info.info.buffer));
					break;
				case IDescriptor::EC_IMAGE:
					hasher.update(&info.info.image,sizeof(info.info.image));
					break;
				default:
					break;
			}
		}
	}
	return true;
}

void CAssetConverter::CHashCache::eraseStale(const IPatchOverride* patchOverride)
{
	auto rehash = [&]<typename AssetType>() -> void
	{
		auto& container = std::get<container_t<AssetType>>(m_containers);
		core::erase_if(container,[&](const auto& entry)->bool
			{
				// backup because `hash(lookup)` call will update it
				const auto oldHash = entry.second;
				const auto& key = entry.first;
				// can re-use cached hashes for dependants if we start ejecting in the correct order
				const auto newHash = hash<AssetType>({key.asset.get(),&key.patch},patchOverride,/*.cacheMistrustLevel = */1);
				return newHash!=oldHash || newHash==NoContentHash;
			}
		);
	};
	// to make the process more efficient we start ejecting from "lowest level" assets
	rehash.operator()<ICPUSampler>();
	rehash.operator()<ICPUDescriptorSetLayout>();
	rehash.operator()<ICPUPipelineLayout>();
	// shaders and images depend on buffers for data sourcing
	rehash.operator()<ICPUBuffer>();
	rehash.operator()<ICPUBufferView>();
	rehash.operator()<ICPUImage>();
	rehash.operator()<ICPUImageView>();
//	rehash.operator()<ICPUBottomLevelAccelerationStructure>();
//	rehash.operator()<ICPUTopLevelAccelerationStructure>();
	// only once all the descriptor types have been hashed, we can hash sets
	rehash.operator()<ICPUDescriptorSet>();
	// naturally any pipeline depends on shaders and pipeline cache
	rehash.operator()<ICPUShader>();
	rehash.operator()<ICPUPipelineCache>();
	rehash.operator()<ICPUComputePipeline>();
	// graphics pipeline needs a renderpass
	rehash.operator()<ICPURenderpass>();
	rehash.operator()<ICPUGraphicsPipeline>();
//	rehash.operator()<ICPUFramebuffer>();
}


//
template<Asset AssetT>
class GetDependantVisitBase
{
	public:
		using AssetType = AssetT;

		const CAssetConverter::SInputs& inputs;
		core::tuple_transform_t<dfs_cache,CAssetConverter::supported_asset_types>& dfsCaches;

	protected:
		template<Asset DepType>
		inline size_t getDependantUniqueCopyGroupID(const size_t usersGroupCopyID, const AssetType* user, const DepType* dep) const
		{
			return inputs.getDependantUniqueCopyGroupID(usersGroupCopyID,user,dep);
		}

		template<Asset DepType>
		asset_cached_t<DepType>::type getDependant(const instance_t<DepType>& dep, const CAssetConverter::patch_t<DepType>& soloPatch) const
		{
			const auto& dfsCache = std::get<dfs_cache<DepType>>(dfsCaches);
			// find matching patch in dfsCache
			const auto patchIx = dfsCache.find(dep,soloPatch);
			// if not found
			if (!patchIx)
				return {}; // nullptr basically
			// grab gpu object from the node of patch index
			return dfsCache.nodes[patchIx.value].gpuObj.value;
		}

		template<typename DepType>
		void nullOptional() const {}
};
template<Asset AssetType>
class GetDependantVisit;

template<>
class GetDependantVisit<ICPUBufferView> : public GetDependantVisitBase<ICPUBufferView>
{
	public:
		SBufferRange<IGPUBuffer> underlying = {};

	protected:
		bool descend_impl(
			const instance_t<AssetType>& user, const CAssetConverter::patch_t<AssetType>& userPatch,
			const instance_t<ICPUBuffer>& dep, const CAssetConverter::patch_t<ICPUBuffer>& soloPatch
		)
		{
			auto depObj = getDependant<ICPUBuffer>(dep,soloPatch);
			if (!depObj)
				return false;
			underlying = {
				.offset = user.asset->getOffsetInBuffer(),
				.size = user.asset->getByteSize(),
				.buffer = std::move(depObj)
			};
			return underlying.isValid();
		}
};
template<>
class GetDependantVisit<ICPUImageView> : public GetDependantVisitBase<ICPUImageView>
{
	public:
		core::smart_refctd_ptr<IGPUImage> image = {};

	protected:
		bool descend_impl(
			const instance_t<AssetType>& user, const CAssetConverter::patch_t<AssetType>& userPatch,
			const instance_t<ICPUImage>& dep, const CAssetConverter::patch_t<ICPUImage>& soloPatch
		)
		{
			image = getDependant<ICPUImage>(dep,soloPatch);
			return !!image;
		}
};
template<>
class GetDependantVisit<ICPUDescriptorSetLayout> : public GetDependantVisitBase<ICPUDescriptorSetLayout>
{
	public:
		core::smart_refctd_ptr<IGPUSampler>* const outImmutableSamplers;

	protected:
		bool descend_impl(
			const instance_t<AssetType>& user, const CAssetConverter::patch_t<AssetType>& userPatch,
			const instance_t<ICPUSampler>& dep, const CAssetConverter::patch_t<ICPUSampler>& soloPatch,
			const uint32_t immutableSamplerStorageOffset
		)
		{
			auto depObj = getDependant<ICPUSampler>(dep,soloPatch);
			if (!depObj)
				return false;
			outImmutableSamplers[immutableSamplerStorageOffset] = std::move(depObj);
			return true;
		}
};
template<>
class GetDependantVisit<ICPUPipelineLayout> : public GetDependantVisitBase<ICPUPipelineLayout>
{
	public:
		core::smart_refctd_ptr<IGPUDescriptorSetLayout> dsLayouts[4] = {};

	protected:
		bool descend_impl(
			const instance_t<AssetType>& user, const CAssetConverter::patch_t<AssetType>& userPatch,
			const instance_t<ICPUDescriptorSetLayout>& dep, const CAssetConverter::patch_t<ICPUDescriptorSetLayout>& soloPatch,
			const uint32_t setIndex
		)
		{
			auto depObj = getDependant<ICPUDescriptorSetLayout>(dep,soloPatch);
			if (!depObj)
				return false;
			dsLayouts[setIndex] = std::move(depObj);
			return true;
		}
};
template<Asset AssetT> requires nbl::is_any_of_v<AssetT,ICPUComputePipeline,ICPUGraphicsPipeline>
class GetDependantVisit<AssetT> : public GetDependantVisitBase<AssetT>
{
		using base_t = GetDependantVisitBase<AssetT>;

	public:
		using AssetType = AssetT;

		inline auto& getSpecInfo(const IShader::E_SHADER_STAGE stage)
		{
			assert(hlsl::bitCount(stage)==1);
			return specInfo[hlsl::findLSB(stage)];
		}

		// ok to do non owning since some cache owns anyway
		IGPUPipelineLayout* layout = nullptr;
		// has to be public to allow for initializer list constructor
		std::array<IGPUShader::SSpecInfo,/*hlsl::mpl::findMSB<ESS_COUNT>::value*/sizeof(IShader::E_SHADER_STAGE)*8> specInfo = {};
		// optionals (done this way because inheritance chain with templated class hides protected methods)
		IGPURenderpass* renderpass = nullptr;

	protected:
		bool descend_impl(
			const instance_t<AssetType>& user, const CAssetConverter::patch_t<AssetType>& userPatch,
			const instance_t<ICPUPipelineLayout>& dep, const CAssetConverter::patch_t<ICPUPipelineLayout>& soloPatch
		)
		{
			auto depObj = base_t::getDependant<ICPUPipelineLayout>(dep,soloPatch);
			if (!depObj)
				return false;
			layout = depObj.get();
			return true;
		}
		bool descend_impl(
			const instance_t<AssetType>& user, const CAssetConverter::patch_t<AssetType>& userPatch,
			const instance_t<ICPUShader>& dep, const CAssetConverter::patch_t<ICPUShader>& soloPatch,
			const IShader::E_SHADER_STAGE stage, const IShader::SSpecInfo<const ICPUShader>& inSpecInfo
		)
		{
			auto depObj = base_t::getDependant<ICPUShader>(dep,soloPatch);
			if (!depObj)
				return false;
			getSpecInfo(stage) = {
				.entryPoint = inSpecInfo.entryPoint,
				.shader = depObj.get(),
				.entries = inSpecInfo.entries,
				.requiredSubgroupSize = inSpecInfo.requiredSubgroupSize,
				.requireFullSubgroups = stage==IShader::E_SHADER_STAGE::ESS_COMPUTE ? inSpecInfo.requireFullSubgroups:uint8_t(0)
			};
			return true;
		}
		template<Asset T=AssetType> requires (std::is_same_v<T,AssetType>&& std::is_same_v<T,ICPUGraphicsPipeline>)
		bool descend_impl(
			const instance_t<T>& user, const CAssetConverter::patch_t<T>& userPatch,
			const instance_t<ICPURenderpass>& dep, const CAssetConverter::patch_t<ICPURenderpass>& soloPatch
		)
		{
			auto depObj = base_t::getDependant<ICPURenderpass>(dep,soloPatch);
			if (!depObj)
				return false;
			renderpass = depObj.get();
			return true;
		}
};
template<>
class GetDependantVisit<ICPUDescriptorSet> : public GetDependantVisitBase<ICPUDescriptorSet>
{
	public:
		// returns if there are any writes to do
		bool finalizeWrites(IGPUDescriptorSet* dstSet)
		{
			if (writes.empty())
				return false;
			// now infos can't move in memory anymore
			auto baseInfoPtr = infos.data();
			for (auto& write : writes)
			{
				write.dstSet = dstSet;
				write.info = baseInfoPtr+reinterpret_cast<const size_t&>(write.info);
			}
			return true;
		}

		core::smart_refctd_ptr<IGPUDescriptorSetLayout> layout = {};
		// okay to do non-owning, cache has ownership
		core::vector<IGPUDescriptorSet::SWriteDescriptorSet> writes = {};
		core::vector<IGPUDescriptorSet::SDescriptorInfo> infos = {};
		// has to be public because of aggregate init, but its only for internal usage!
		uint32_t lastBinding;
		uint32_t lastElement;
		// special state to pass around
		IGPUSampler* lastCombinedSampler;

	protected:
		bool descend_impl(
			const instance_t<AssetType>& user, const CAssetConverter::patch_t<AssetType>& userPatch,
			const instance_t<ICPUDescriptorSetLayout>& dep, const CAssetConverter::patch_t<ICPUDescriptorSetLayout>& soloPatch
		)
		{
			auto depObj = getDependant<ICPUDescriptorSetLayout>(dep,soloPatch);
			if (!depObj)
				return false;
			layout = std::move(depObj);
			// set initial state
			lastBinding = 0xdeadbeefu;
			lastElement = 0xdeadbeefu;
			lastCombinedSampler = nullptr;
			// could reserve `writes` and `infos`
			return true;
		}
		template<Asset DepType, typename... ExtraArgs> requires std::is_base_of_v<IDescriptor,DepType>
		bool descend_impl(
			const instance_t<AssetType>& user, const CAssetConverter::patch_t<AssetType>& userPatch,
			const instance_t<DepType>& dep, const CAssetConverter::patch_t<DepType>& soloPatch,
			const IDescriptor::E_TYPE type,
			const IDescriptorSetLayoutBase::CBindingRedirect::binding_number_t binding,
			const uint32_t element, ExtraArgs&&... extraArgs
		)
		{
			auto depObj = getDependant<DepType>(dep,soloPatch);
			if (!depObj)
				return false;
			// special path for handling combined samplers (remeber them for the image call)
			if constexpr (std::is_same_v<DepType,ICPUSampler>)
			if (type==IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER)
			{
				lastCombinedSampler = depObj.get();
				return true;
			}
			// a bit of RLE
			if (binding.data!=lastBinding || element!=(lastElement+1))
			{
				writes.push_back({
					.dstSet = nullptr, // will be patched later
					.binding = binding.data,
					.arrayElement = element,
					.count = 1,
					.info = reinterpret_cast<const IGPUDescriptorSet::SDescriptorInfo*>(infos.size()) // patch base ptr later
				});
				lastBinding = binding.data;
			}
			else
				writes.back().count++;
			lastElement = element;
			//
			auto& outInfo = infos.emplace_back();
			outInfo.desc = std::move(depObj);
			// extra stuff
			auto argTuple = std::tuple<const ExtraArgs&...>(extraArgs...);
			if constexpr (std::is_same_v<DepType,ICPUBuffer>)
			{
				if (IDescriptor::GetTypeCategory(type)==IDescriptor::E_CATEGORY::EC_BUFFER)
				{
					//outInfo.info.buffer = std::get<0>(argTuple);
					outInfo.info.buffer.offset= std::get<0>(argTuple).offset;
					outInfo.info.buffer.size = std::get<0>(argTuple).size;
				}
			}
			if constexpr (std::is_same_v<DepType,ICPUImage>)
			{
				outInfo.info.image.imageLayout = std::get<0>(argTuple);
				if (type==IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER)
				{
					assert(lastCombinedSampler);
					outInfo.info.combinedImageSampler = std::get<1>(argTuple);
				}
			}
			return true;
		}
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
auto CAssetConverter::reserve(const SInputs& inputs) -> SReserveResult
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

	SReserveResult retval = {};
	
	// this will allow us to look up the conversion parameter (actual patch for an asset) and therefore write the GPUObject to the correct place in the return value
	core::vector<input_metadata_t> inputsMetadata[core::type_list_size_v<supported_asset_types>];
	// One would think that we first need an (AssetPtr,Patch) -> ContentHash map and then a ContentHash -> GPUObj map to
	// save ourselves iterating over redundant assets. The truth is that we going from a ContentHash to GPUObj is blazing fast.
	core::tuple_transform_t<dfs_cache,supported_asset_types> dfsCaches = {};

	{
		// gather all dependencies (DFS graph search) and patch, this happens top-down
		// do not deduplicate/merge assets at this stage, only patch GPU creation parameters
		{
			// stack is nice and polymorphic
			core::stack<patched_instance_t> stack = {};

			// initialize stacks
			auto initialize = [&]<typename AssetType>(const std::span<const AssetType* const> assets)->void
			{
				const auto count = assets.size();
				const auto& patches = std::get<SInputs::patch_span_t<AssetType>>(inputs.patches);
				// size and fill the result array with nullptr
				std::get<SReserveResult::vector_t<AssetType>>(retval.m_gpuObjects).resize(count);
				// size the final patch mapping
				auto& metadata = inputsMetadata[index_of_v<AssetType,supported_asset_types>];
				metadata.resize(count);
				for (size_t i=0; i<count; i++)
				if (auto asset=assets[i]; asset) // skip invalid inputs silently
				{
					patch_t<AssetType> patch(asset);
					if (i<patches.size())
					{
						// derived patch has to be valid
						if (!patch.valid(device))
							continue;
						// the overriden one too
						auto overidepatch = patches[i];
						if (!overidepatch.valid(device))
							continue;
						// the combination must be a success (doesn't need to be valid though)
						bool combineSuccess;
						std::tie(combineSuccess,patch) = patch.combine(overidepatch);
						if (!combineSuccess)
							continue;
					}
					const size_t uniqueGroupID = inputs.getDependantUniqueCopyGroupID(0xdeadbeefBADC0FFEull,nullptr,asset);
					metadata[i] = DFSVisitor<AssetType>{
						.inputs = inputs,
						.device = device,
						.dfsCaches = dfsCaches,
						.stack = stack
					}.descend_impl_impl<AssetType>({},{asset,uniqueGroupID},std::move(patch));
				}
			};
			core::for_each_in_tuple(inputs.assets,initialize);

			// wrap in templated lambda
			auto visit = [&]<Asset AssetType>(const patched_instance_t& user)->void
			{
				// we don't use the result yet
				const bool success = AssetVisitor<DFSVisitor<AssetType>>{
					{
						.inputs = inputs,
						.device = device,
						.dfsCaches = dfsCaches,
						.stack = stack
					},
					// construct a casted instance type
					{static_cast<const AssetType*>(user.instance.asset),user.instance.uniqueCopyGroupID},
					// This is fairly risky, because its a reference to a vector element while we're pushing new elements to a vector during DFS
					// however we have a DAG and AssetType cannot depend on the same AssetType and we don't recurse inside `visit` so we never grow our own vector.
					std::get<dfs_cache<AssetType>>(dfsCaches).nodes[user.patchIx.value].patch
				}();
			};
			// Perform Depth First Search of the Asset Graph
			while (!stack.empty())
			{
				auto entry = stack.top();
				stack.pop();
				// everything we popped has already been cached in dfsCache, now time to go over dependents
				switch (entry.instance.asset->getAssetType())
				{
					case ICPUDescriptorSetLayout::AssetType:
						visit.operator()<ICPUDescriptorSetLayout>(entry);
						break;
					case ICPUPipelineLayout::AssetType:
						visit.operator()<ICPUPipelineLayout>(entry);
						break;
					case ICPUComputePipeline::AssetType:
						visit.operator()<ICPUComputePipeline>(entry);
						break;
					case ICPUGraphicsPipeline::AssetType:
						visit.operator()<ICPUGraphicsPipeline>(entry);
						break;
					case ICPUDescriptorSet::AssetType:
						visit.operator()<ICPUDescriptorSet>(entry);
						break;
					case ICPUBufferView::AssetType:
						visit.operator()<ICPUBufferView>(entry);
						break;
					case ICPUImageView::AssetType:
						visit.operator()<ICPUImageView>(entry);
						break;
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
		auto requestAllocation = [&inputs,device,&allocationRequests]<Asset AssetType>(asset_cached_t<AssetType>* pGpuObj)->bool
		{
			auto* gpuObj = pGpuObj->get();
			const IDeviceMemoryBacked::SDeviceMemoryRequirements& memReqs = gpuObj->getMemoryReqs();
			// this shouldn't be possible
			assert(memReqs.memoryTypeBits);
			// allocate right away those that need their own allocation
			if (memReqs.requiresDedicatedAllocation)
			{
				// allocate and bind right away
				auto allocation = device->allocate(memReqs,gpuObj);
				if (!allocation.isValid())
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
				if constexpr (std::is_same_v<std::remove_pointer_t<decltype(gpuObj)>,IGPUBuffer>)
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
					PatchOverride patchOverride(inputs,dfsCaches,instance.uniqueCopyGroupID);
					contentHash = retval.getHashCache()->hash<AssetType>(
						{instance.asset,&created.patch},
						&patchOverride,
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

			GetDependantVisitBase<AssetType> visitBase = {
				.inputs = inputs,
				.dfsCaches = dfsCaches
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
					const auto& patch = dfsCache.nodes[entry.second.patchIndex.value].patch;
					//
					IGPUBuffer::SCreationParams params = {};
					params.size = entry.second.canonicalAsset->getSize();
					params.usage = patch.usage;
					// concurrent ownership if any
					const auto outIx = i+entry.second.firstCopyIx;
					const auto uniqueCopyGroupID = gpuObjUniqueCopyGroupIDs[outIx];
					const auto queueFamilies =  inputs.getSharedOwnershipQueueFamilies(uniqueCopyGroupID,entry.second.canonicalAsset,patch);
					params.queueFamilyIndexCount = queueFamilies.size();
					params.queueFamilyIndices = queueFamilies.data();
					// if creation successful, we 
					if (assign(entry.first,entry.second.firstCopyIx,i,device->createBuffer(std::move(params))))
						retval.m_queueFlags |= IQueue::FAMILY_FLAGS::TRANSFER_BIT;
				}
			}
			if constexpr (std::is_same_v<AssetType,ICPUBufferView>)
			{
				for (auto& entry : conversionRequests)
				{
					const ICPUBufferView* asset = entry.second.canonicalAsset;
					const auto& patch = dfsCache.nodes[entry.second.patchIndex.value].patch;
					for (auto i=0ull; i<entry.second.copyCount; i++)
					{
						const auto outIx = i+entry.second.firstCopyIx;
						const auto uniqueCopyGroupID = gpuObjUniqueCopyGroupIDs[outIx];
						AssetVisitor<GetDependantVisit<ICPUBufferView>> visitor = {
							{visitBase},
							{asset,uniqueCopyGroupID},
							patch
						};
						if (!visitor())
							continue;
						// no format promotion for buffer views
						assign(entry.first,entry.second.firstCopyIx,i,device->createBufferView(visitor.underlying,asset->getFormat()));
					}
				}
			}
			if constexpr (std::is_same_v<AssetType,ICPUImageView>)
			{
				for (auto& entry : conversionRequests)
				{
					const ICPUImageView* asset = entry.second.canonicalAsset;
					const auto& cpuParams = asset->getCreationParameters();
					const auto& patch = dfsCache.nodes[entry.second.patchIndex.value].patch;
					for (auto i=0ull; i<entry.second.copyCount; i++)
					{
						const auto outIx = i+entry.second.firstCopyIx;
						const auto uniqueCopyGroupID = gpuObjUniqueCopyGroupIDs[outIx];
						AssetVisitor<GetDependantVisit<ICPUImageView>> visitor = {
							{visitBase},
							{asset,uniqueCopyGroupID},
							patch
						};
						if (!visitor())
							continue;
						IGPUImageView::SCreationParams params = {};
						params.flags = cpuParams.flags;
						// Only restrict ourselves to an explicit usage list if our view's format prevents all parent image's usages!
						{
							const auto& validUsages = device->getPhysicalDevice()->getImageFormatUsages(visitor.image->getTiling());
							if (!patch.getFormatUsage().isSubsetOf(validUsages[patch.format]))
								params.subUsages = patch.subUsages;
						}
						params.image = std::move(visitor.image);
						params.viewType = cpuParams.viewType;
						params.format = patch.format;
						memcpy(&params.components,&cpuParams.components,sizeof(params.components));
						params.subresourceRange = cpuParams.subresourceRange;
						if (params.subresourceRange.levelCount!=IGPUImageView::remaining_mip_levels)
							inputs.logger.log("Asset %p getting created with EXPLICIT mip level count %d, are you sure?",system::ILogger::ELL_WARNING,asset,params.subresourceRange.levelCount);
						assign(entry.first,entry.second.firstCopyIx,i,device->createImageView(std::move(params)));
					}
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
					// to let us know what binding has immutables, and set up a mapping
					core::vector<core::smart_refctd_ptr<IGPUSampler>> immutableSamplers(asset->getImmutableSamplers().size());
					{
						const auto& immutableSamplerRedirects = asset->getImmutableSamplerRedirect();
						auto outImmutableSamplers = immutableSamplers.data();
						for (auto j=0u; j<immutableSamplerRedirects.getBindingCount(); j++)
						{
							const storage_range_index_t storageRangeIx(j);
							// assuming the asset was validly created, the binding must exist
							const auto binding = immutableSamplerRedirects.getBinding(storageRangeIx);
							// TODO: optimize this, the `bindings` are sorted within a given type (can do binary search in SAMPLER and COMBINED)
							auto outBinding = std::find_if(bindings.begin(),bindings.end(),[=](const IGPUDescriptorSetLayout::SBinding& item)->bool{return item.binding==binding.data;});
							// the binding must be findable, otherwise above code logic is wrong
							assert(outBinding!=bindings.end());
							// set up the mapping
							outBinding->immutableSamplers = immutableSamplers.data()+immutableSamplerRedirects.getStorageOffset(storageRangeIx).data;
						}
					}
					//
					for (auto i=0ull; i<entry.second.copyCount; i++)
					{
						const auto outIx = i+entry.second.firstCopyIx;
						const auto uniqueCopyGroupID = gpuObjUniqueCopyGroupIDs[outIx];
						// visit the immutables, can't be factored out because depending on groupID the dependant might change
						AssetVisitor<GetDependantVisit<ICPUDescriptorSetLayout>> visitor = {
							{
								visitBase,
								immutableSamplers.data()
							},
							{asset,uniqueCopyGroupID},
							{} // no patch
						};
						if (!visitor())
							continue;
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
						pcRanges.clear();
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
						AssetVisitor<GetDependantVisit<ICPUPipelineLayout>> visitor = {
							{visitBase},
							{asset,uniqueCopyGroupID},
							patch
						};
						if (!visitor())
							continue;
						auto layout = device->createPipelineLayout(pcRanges,std::move(visitor.dsLayouts[0]),std::move(visitor.dsLayouts[1]),std::move(visitor.dsLayouts[2]),std::move(visitor.dsLayouts[3]));
						assign(entry.first,entry.second.firstCopyIx,i,std::move(layout));
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
					// there is no patching possible for this asset
					for (auto i=0ull; i<entry.second.copyCount; i++)
					{
						const auto outIx = i+entry.second.firstCopyIx;
						const auto uniqueCopyGroupID = gpuObjUniqueCopyGroupIDs[outIx];
						AssetVisitor<GetDependantVisit<ICPUComputePipeline>> visitor = {
							{visitBase},
							{asset,uniqueCopyGroupID},
							{}
						};
						if (!visitor())
							continue;
						// ILogicalDevice::createComputePipelines is rather aggressive on the spec constant validation, so we create one pipeline at a time
						core::smart_refctd_ptr<IGPUComputePipeline> ppln;
						{
							// no derivatives, special flags, etc.
							IGPUComputePipeline::SCreationParams params = {};
							params.layout = visitor.layout;
							// while there are patches possible for shaders, the only patch which can happen here is changing a stage from UNKNOWN to COMPUTE
							params.shader = visitor.getSpecInfo(IShader::E_SHADER_STAGE::ESS_COMPUTE);
							device->createComputePipelines(inputs.pipelineCache,{&params,1},&ppln);
						}
						assign(entry.first,entry.second.firstCopyIx,i,std::move(ppln));
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
						AssetVisitor<GetDependantVisit<ICPUGraphicsPipeline>> visitor = {
							{visitBase},
							{asset,uniqueCopyGroupID},
							{}
						};
						if (!visitor())
							continue;
						// ILogicalDevice::createComputePipelines is rather aggressive on the spec constant validation, so we create one pipeline at a time
						core::smart_refctd_ptr<IGPUGraphicsPipeline> ppln;
						{
							// no derivatives, special flags, etc.
							IGPUGraphicsPipeline::SCreationParams params = {};
							bool depNotFound = false;
							{
								params.layout = visitor.layout;
								params.renderpass = visitor.renderpass;
								// while there are patches possible for shaders, the only patch which can happen here is changing a stage from UNKNOWN to match the slot here
								tmpSpecInfo.clear();
								using stage_t = ICPUShader::E_SHADER_STAGE;
								for (stage_t stage : {stage_t::ESS_VERTEX,stage_t::ESS_TESSELLATION_CONTROL,stage_t::ESS_TESSELLATION_EVALUATION,stage_t::ESS_GEOMETRY,stage_t::ESS_FRAGMENT})
								{
									auto& info = visitor.getSpecInfo(stage);
									if (info.shader)
										tmpSpecInfo.push_back(std::move(info));
								}
								params.shaders = tmpSpecInfo;
							}
							params.cached = asset->getCachedCreationParams();
							device->createGraphicsPipelines(inputs.pipelineCache,{&params,1},&ppln);
							assign(entry.first,entry.second.firstCopyIx,i,std::move(ppln));
						}
					}
				}
			}
			if constexpr (std::is_same_v<AssetType,ICPUDescriptorSet>)
			{
				// Why we're not grouping multiple descriptor sets into few pools and doing 1 pool per descriptor set.
				// Descriptor Pools have large up-front slots reserved for all descriptor types, if we were to merge 
				// multiple descriptor sets to be allocated from one pool, dropping any set wouldn't result in the
				// reclamation of the memory used, it would at most (with the FREE pool create flag) return to pool. 
				for (auto& entry : conversionRequests)
				{
					const ICPUDescriptorSet* asset = entry.second.canonicalAsset;
					for (auto i=0ull; i<entry.second.copyCount; i++)
					{
						const auto outIx = i+entry.second.firstCopyIx;
						const auto uniqueCopyGroupID = gpuObjUniqueCopyGroupIDs[outIx];
						AssetVisitor<GetDependantVisit<ICPUDescriptorSet>> visitor = {
							{visitBase},
							{asset,uniqueCopyGroupID},
							{}
						};
						if (!visitor())
							continue;
						const auto* layout = visitor.layout.get();
						const bool hasUpdateAfterBind = layout->needUpdateAfterBindPool();
						using pool_flags_t = IDescriptorPool::E_CREATE_FLAGS;
						auto pool = device->createDescriptorPoolForDSLayouts(
							hasUpdateAfterBind ? pool_flags_t::ECF_UPDATE_AFTER_BIND_BIT:pool_flags_t::ECF_NONE,{&layout,1}
						);
						core::smart_refctd_ptr<IGPUDescriptorSet> ds;
						if (pool)
						{
							ds = pool->createDescriptorSet(std::move(visitor.layout));
							if (ds && visitor.finalizeWrites(ds.get()) && !device->updateDescriptorSets(visitor.writes,{}))
							{
								inputs.logger.log("Failed to write Descriptors into Descriptor Set's bindings!",system::ILogger::ELL_ERROR);
								// fail
								ds = nullptr;
							}
						}
						else
							inputs.logger.log("Failed to create Descriptor Pool suited for Layout %s",system::ILogger::ELL_ERROR,layout->getObjectDebugName());
						assign(entry.first,entry.second.firstCopyIx,i,std::move(ds));
					}
				}
			}

			// Propagate the results back, since the dfsCache has the original asset pointers as keys, we map in reverse
			auto& stagingCache = std::get<SReserveResult::staging_cache_t<AssetType>>(retval.m_stagingCaches);
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
					if constexpr (std::is_base_of_v<IDeviceMemoryBacked,typename asset_traits<AssetType>::video_t>)
					{
						if (!requestAllocation(&created.gpuObj))
						{
							created.gpuObj.value = nullptr;
							return;
						}
					}
					//
					if constexpr (has_type<AssetType>(SReserveResult::convertible_asset_types{}))
					{
						auto& requests = std::get<SReserveResult::conversion_requests_t<ICPUBuffer>>(retval.m_conversionRequests);
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
						return lhsWorstSize>rhsWorstSize;
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
					auto& binItems = reqBin.second;
					const auto binItemCount = reqBin.second.size();
					if (!binItemCount)
						continue;

					// the `std::exclusive_scan` syntax is more effort for this
					{
						offsetsTmp.resize(binItemCount);
						offsetsTmp[0] = 0;
						for (size_t i=0; true;)
						{
							const auto* memBacked = getAsBase(binItems[i]);
							const auto& memReqs = memBacked->getMemoryReqs();
							// round up the offset to get the correct alignment
							offsetsTmp[i] = core::roundUp(offsetsTmp[i],0x1ull<<memReqs.alignmentLog2);
							// record next offset
							if (i<binItemCount-1)
								offsetsTmp[++i] = offsetsTmp[i]+memReqs.size;
							else
								break;
						}
					}
					// to replace
					core::vector<memory_backed_ptr_variant_t> failures;
					failures.reserve(binItemCount);
					// ...
					using allocate_flags_t = IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS;
					IDeviceMemoryAllocator::SAllocateInfo info = {
						.size = 0xdeadbeefBADC0FFEull, // set later
						.flags = reqBin.first.needsDeviceAddress ? allocate_flags_t::EMAF_DEVICE_ADDRESS_BIT:allocate_flags_t::EMAF_NONE,
						.memoryTypeIndex = memTypeIx,
						.dedication = nullptr
					};
					// allocate in progression of combined allocations, while trying allocate as much as possible in a single allocation
					auto binItemsIt = binItems.begin();
					for (auto firstOffsetIt=offsetsTmp.begin(); firstOffsetIt!=offsetsTmp.end(); )
					for (auto nextOffsetIt=offsetsTmp.end(); nextOffsetIt>firstOffsetIt; nextOffsetIt--)
					{
						const size_t combinedCount = std::distance(firstOffsetIt,nextOffsetIt);
						const size_t lastIx = combinedCount-1;
						// if we take `combinedCount` starting at `firstItem` their allocation would need this size
						info.size = (firstOffsetIt[lastIx]-*firstOffsetIt)+getAsBase(binItemsIt[lastIx])->getMemoryReqs().size;
						auto allocation = device->allocate(info);
						if (allocation.isValid())
						{
							// bind everything
							for (auto i=0; i<combinedCount; i++)
							{
								const auto& toBind = binItems[i];
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
							// move onto next batch
							firstOffsetIt = nextOffsetIt;
							binItemsIt += combinedCount;
							break;
						}
						// we're unable to allocate even for a single item with a dedicated allocation, skip trying then
						else if (combinedCount==1)
						{
							firstOffsetIt = nextOffsetIt;
							failures.push_back(std::move(*binItemsIt));
							binItemsIt++;
							break;
						}
					}
					// leave only the failures behind
					binItems = std::move(failures);
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
		dedupCreateProp.operator()<ICPUBufferView>();
		dedupCreateProp.operator()<ICPUShader>();
		dedupCreateProp.operator()<ICPUSampler>();
		dedupCreateProp.operator()<ICPUDescriptorSetLayout>();
		dedupCreateProp.operator()<ICPUPipelineLayout>();
		dedupCreateProp.operator()<ICPUPipelineCache>();
		dedupCreateProp.operator()<ICPUComputePipeline>();
		dedupCreateProp.operator()<ICPURenderpass>();
		dedupCreateProp.operator()<ICPUGraphicsPipeline>();
		dedupCreateProp.operator()<ICPUDescriptorSet>();
//		dedupCreateProp.operator()<ICPUFramebuffer>();
	}

	// write out results
	auto finalize = [&]<typename AssetType>(const std::span<const AssetType* const> assets)->void
	{
		const auto count = assets.size();
		//
		const auto& metadata = inputsMetadata[index_of_v<AssetType,supported_asset_types>];
		const auto& dfsCache = std::get<dfs_cache<AssetType>>(dfsCaches);
		const auto& stagingCache = std::get<SReserveResult::staging_cache_t<AssetType>>(retval.m_stagingCaches);
		auto& results = std::get<SReserveResult::vector_t<AssetType>>(retval.m_gpuObjects);
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
auto CAssetConverter::convert_impl(SReserveResult&& reservations, SConvertParams& params) -> SReserveResult::SConvertResult
{
	if (!reservations.m_converter)
	{
		reservations.m_logger.log("Cannot call convert on an unsuccessful reserve result!",system::ILogger::ELL_ERROR);
		return {};
	}
	assert(reservations.m_converter.get()==this);
	auto device = m_params.device;
	const auto reqQueueFlags = reservations.getRequiredQueueFlags();

	SReserveResult::SConvertResult retval = {};
	// Anything to do?
	if (reqQueueFlags.value!=IQueue::FAMILY_FLAGS::NONE)
	{
		if (reqQueueFlags.hasFlags(IQueue::FAMILY_FLAGS::TRANSFER_BIT) && (!params.utilities || params.utilities->getLogicalDevice()!=device))
		{
			reservations.m_logger.log("Transfer Capability required for this conversion and no compatible `utilities` provided!", system::ILogger::ELL_ERROR);
			return {};
		}

		auto invalidQueue = [reqQueueFlags,device,&reservations](const IQueue::FAMILY_FLAGS flag, IQueue* queue)->bool
		{
			if (!reqQueueFlags.hasFlags(flag))
				return false;
			if (!queue || queue->getOriginDevice()!=device)
			{
				reservations.m_logger.log("Provided Queue;s device %p doesn't match CAssetConverter's device %p!",system::ILogger::ELL_ERROR,queue->getOriginDevice(),device);
				return true;
			}
			const auto& qFamProps = device->getPhysicalDevice()->getQueueFamilyProperties();
			if (!qFamProps[queue->getFamilyIndex()].queueFlags.hasFlags(flag))
			{
				reservations.m_logger.log("Provided Queue %p in Family %d does not have the required capabilities %d!",system::ILogger::ELL_ERROR,queue,queue->getFamilyIndex(),flag);
				return true;
			}
			return false;
		};
		// If the transfer queue will be used, the transfer Intended Submit Info must be valid and utilities must be provided
		if (invalidQueue(IQueue::FAMILY_FLAGS::TRANSFER_BIT,params.transfer.queue))
			return {};
		// If the compute queue will be used, the compute Intended Submit Info must be valid and utilities must be provided
		if (invalidQueue(IQueue::FAMILY_FLAGS::COMPUTE_BIT,params.transfer.queue))
			return {};

		// weak patch
		auto condBeginCmdBuf = [](IGPUCommandBuffer* cmdbuf)->void
		{
			if (cmdbuf)
			switch (cmdbuf->getState())
			{
				case IGPUCommandBuffer::STATE::INITIAL:
				case IGPUCommandBuffer::STATE::INVALID:
					if (cmdbuf->isResettable() && cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT))
						break;
					break;
				default:
					break;
			}
		};
		if (reqQueueFlags.hasFlags(IQueue::FAMILY_FLAGS::TRANSFER_BIT))
			condBeginCmdBuf(params.transfer.getScratchCommandBuffer());
		if (reqQueueFlags.hasFlags(IQueue::FAMILY_FLAGS::COMPUTE_BIT))
			condBeginCmdBuf(params.compute.getScratchCommandBuffer());

		// wipe gpu item in staging cache (this may drop it as well if it was made for only a root asset == no users)
		auto makFailureInStaging = [&]<Asset AssetType>(asset_traits<AssetType>::video_t* gpuObj)->void
		{
			auto& stagingCache = std::get<SReserveResult::staging_cache_t<AssetType>>(reservations.m_stagingCaches);
			const auto found = stagingCache.find(gpuObj);
			assert(found!=stagingCache.end());
			// change the content hash on the reverse map to a NoContentHash
			const_cast<core::blake3_hash_t&>(found->second.value) = {};
		};

		// upload Buffers
		auto& buffersToUpload = std::get<SReserveResult::conversion_requests_t<ICPUBuffer>>(reservations.m_conversionRequests);
		{
			core::vector<IGPUCommandBuffer::SBufferMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier>> ownershipTransfers;
			ownershipTransfers.reserve(buffersToUpload.size());
			// do the uploads
			for (auto& item : buffersToUpload)
			{
				auto found = std::get<SReserveResult::staging_cache_t<ICPUBuffer>>(reservations.m_stagingCaches).find(item.gpuObj);
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
				retval.submitsNeeded |= IQueue::FAMILY_FLAGS::TRANSFER_BIT;
				// enqueue ownership release if necessary
				if (const auto ownerQueueFamily=params.getFinalOwnerQueueFamily(item.gpuObj,{}); ownerQueueFamily!=IQueue::FamilyIgnored)
				{
					// silently skip ownership transfer
					if (item.gpuObj->getCachedCreationParams().isConcurrentSharing())
					{
						reservations.m_logger.log("Buffer %s created with concurrent sharing, you cannot perform an ownership transfer on it!",system::ILogger::ELL_ERROR,item.gpuObj->getObjectDebugName());
						continue;
					}
					// we already own
					if (params.transfer.queue->getFamilyIndex()==ownerQueueFamily)
						continue;
					// else record our half of the ownership transfer 
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
			if (!ownershipTransfers.empty())
			if (!params.transfer.getScratchCommandBuffer()->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE,{.memBarriers={},.bufBarriers=ownershipTransfers}))
				reservations.m_logger.log("Ownership Releases of Buffers Failed",system::ILogger::ELL_ERROR);
		}

#if 0
		auto& imagesToUpload = std::get<SReserveResult::conversion_requests_t<ICPUImage>>(reservations.m_conversionRequests);
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
		auto& stagingCache = std::get<SReserveResult::staging_cache_t<AssetType>>(reservations.m_stagingCaches);
		auto found = stagingCache.find(const_cast<asset_traits<AssetType>::video_t*>(dep));
		if (found!=stagingCache.end() && found->second.value==core::blake3_hash_t{})
			return true;
		// dependent might be in readCache of one or more converters, so if in doubt assume its okay
		return false;
	};
	// insert items into cache if overflows handled fine and commandbuffers ready to be recorded
	auto mergeCache = [&]<Asset AssetType>()->void
	{
		auto& stagingCache = std::get<SReserveResult::staging_cache_t<AssetType>>(reservations.m_stagingCaches);
		auto& cache = std::get<CCache<AssetType>>(m_caches);
		cache.m_forwardMap.reserve(cache.m_forwardMap.size()+stagingCache.size());
		cache.m_reverseMap.reserve(cache.m_reverseMap.size()+stagingCache.size());
		for (auto& item : stagingCache)
		if (item.second.value!=core::blake3_hash_t{}) // didn't get wiped
		{
			// rescan all the GPU objects and find out if they depend on anything that failed, if so add to failure set
			bool depsMissing = false;
			// only go over types we could actually break via missing upload/build (i.e. pipelines are unbreakable)
			if constexpr (std::is_same_v<AssetType,ICPUBufferView>)
				depsMissing = missingDependent.operator()<ICPUBuffer>(item.first->getUnderlyingBuffer());
			if constexpr (std::is_same_v<AssetType,ICPUImageView>)
				depsMissing = missingDependent.operator()<ICPUImage>(item.first->getCreationParams().image);
			if constexpr (std::is_same_v<AssetType,ICPUDescriptorSet>)
			{
				const IGPUDescriptorSetLayout* layout = item.first->getLayout();
				// check samplers
				{
					const auto count = layout->getTotalMutableCombinedSamplerCount();
					const auto* samplers = item.first->getAllMutableCombinedSamplers();
					for (auto i=0u; !depsMissing && i<count; i++)
					if (samplers[i])
						depsMissing = missingDependent.operator()<ICPUSampler>(samplers[i].get());
				}
				for (auto i=0u; !depsMissing && i<static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT); i++)
				{
					const auto type = static_cast<asset::IDescriptor::E_TYPE>(i);
					const auto count = layout->getTotalDescriptorCount(type);
					auto* psDescriptors = item.first->getAllDescriptors(type);
					if (!psDescriptors)
						continue;
					for (auto i=0u; !depsMissing && i<count; i++)
					{
						auto* untypedDesc = psDescriptors[i].get();
						if (untypedDesc)
						switch (asset::IDescriptor::GetTypeCategory(type))
						{
							case asset::IDescriptor::EC_BUFFER:
								depsMissing = missingDependent.operator()<ICPUBuffer>(static_cast<const IGPUBuffer*>(untypedDesc));
								break;
							case asset::IDescriptor::EC_SAMPLER:
								depsMissing = missingDependent.operator()<ICPUSampler>(static_cast<const IGPUSampler*>(untypedDesc));
								break;
							case asset::IDescriptor::EC_IMAGE:
//								depsMissing = missingDependent.operator()<ICPUImage>(static_cast<const IGPUImageView*>(untypedDesc));
								break;
							case asset::IDescriptor::EC_BUFFER_VIEW:
								depsMissing = missingDependent.operator()<ICPUBufferView>(static_cast<const IGPUBufferView*>(untypedDesc));
								break;
							case asset::IDescriptor::EC_ACCELERATION_STRUCTURE:
								_NBL_TODO();
								[[fallthrough]];
							default:
								assert(false);
								depsMissing = true;
								break;
						}
					}
				}
			}
			if (depsMissing)
			{
				const auto* hashAsU64 = reinterpret_cast<const uint64_t*>(item.second.value.data);
				reservations.m_logger.log("GPU Obj %s not writing to final cache because conversion of a dependant failed!", system::ILogger::ELL_ERROR, item.first->getObjectDebugName());
				// wipe self, to let users know
				item.second.value = {};
				continue;
			}
			if (!params.writeCache(item.second))
				continue;
			asset_cached_t<AssetType> cached;
			cached.value = core::smart_refctd_ptr<typename asset_traits<AssetType>::video_t>(item.first);
			cache.m_forwardMap.emplace(item.second,std::move(cached));
			cache.m_reverseMap.emplace(item.first,item.second);
		}
	};
	// again, need to go bottom up so we can check dependencies being successes
	mergeCache.operator()<ICPUBuffer>();
//	mergeCache.operator()<ICPUImage>();
//	mergeCache.operator()<ICPUBottomLevelAccelerationStructure>();
//	mergeCache.operator()<ICPUTopLevelAccelerationStructure>();
	mergeCache.operator()<ICPUBufferView>();
	mergeCache.operator()<ICPUShader>();
	mergeCache.operator()<ICPUSampler>();
	mergeCache.operator()<ICPUDescriptorSetLayout>();
	mergeCache.operator()<ICPUPipelineLayout>();
	mergeCache.operator()<ICPUPipelineCache>();
	mergeCache.operator()<ICPUComputePipeline>();
	mergeCache.operator()<ICPURenderpass>();
	mergeCache.operator()<ICPUGraphicsPipeline>();
	mergeCache.operator()<ICPUDescriptorSet>();
//	mergeCache.operator()<ICPUFramebuffer>();

	// to make it valid
	retval.device = device;
	retval.transfer = &params.transfer;
	retval.compute = &params.compute;
	return retval;
}

}
}