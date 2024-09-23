// Copyright (C) 2024-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
#ifndef _NBL_VIDEO_C_ASSET_CONVERTER_INCLUDED_
#define _NBL_VIDEO_C_ASSET_CONVERTER_INCLUDED_


#include "nbl/video/utilities/IUtilities.h"
#include "nbl/video/asset_traits.h"
#include "nbl/builtin/hlsl/cpp_compat.hlsl"


namespace nbl::video
{
/*
* This whole class assumes all assets you are converting will be used in read-only mode by the Device.
* It's a valid assumption for everything from pipelines to shaders, but not for descriptors (with exception of samplers) and their sets.
* 
* Only Descriptors (by proxy their backing objects) and their Sets can have their contents changed after creation.
* 
* What this converter does is it computes hashes and compares equality based on the contents of an IAsset, not the pointer!
* With some sane limits, its not going to compare the contents of an ICPUImage or ICPUBuffer.
* 
* Therefore if you don't want some resource to be deduplicated you need to "explicitly" let us know via `SInputs::getDependantUniqueCopyGroupID`.
*/
class CAssetConverter : public core::IReferenceCounted
{
	public:
		// Shader, DSLayout, PipelineLayout, Compute Pipeline
		// Renderpass, Graphics Pipeline
		// Buffer, BufferView, Sampler, Image, Image View, Bottom Level AS, Top Level AS, Descriptor Set, Framebuffer  
		// Buffer -> SRange, patched usage, owner(s)
		// BufferView -> SRange, promoted format
		// Sampler -> Clamped Params (only aniso, really)
		// Image -> this, patched usage, promoted format
		// Image View -> ref to patched Image, patched usage, promoted format
		// Descriptor Set -> unique layout, 
		using supported_asset_types = core::type_list<
			asset::ICPUSampler,
			asset::ICPUShader,
			asset::ICPUBuffer,
			// acceleration structures,
			asset::ICPUImage,
			asset::ICPUBufferView,
			asset::ICPUImageView,
			asset::ICPUDescriptorSetLayout,
			asset::ICPUPipelineLayout,
			asset::ICPUPipelineCache,
			asset::ICPUComputePipeline,
			asset::ICPURenderpass,
			asset::ICPUGraphicsPipeline,
			asset::ICPUDescriptorSet
			//asset::ICPUFramebuffer doesn't exist yet XD
		>;

		struct SCreationParams
		{
			inline bool valid() const
			{
				if (!device)
					return false;

				return true;
			}

			// required not null
			ILogicalDevice* device = nullptr;
			// optional
			core::smart_refctd_ptr<const asset::ISPIRVOptimizer> optimizer = {};
		};
		static inline core::smart_refctd_ptr<CAssetConverter> create(SCreationParams&& params)
		{
			if (!params.valid())
				return nullptr;
			return core::smart_refctd_ptr<CAssetConverter>(new CAssetConverter(std::move(params)),core::dont_grab);
		}
#define NBL_API
		// When getting dependents, the creation parameters of GPU objects will be produced and patched appropriately.
		// `patch_t` uses CRTP to inherit from `patch_impl_t` to provide default `operator==` and `update_hash()` definition.
		// The default specialization kicks in for any `AssetType` that has nothing possible to patch (e.g. Descriptor Set Layout).
		template<asset::Asset AssetType>
		struct patch_impl_t
		{
#define PATCH_IMPL_BOILERPLATE(ASSET_TYPE) using this_t = patch_impl_t<ASSET_TYPE>; \
			public: \
				inline patch_impl_t() = default; \
				inline patch_impl_t(const this_t& other) = default; \
				inline patch_impl_t(this_t&& other) = default; \
				inline this_t& operator=(const this_t& other) = default; \
				inline this_t& operator=(this_t&& other) = default; \
				patch_impl_t(const ASSET_TYPE* asset); \
				bool valid(const ILogicalDevice* device)

				PATCH_IMPL_BOILERPLATE(AssetType);

			protected:
				// there's nothing to combine, so combining always produces the input successfully
				inline std::pair<bool,this_t> combine(const this_t& other) const
				{
					return {true,*this};
				}
		};
		template<>
		struct patch_impl_t<asset::ICPUSampler>
		{
			public:
				PATCH_IMPL_BOILERPLATE(asset::ICPUSampler);

				uint8_t anisotropyLevelLog2 = 6;
				
			protected:
				inline std::pair<bool,this_t> combine(const this_t& other) const
				{
					// The only reason why someone would have a different level to creation parameters is
					// because the HW doesn't support that level and the level gets clamped. So must be same.
					if (anisotropyLevelLog2!=other.anisotropyLevelLog2)
						return {false,{}}; // invalid
					return {true,*this};
				}
		};
		template<>
		struct patch_impl_t<asset::ICPUShader>
		{
			public:
				PATCH_IMPL_BOILERPLATE(asset::ICPUShader);

				using shader_stage_t = asset::IShader::E_SHADER_STAGE;
				shader_stage_t stage = shader_stage_t::ESS_UNKNOWN;

			protected:
				inline std::pair<bool,this_t> combine(const this_t& other) const
				{
					// because of the assumption that we'll only be combining valid patches, we can't have the stages differ
					return {stage==other.stage,*this};
				}
		};
		template<>
		struct patch_impl_t<asset::ICPUBuffer>
		{
			public:
				PATCH_IMPL_BOILERPLATE(asset::ICPUBuffer);

				using usage_flags_t = IGPUBuffer::E_USAGE_FLAGS;
				core::bitflag<usage_flags_t> usage = usage_flags_t::EUF_NONE;

			protected:
				inline std::pair<bool,this_t> combine(const this_t& other) const
				{
					this_t retval = *this;
					retval.usage |= other.usage;
					return {true,retval};
				}
		};
		template<>
		struct patch_impl_t<asset::ICPUImage>
		{
			public:
				PATCH_IMPL_BOILERPLATE(asset::ICPUImage);

				// make our promotion policy explicit
				inline bool canAttemptFormatPromotion() const
				{
					// if there exist views of the image that reinterpret cast its texel blocks, stop promotion, aliasing can't work with promotion!
					if (mutableFormat)
						return false;
					// we don't support promoting formats in renderpasses' attachment descriptions, so stop it here too
					if (!usageFlags.hasFlags(IGPUImage::E_USAGE_FLAGS::EUF_RENDER_ATTACHMENT_BIT))
						return false;
					if (!stencilUsage.hasFlags(IGPUImage::E_USAGE_FLAGS::EUF_RENDER_ATTACHMENT_BIT))
						return false;
					return true;
				}

				// the most important thing about an image
				asset::E_FORMAT format = asset::EF_UNKNOWN;
				// but we also track separate dpeth and stencil usage flags
				using usage_flags_t = IGPUImage::E_USAGE_FLAGS;
				core::bitflag<usage_flags_t> usageFlags = usage_flags_t::EUF_NONE;
				core::bitflag<usage_flags_t> stencilUsage = usage_flags_t::EUF_NONE;
				// all converted images default to optimal!
				uint16_t linearTiling : 1 = false;
				// moar stuff
				uint16_t mutableFormat : 1 = false;
				uint16_t cubeCompatible : 1 = false;
				uint16_t _3Dbut2DArrayCompatible : 1 = false;
				// we sort of ignore that at the end if the format doesn't stay block compressed
				uint16_t uncompressedViewOfCompressed : 1 = false;
				// Extra metadata needed for format promotion, if you want any of them (except for `linearlySampled` and `depthCompareSampledImage`)
				// as anything other than the default values, use explicit input roots with patches. Otherwise if `format` is not supported by device
				// the view can get promoted to a format that doesn't have these usage capabilities.
				uint16_t linearlySampled : 1 = false;
				uint16_t storageAtomic : 1 = false;
				uint16_t storageImageLoadWithoutFormat : 1 = false;
				uint16_t depthCompareSampledImage : 1 = false;
				// aside from format promotion, we can also promote images to have a fuller mip chain and recompute it
				uint16_t mipLevels : 6 = 0;
				uint16_t recomputeMips : 1 = false;

			protected:
				inline std::pair<bool,this_t> combine(const this_t& other) const
				{
					this_t retval = *this;
					if (canAttemptFormatPromotion())
					{
						// promoted format patch `this` needs to merge successfully with unpromoted `other`
						retval.format = format;
// TODO: HUGE PROBLEM! What if due to different usages `this` and `other` get promoted to different formats during merge? (they've both had `valid()` called on them!)
					}
					else if (format!=format)
						return {false,retval};
					//
					usageFlags |= other.usageFlags;
					return {true,retval};
				}
		};
		template<>
		struct patch_impl_t<asset::ICPUBufferView>
		{
			public:
				PATCH_IMPL_BOILERPLATE(asset::ICPUBufferView);

				uint8_t stbo : 1 = false;
				uint8_t utbo : 1 = false;
				uint8_t mustBeZero : 6 = 0;

			protected:
				inline std::pair<bool,this_t> combine(const this_t& other) const
				{
					this_t retval = *this;
					retval.stbo |= other.stbo;
					retval.utbo |= other.utbo;
					return {true,retval};
				}
		};
		template<>
		struct patch_impl_t<asset::ICPUImageView>
		{
			private:
				using this_t = patch_impl_t<asset::ICPUImageView>;

			public:
				inline patch_impl_t() = default;
				inline patch_impl_t(const this_t& other) = default;
				inline patch_impl_t(this_t&& other) = default;
				inline this_t& operator=(const this_t& other) = default;
				inline this_t& operator=(this_t&& other) = default;

				using usage_flags_t = IGPUImage::E_USAGE_FLAGS;
				// slightly weird constructor because it deduces the metadata from subusages, so need the subusages right away, not patched later
				NBL_API patch_impl_t(const asset::ICPUImageView* asset, const core::bitflag<usage_flags_t> extraSubUsages=usage_flags_t::EUF_NONE);

				NBL_API bool valid(const ILogicalDevice* device);

				//
				inline bool formatFollowsImage() const
				{
					return originalFormat==asset::EF_UNKNOWN;
				}

				// just because we record all subusages we can find, doesn't mean we will set them on the created image
				core::bitflag<usage_flags_t> subUsages = usage_flags_t::EUF_NONE;
				// Extra metadata needed for format promotion, if you want any of them (except for `linearlySampled` and `depthCompareSampledImage`)
				// as anything other than the default values, use explicit input roots with patches. Otherwise if `format` is not supported by device
				// the view can get promoted to a format that doesn't have these usage capabilities.
				uint8_t linearlySampled : 1 = false;
				uint8_t storageAtomic : 1 = false;
				uint8_t storageImageLoadWithoutFormat : 1 = false;
				uint8_t depthCompareSampledImage : 1 = false;
				// whether to override and extend the mip-chain fully
				uint8_t forceFullMipChain : 1 = false;

			protected:
				uint8_t invalid : 1 = false;
				// to not mess with hashing and comparison
				uint8_t padding : 2 = 0;
				// normally wouldn't store that but we don't provide a ref/pointer to the asset when combining or checking validity, treat member as impl detail
				asset::E_FORMAT originalFormat = asset::EF_UNKNOWN;

				inline std::pair<bool,this_t> combine(const this_t& other) const
				{
					assert(padding==0);
					if (invalid || other.invalid)
						return {false,*this};

					this_t retval = *this;
					// So we have two patches of the same image view, ergo they were the same format.
					// If one mutates and other doesn't its because of added usages that preclude, so make us immutable again.
					if (formatFollowsImage() && !other.formatFollowsImage())
						retval.originalFormat = other.originalFormat;
					// When combining usages, we already:
					// - require that two patches' formats were identical
					// - require that each patch be valid in on its own
					// therefore both patches' usages are valid for the format at the time of combining
					retval.subUsages |= other.subUsages;
					retval.linearlySampled |= other.linearlySampled;
					retval.storageAtomic |= other.storageAtomic;
					retval.storageImageLoadWithoutFormat |= other.storageImageLoadWithoutFormat;
					retval.depthCompareSampledImage |= other.depthCompareSampledImage;
					retval.forceFullMipChain |= other.forceFullMipChain;
					return {true,retval};
				}
		};
		template<>
		struct patch_impl_t<asset::ICPUPipelineLayout>
		{
			public:
				PATCH_IMPL_BOILERPLATE(asset::ICPUPipelineLayout);

				using shader_stage_t = asset::IShader::E_SHADER_STAGE;
				std::array<core::bitflag<shader_stage_t>,asset::CSPIRVIntrospector::MaxPushConstantsSize> pushConstantBytes = {shader_stage_t::ESS_UNKNOWN};
				
			protected:
				inline std::pair<bool,this_t> combine(const this_t& other) const
				{
					if (invalid || other.invalid)
						return {false,*this};
					this_t retval = *this;
					for (auto byte=0; byte!=pushConstantBytes.size(); byte++)
						retval.pushConstantBytes[byte] |= other.pushConstantBytes[byte];
					return {true,retval};
				}

				bool invalid = true;
		};
#undef PATCH_IMPL_BOILERPLATE
		// The default specialization provides simple equality operations and hash operations, this will work as long as your patch_impl_t doesn't:
		// - use a container like `core::vector<T>`, etc.
		// - use pointers to other objects or arrays whose contents must be analyzed
		template<asset::Asset AssetType>
		struct patch_t final : patch_impl_t<AssetType>
		{
			using this_t = patch_t<AssetType>;
			using base_t = patch_impl_t<AssetType>;

			// forwarding
			using base_t::base_t;
			inline patch_t(const this_t& other) : base_t(other) {}
			inline patch_t(this_t&& other) : base_t(std::move(other)) {}
			inline patch_t(base_t&& other) : base_t(std::move(other)) {}

			inline this_t& operator=(const this_t& other) = default;
			inline this_t& operator=(this_t&& other) = default;

			// The assumption is we'll only ever be combining valid patches together.
			// Returns: whether the combine op was a success, DOESN'T MEAN the result is VALID!
			inline std::pair<bool,this_t> combine(const this_t& other) const
			{
				//assert(base_t::valid() && other.valid());
				return base_t::combine(other);
			}

			// actual new methods
			inline bool operator==(const patch_t<AssetType>& other) const
			{
				if (std::is_empty_v<base_t>)
					return true; 
				return memcmp(this,&other,sizeof(base_t))==0;
			}
		};
		// A class to accelerate our hash computations
		class CHashCache final : public core::IReferenceCounted
		{
			public:
				//
				template<asset::Asset AssetType>
				struct lookup_t
				{
					const AssetType* asset = nullptr;
					const patch_t<AssetType>* patch = {};
				};

			private:
				//
				template<asset::Asset AssetType>
				struct key_t
				{
					core::smart_refctd_ptr<const AssetType> asset = {};
					patch_t<AssetType> patch = {};
				};
				template<asset::Asset AssetType>
				struct HashEquals
				{
					using is_transparent = void;

					inline size_t operator()(const key_t<AssetType>& key) const
					{
						return operator()(lookup_t<AssetType>{key.asset.get(),&key.patch});
					}
					inline size_t operator()(const lookup_t<AssetType>& lookup) const
					{
						core::blake3_hasher hasher;
						hasher << ptrdiff_t(lookup.asset);
						hasher << *lookup.patch;
						// put long hash inside a small hash
						return std::hash<core::blake3_hash_t>()(static_cast<core::blake3_hash_t>(hasher));
					}

					inline bool operator()(const key_t<AssetType>& lhs, const key_t<AssetType>& rhs) const
					{
						return lhs.asset.get()==rhs.asset.get() && lhs.patch==rhs.patch;
					}
					inline bool operator()(const key_t<AssetType>& lhs, const lookup_t<AssetType>& rhs) const
					{
						return lhs.asset.get()==rhs.asset && rhs.patch && lhs.patch==*rhs.patch;
					}
					inline bool operator()(const lookup_t<AssetType>& lhs, const key_t<AssetType>& rhs) const
					{
						return lhs.asset==rhs.asset.get() && lhs.patch && *lhs.patch==rhs.patch;
					}
				};
				template<asset::Asset AssetType>
				using container_t = core::unordered_map<key_t<AssetType>,core::blake3_hash_t,HashEquals<AssetType>,HashEquals<AssetType>>;

			public:
				static const core::blake3_hash_t NoContentHash;

				inline CHashCache() = default;

				//
				template<asset::Asset AssetType>
				inline container_t<AssetType>::iterator find(const lookup_t<AssetType>& assetAndPatch)
				{
					return std::get<container_t<AssetType>>(m_containers).find<lookup_t<AssetType>>(assetAndPatch);
				}
				template<asset::Asset AssetType>
				inline container_t<AssetType>::const_iterator find(const lookup_t<AssetType>& assetAndPatch) const
				{
					return std::get<container_t<AssetType>>(m_containers).find<lookup_t<AssetType>>(assetAndPatch);
				}
				template<asset::Asset AssetType>
				inline container_t<AssetType>::const_iterator end() const
				{
					return std::get<container_t<AssetType>>(m_containers).end();
				}

				//
				class IPatchOverride
				{
					public:
						virtual const patch_t<asset::ICPUSampler>* operator()(const lookup_t<asset::ICPUSampler>&) const = 0;
						virtual const patch_t<asset::ICPUShader>* operator()(const lookup_t<asset::ICPUShader>&) const = 0;
						virtual const patch_t<asset::ICPUBuffer>* operator()(const lookup_t<asset::ICPUBuffer>&) const = 0;
						virtual const patch_t<asset::ICPUImage>* operator()(const lookup_t<asset::ICPUImage>&) const = 0;
						virtual const patch_t<asset::ICPUBufferView>* operator()(const lookup_t<asset::ICPUBufferView>&) const = 0;
						virtual const patch_t<asset::ICPUImageView>* operator()(const lookup_t<asset::ICPUImageView>&) const = 0;
						virtual const patch_t<asset::ICPUPipelineLayout>* operator()(const lookup_t<asset::ICPUPipelineLayout>&) const = 0;

						// certain items are not patchable, so there's no `patch_t` with non zero size
						inline const patch_t<asset::ICPUDescriptorSetLayout>* operator()(const lookup_t<asset::ICPUDescriptorSetLayout>& unpatchable) const
						{
							return unpatchable.patch;
						}
						inline const patch_t<asset::ICPURenderpass>* operator()(const lookup_t<asset::ICPURenderpass>& unpatchable) const
						{
							return unpatchable.patch;
						}
						inline const patch_t<asset::ICPUDescriptorSet>* operator()(const lookup_t<asset::ICPUDescriptorSet>& unpatchable) const
						{
							return unpatchable.patch;
						}

						// while other things are top level assets in the graph and `operator()` would never be called on their patch
				};
				// `cacheMistrustLevel` is how deep from `asset` do we start trusting the cache to contain correct non stale hashes
				template<asset::Asset AssetType>
				inline core::blake3_hash_t hash(const lookup_t<AssetType>& lookup, const IPatchOverride* patchOverride, const uint32_t cacheMistrustLevel=0)
				{
					if (!patchOverride || !lookup.asset || !lookup.patch)// || !lookup.patch->valid()) we assume any patch gotten is valid (to not have a dependancy on the device)
						return NoContentHash;

					// consult cache
					auto foundIt = find(lookup);
					auto& container = std::get<container_t<AssetType>>(m_containers);
					const bool found = foundIt!=container.end();
					// if found and we trust then return the cached hash
					if (cacheMistrustLevel==0 && found)
						return foundIt->second;

					// proceed with full hash computation
					core::blake3_hasher hasher = {};
					// We purposefully don't hash asset pointer, we hash the contents instead
					hash_impl impl = {{
							.hashCache = this,
							.patchOverride = patchOverride,
							.nextMistrustLevel = cacheMistrustLevel ? (cacheMistrustLevel-1):0,
							.hasher  = hasher
					}};
					// failed to hash (missing required deps), so return invalid hash
					// but don't eject stale entry, this may have been a mistake
					if (!impl(lookup))
						return NoContentHash;
					const auto retval = static_cast<core::blake3_hash_t>(hasher);
					assert(retval!=NoContentHash);

					if (found) // replace stale entry
						foundIt->second = retval;
					else // insert new entry
					{
						auto [insertIt,inserted] = container.emplace(
							key_t<AssetType>{
								.asset = core::smart_refctd_ptr<const AssetType>(lookup.asset),
								.patch = *lookup.patch
							},
							retval
						);
						assert(inserted && HashEquals<AssetType>()(insertIt->first,lookup) && insertIt->second==retval);
					}
					return retval;
				}

				// Its fastest to erase if you know your patch
				template<asset::Asset AssetType>
				inline bool erase(const lookup_t<AssetType>& what)
				{
					return std::get<container_t<AssetType>>(m_containers).erase(what)>0;
				}
				// Warning: Linear Search! Super slow!
				template<asset::Asset AssetType>
				inline bool erase(const AssetType* asset)
				{
					// TODO: improve by cycling through possible patches when the set of possibilities is small
					return core::erase_if(std::get<container_t<AssetType>>(m_containers),[asset](const auto& entry)->bool
						{
							auto const& [key,value] = entry;
							return key.asset==asset;
						}
					);
				}
				// TODO: `eraseStale(const IAsset*)` which erases a subgraph?
				// An asset being pointed to can mutate and that would invalidate the hash, this recomputes all hashes.
				void eraseStale(const IPatchOverride* patchOverride);
				// Clear the cache for a given type
				template<asset::Asset AssetType>
				inline void clear()
				{
					std::get<container_t<AssetType>>(m_containers).clear();
				}
				// Clear the caches completely
				inline void clear()
				{
					core::for_each_in_tuple(m_containers,[](auto& container)->void{container.clear();});
				}

				// only public to allow inheritance later in the cpp file
				struct hash_impl_base
				{
					CHashCache* hashCache;
					const IPatchOverride* patchOverride;
					const uint32_t nextMistrustLevel;
					core::blake3_hasher& hasher;
				};

			private:
				inline ~CHashCache() = default;

				// only public to allow inheritance later in the cpp file
				struct hash_impl : hash_impl_base
				{
					NBL_API bool operator()(lookup_t<asset::ICPUSampler>);
					NBL_API bool operator()(lookup_t<asset::ICPUShader>);
					NBL_API bool operator()(lookup_t<asset::ICPUBuffer>);
					NBL_API bool operator()(lookup_t<asset::ICPUImage>);
					NBL_API bool operator()(lookup_t<asset::ICPUBufferView>);
					NBL_API bool operator()(lookup_t<asset::ICPUImageView>);
					NBL_API bool operator()(lookup_t<asset::ICPUDescriptorSetLayout>);
					NBL_API bool operator()(lookup_t<asset::ICPUPipelineLayout>);
					NBL_API bool operator()(lookup_t<asset::ICPUPipelineCache>);
					NBL_API bool operator()(lookup_t<asset::ICPUComputePipeline>);
					NBL_API bool operator()(lookup_t<asset::ICPURenderpass>);
					NBL_API bool operator()(lookup_t<asset::ICPUGraphicsPipeline>);
					NBL_API bool operator()(lookup_t<asset::ICPUDescriptorSet>);
				};

				//
				core::tuple_transform_t<container_t,supported_asset_types> m_containers;
		};
		// Typed Cache (for a particular AssetType)
		class CCacheBase
		{
			public:
				// Make it clear to users that we don't look up just by the asset content hash
				struct key_t
				{
					inline key_t(const core::blake3_hash_t& contentHash, const size_t uniqueCopyGroupID) : value(contentHash)
					{
						reinterpret_cast<size_t*>(value.data)[0] ^= uniqueCopyGroupID;
					}

					inline bool operator==(const key_t&) const = default;

					// The blake3 hash is quite fat (256bit), so we don't actually store a full asset ref for comparison.
					// Assuming a uniform distribution of keys and perfect hashing, we'd expect a collision on average every 2^256 asset loads.
					// Or if you actually calculate the P(X>1) for any reasonable number of asset loads (k trials), the Poisson CDF will be pratically 0.
					core::blake3_hash_t value;
				};

			protected:
				struct ForwardHash
				{
					inline size_t operator()(const key_t& key) const
					{
						return std::hash<core::blake3_hash_t>()(key.value);
					}
				};
		};
		template<asset::Asset AssetType>
        class CCache final : public CCacheBase
        {
			public:
				// typedefs
				using forward_map_t = core::unordered_map<key_t,asset_cached_t<AssetType>,ForwardHash>;
				using reverse_map_t = core::unordered_map<typename asset_traits<AssetType>::lookup_t,key_t>;


				//
				inline CCache() = default;
				inline CCache(const CCache&) = default;
				inline CCache(CCache&&) = default;
				inline ~CCache() = default;

				inline CCache& operator=(const CCache&) = default;
				inline CCache& operator=(CCache&&) = default;

				// no point returning iterators to inserted positions, they're not stable
				inline bool insert(const key_t& _key, const asset_cached_t<AssetType>& _gpuObj)
				{
					auto [unused0,insertedF] = m_forwardMap.emplace(_key,_gpuObj);
					if (!insertedF)
						return false;
					auto [unused1,insertedR] = m_reverseMap.emplace(_gpuObj.get(),_key);
					assert(insertedR);
					return true;
				}

				//
				inline size_t size() const
				{
					assert(m_forwardMap.size()==m_reverseMap.size());
					return m_forwardMap.size();
				}

				//
				inline forward_map_t::const_iterator forwardMapEnd() const {return m_forwardMap.end();}
				inline reverse_map_t::const_iterator reverseMapEnd() const {return m_reverseMap.end();}

				// fastest lookup
				inline forward_map_t::const_iterator find(const key_t& _key) const {return m_forwardMap.find(_key);}
				inline reverse_map_t::const_iterator find(asset_traits<AssetType>::lookup_t gpuObject) const {return m_reverseMap.find(gpuObject);}

				// fastest erase
				inline bool erase(forward_map_t::const_iterator fit, reverse_map_t::const_iterator rit)
				{
					if (fit->first!=rit->second || fit->second.get()!=rit->first)
						return false;
					m_reverseMap.erase(rit);
					m_forwardMap.erase(fit);
					return true;
				}
				inline bool erase(forward_map_t::const_iterator it)
				{
					return erase(it,find(it->second));
				}
				inline bool erase(reverse_map_t::const_iterator it)
				{
					return erase(find(it->second),it);
				}

				//
				inline void merge(const CCache<AssetType>& other)
				{
					m_forwardMap.insert(other.m_forwardMap.begin(),other.m_forwardMap.end());
					m_reverseMap.insert(other.m_reverseMap.begin(),other.m_reverseMap.end());
				}

			private:
				friend class CAssetConverter;

				forward_map_t m_forwardMap;
				reverse_map_t m_reverseMap;
        };

		// A meta class to encompass all the Assets you might want to convert at once
        struct SInputs
        {
			// Normally all references to the same IAsset* would spawn the same IBackendObject*.
			// You need to tell us if an asset needs multiple copies, separate for each user. The return value of this function dictates what copy of the asset each user gets.
			// Each unique integer value returned for a given input `dependant` "spawns" a new copy.
			// Note that the group ID is the same size as a pointer, so you can e.g. cast a pointer of the user (parent reference) to size_t and use that for a unique copy for the user.
			// Note that we also call it with `user=={nullptr,0xdeadbeefBADC0FFEull}` for each entry in `SInputs::assets`.
			// NOTE: You might get extra copies within the same group ID due to inability to patch entries
			virtual inline size_t getDependantUniqueCopyGroupID(const size_t usersGroupCopyID, const asset::IAsset* user, const asset::IAsset* dependant) const
			{
				return 0;
			}

			// if you want concurrent sharing return a list here
			virtual inline std::span<const uint32_t> getSharedOwnershipQueueFamilies(const size_t groupCopyID, const asset::ICPUBuffer* buffer, const patch_t<asset::ICPUBuffer>& patch) const
			{
				return {};
			}

			virtual inline std::span<const uint32_t> getSharedOwnershipQueueFamilies(const size_t groupCopyID, const asset::ICPUImage* buffer, const patch_t<asset::ICPUImage>& patch) const
			{
				return {};
			}
#if 0
			// most plain PNG, JPG, etc. loaders don't produce images with mipmaps
			virtual inline uint16_t getMipLevelCount(const size_t groupCopyID, const asset::ICPUImageView* view, const patch_t<asset::ICPUImageView>& patch) const
			{
				const auto origCount = image->getCreationParameters().mipLevels;
				assert(img);
				if (img->getRegions().empty())
					return origCount;
				// makes no sense to mip-map integer values, and we can't encode into BC formats
				auto format = img->getCreationParameters().format;
				if (asset::isIntegerFormat(format) || asset::isBlockCompressionFormat(format))
					return origCount;
				// its enough to define a single mipmap region above the base level to prevent automatic computation
				for (auto& region : img->getRegions())
				if (region.imageSubresource.mipLevel)
					return false;

				// override the mip-count if its not an integer format and there was no mip-pyramid specified 
				if (params.mipLevels == 1u && !asset::isIntegerFormat(params.format))
					params.mipLevels = 1u + static_cast<uint32_t>(std::log2(static_cast<float>(core::max<uint32_t>(core::max<uint32_t>(params.extent.width, params.extent.height), params.extent.depth))));
				return true;
			}
#endif

			// Typed Range of Inputs of the same type
            template<asset::Asset AssetType>
            using asset_span_t = std::span<const typename asset_traits<AssetType>::asset_t* const>;
            template<asset::Asset AssetType>
            using patch_span_t = std::span<const patch_t<AssetType>>;

			// can be `nullptr` and even equal to `this`
			const CAssetConverter* readCache = nullptr;

			// recommended you set this
			system::logger_opt_ptr logger = nullptr;

			// A type-sorted non-polymorphic list of "root assets"
			core::tuple_transform_t<asset_span_t,supported_asset_types> assets = {};
			// Optional: Whatever is not in `patches` will generate a default patch
			core::tuple_transform_t<patch_span_t,supported_asset_types> patches = {};

			// optional, useful for shaders
			asset::IShaderCompiler::CCache* readShaderCache = nullptr;
			asset::IShaderCompiler::CCache* writeShaderCache = nullptr;
			IGPUPipelineCache* pipelineCache = nullptr;
        };
		// Split off from inputs because only assets that build on IPreHashed need uploading
		struct SConvertParams
		{
			// By default the last to queue to touch a GPU object will own it after any transfer or compute operations are complete.
			// If you want to record a pipeline barrier that will release ownership to another family, override this
			virtual inline uint32_t getFinalOwnerQueueFamily(const IDeviceMemoryBacked* imageOrBuffer, const core::blake3_hash_t& createdFrom)
			{
				return IQueue::FamilyIgnored;
			}
			// You can choose what layout the images get transitioned to at the end of an upload
			// (the images that don't get uploaded to can be transitioned from UNDEFINED without needing any work here)
			virtual inline IGPUImage::LAYOUT getFinalLayout(const IGPUImage* image, const core::blake3_hash_t& createdFrom)
			{
				using layout_t = IGPUImage::LAYOUT;
				using flags_t = IGPUImage::E_USAGE_FLAGS;
				const auto usages = image->getCreationParameters().usage;
				if (usages.hasFlags(flags_t::EUF_SAMPLED_BIT) || usages.hasFlags(flags_t::EUF_INPUT_ATTACHMENT_BIT))
					return layout_t::READ_ONLY_OPTIMAL;
				if (usages.hasFlags(flags_t::EUF_RENDER_ATTACHMENT_BIT) || usages.hasFlags(flags_t::EUF_TRANSIENT_ATTACHMENT_BIT))
					return layout_t::ATTACHMENT_OPTIMAL;
				// best guess
				return layout_t::GENERAL;
			}
			// By default we always insert into the cache
			virtual inline bool writeCache(const CCacheBase::key_t& createdFrom)
			{
				return true;
			}

			// One queue is for copies, another is for mip map generation and Acceleration Structure building
			SIntendedSubmitInfo transfer = {};
			SIntendedSubmitInfo compute = {};
			// required for Buffer or Image upload operations
			IUtilities* utilities = nullptr;
		};
        struct SReserveResult final
        {
				template<asset::Asset AssetType>
				using vector_t = core::vector<asset_cached_t<AssetType>>;

			public:
				template<asset::Asset AssetType>
				using staging_cache_t = core::unordered_map<typename asset_traits<AssetType>::video_t*,typename CCache<AssetType>::key_t>;

				inline SReserveResult(SReserveResult&&) = default;
				inline SReserveResult(const SReserveResult&) = delete;
				inline ~SReserveResult() = default;
				inline SReserveResult& operator=(const SReserveResult&) = delete;
				inline SReserveResult& operator=(SReserveResult&&) = default;

				// What queues you'll need to run the submit
				inline core::bitflag<IQueue::FAMILY_FLAGS> getRequiredQueueFlags() const {return m_queueFlags;}

				//
				inline operator bool() const {return bool(m_converter);}

				// until `convert` is called, this will only contain valid entries for items already found in `SInput::readCache`
				template<asset::Asset AssetType>
				std::span<const asset_cached_t<AssetType>> getGPUObjects() const {return std::get<vector_t<AssetType>>(m_gpuObjects);}

				// If you ever need to look up the content hashes of the assets AT THE TIME you converted them
				// REMEMBER it can have stale hashes (asset or its dependants mutated since hash computed),
				// then you can get hash mismatches or plain wrong hashes.
				CHashCache* getHashCache() {return m_hashCache.get();}
				const CHashCache* getHashCache() const {return m_hashCache.get();}

				// useful for virtual function implementations in `SConvertParams`
				template<asset::Asset AssetType>
				const auto& getStagingCache() const {return std::get<staging_cache_t<AssetType>>(m_stagingCaches);}

				// You only get to call this once if successful, return value tells you whether you can submit the cmdbuffers
				struct SConvertResult final
				{
					public:
						inline ~SConvertResult() = default;

						// Just because this class converts to true, DOESN'T MEAN the gpu objects are ready for use! You need to submit the intended submits for that first!
						inline operator bool() const {return device;}

						struct SSubmitResult
						{
							inline bool blocking() const {return transfer.blocking()||compute.blocking();}

							inline bool ready() const {return transfer.ready()&&compute.ready();}

							inline ISemaphore::WAIT_RESULT wait() const
							{
								if (transfer.blocking())
								{
									if (compute.blocking())
									{
										const ISemaphore::SWaitInfo waitInfos[2] = {transfer,compute};
										auto* device = const_cast<ILogicalDevice*>(waitInfos[0].semaphore->getOriginDevice());
										assert(waitInfos[1].semaphore->getOriginDevice()==device);
										return device->blockForSemaphores(waitInfos,true);
									}
									return transfer.wait();
								}
								return compute.wait();
							}

							inline operator bool() const
							{
								if (wait()!=ISemaphore::WAIT_RESULT::SUCCESS)
									return false;
								return transfer.copy()==IQueue::RESULT::SUCCESS && compute.copy()==IQueue::RESULT::SUCCESS;
							}

							ISemaphore::future_t<IQueue::RESULT> transfer = IQueue::RESULT::OTHER_ERROR;
							ISemaphore::future_t<IQueue::RESULT> compute = IQueue::RESULT::OTHER_ERROR;
						};
						// Submits the buffered up calls, unline IUtilities::autoSubmit, no patching, it complicates life too much, just please pass correct open SIntendedSubmits.
						inline SSubmitResult submit(
							std::span<const IQueue::SSubmitInfo::SSemaphoreInfo> extraTransferSignalSemaphores={},
							std::span<const IQueue::SSubmitInfo::SSemaphoreInfo> extraComputeSignalSemaphores={}
						)
						{
							// invalid
							if (!device)
								return {};
							for (const auto& signal : extraTransferSignalSemaphores)
							if (signal.semaphore->getOriginDevice()!=device)
								return {};
							for (const auto& signal : extraComputeSignalSemaphores)
							if (signal.semaphore->getOriginDevice()!=device)
								return {};
							// you only get one shot at this!
							device = nullptr;

							SSubmitResult retval = {};
							// patch tmps
							core::smart_refctd_ptr<ISemaphore> patchSema;
							IQueue::SSubmitInfo::SSemaphoreInfo patch;
							// first submit transfer
							if (submitsNeeded.hasFlags(IQueue::FAMILY_FLAGS::TRANSFER_BIT) && transfer->getScratchCommandBuffer()->getState()==IGPUCommandBuffer::STATE::RECORDING)
							{
								assert(transfer);
								transfer->getScratchCommandBuffer()->end();
								// patch if needed
								if (extraTransferSignalSemaphores.empty())
								{
									patchSema = device->createSemaphore(0);
									// cannot signal from TRANSFER stages because there might be a ownership transfer or layout transition
									// and we need to wait for right after that, which doesn't have an explicit stage
									patch = {patchSema.get(),1,asset::PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS};
									extraTransferSignalSemaphores = {&patch,1};
								}
								// submit
								auto submit = transfer->popSubmit(extraTransferSignalSemaphores);
								if (const auto error=transfer->queue->submit(submit); error!=IQueue::RESULT::SUCCESS)
									return {error};
								retval.transfer.set({extraTransferSignalSemaphores.back().semaphore,extraTransferSignalSemaphores.back().value});
							}
							else
							for (auto& sema : extraTransferSignalSemaphores)
								sema.semaphore->signal(sema.value);
							retval.transfer.set(IQueue::RESULT::SUCCESS);

							// then submit compute
							if (submitsNeeded.hasFlags(IQueue::FAMILY_FLAGS::COMPUTE_BIT) && compute->getScratchCommandBuffer()->getState()==IGPUCommandBuffer::STATE::RECORDING)
							{
								assert(compute);
								compute->getScratchCommandBuffer()->end();
								// patch if needed
								if (extraComputeSignalSemaphores.empty())
								{
									patchSema = device->createSemaphore(0);
									// cannot signal from COMPUTE stages because there might be a ownership transfer or layout transition
									// and we need to wait for right after that, which doesn't have an explicit stage
									patch = {patchSema.get(),1,asset::PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS};
									extraComputeSignalSemaphores = {&patch,1};
								}
								// submit
								auto submit = compute->popSubmit(extraComputeSignalSemaphores);
								if (const auto error=compute->queue->submit(submit); error!=IQueue::RESULT::SUCCESS)
									return {std::move(retval.transfer),error};
								retval.compute.set({extraComputeSignalSemaphores.back().semaphore,extraComputeSignalSemaphores.back().value});
							}
							else
							for (auto& sema : extraComputeSignalSemaphores)
								sema.semaphore->signal(sema.value);
							retval.compute.set(IQueue::RESULT::SUCCESS);

							return retval;
						}

					private:
						friend class SReserveResult;
						friend class CAssetConverter;

						inline SConvertResult() = default;
						// because the intended submits are pointers, return value of `convert` should not be allowed to escape its scope
						inline SConvertResult(SConvertResult&&) = default;
						inline SConvertResult& operator=(SConvertResult&&) = default;

						// these get set after a successful conversion
						SIntendedSubmitInfo* transfer = nullptr;
						SIntendedSubmitInfo* compute = nullptr;
						ILogicalDevice* device = nullptr;
						core::bitflag<IQueue::FAMILY_FLAGS> submitsNeeded = IQueue::FAMILY_FLAGS::NONE;
				};
				inline SConvertResult convert(SConvertParams& params)
				{
					SConvertResult enqueueSuccess = m_converter->convert_impl(std::move(*this),params);
					if (enqueueSuccess)
					{
						// wipe after success
						core::for_each_in_tuple(m_stagingCaches,[](auto& stagingCache)->void{stagingCache.clear();});
						// disallow a double run
						m_converter = nullptr;
					}
					return enqueueSuccess;
				}

			private:
				friend class CAssetConverter;

				inline SReserveResult() = default;

				// we need to remember a few things so that `convert` can work seamlessly
				core::smart_refctd_ptr<CAssetConverter> m_converter = nullptr;
				system::logger_opt_smart_ptr m_logger = nullptr;

				//
				core::smart_refctd_ptr<CHashCache> m_hashCache = nullptr;

				// for every entry in the input array, we have this mapped 1:1
				core::tuple_transform_t<vector_t,supported_asset_types> m_gpuObjects = {};
				
				// we don't insert into the writeCache until conversions are successful
				core::tuple_transform_t<staging_cache_t,supported_asset_types> m_stagingCaches;
				// need a more explicit list of GPU objects that need device-assisted conversion
				template<asset::Asset AssetType>
				struct ConversionRequest
				{
					// canonical asset (the one that provides content)
					core::smart_refctd_ptr<const AssetType> canonical;
					// gpu object to transfer canonical's data to or build it from
					asset_traits<AssetType>::video_t* gpuObj;
				};
				template<asset::Asset AssetType>
				using conversion_requests_t = core::vector<ConversionRequest<AssetType>>;
				using convertible_asset_types = core::type_list<
					asset::ICPUBuffer,
					asset::ICPUImage/*,
					asset::ICPUBottomLevelAccelerationStructure,
					asset::ICPUTopLevelAccelerationStructure*/
				>;
				core::tuple_transform_t<conversion_requests_t,convertible_asset_types> m_conversionRequests;

				//
				core::bitflag<IQueue::FAMILY_FLAGS> m_queueFlags = IQueue::FAMILY_FLAGS::NONE;
        };
		// First Pass: Explore the DAG of Assets and "gather" patch infos and create equivalent GPU Objects.
		NBL_API SReserveResult reserve(const SInputs& inputs);

#undef NBL_API

		// Only const methods so others are not able to insert things made by e.g. different devices
		template<asset::Asset AssetType>
		inline const CCache<AssetType>& getCache() const
		{
			return std::get<CCache<AssetType>>(m_caches);
		}

		//
		inline void merge(const CAssetConverter* other)
		{
			std::apply([&](auto&... caches)->void{
				(..., caches.merge(std::get<std::remove_reference_t<decltype(caches)>>(other->m_caches)));
			},m_caches);
		}

    protected:
        inline CAssetConverter(const SCreationParams& params) : m_params(std::move(params)) {}
        virtual inline ~CAssetConverter() = default;
		
		template<asset::Asset AssetType>
		inline CCache<AssetType>& getCache()
		{
			return std::get<CCache<AssetType>>(m_caches);
		}

		friend struct SReserveResult;
		SReserveResult::SConvertResult convert_impl(SReserveResult&& reservations, SConvertParams& params);

        SCreationParams m_params;
		core::tuple_transform_t<CCache,supported_asset_types> m_caches;
};


// nothing to do
template<asset::Asset AssetType>
inline CAssetConverter::patch_impl_t<AssetType>::patch_impl_t(const AssetType* asset) {}
// always valid
template<asset::Asset AssetType>
inline bool CAssetConverter::patch_impl_t<AssetType>::valid(const ILogicalDevice* device) { return true; }

}
#endif