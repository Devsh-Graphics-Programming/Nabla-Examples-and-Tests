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
			// acceleration structures
			// image,
//			asset::ICPUBufferView,
			// image view
			asset::ICPUDescriptorSetLayout,
			asset::ICPUPipelineLayout,
			asset::ICPUPipelineCache,
			asset::ICPUComputePipeline,
			asset::ICPURenderpass,
			asset::ICPUGraphicsPipeline
			// descriptor sets
			// framebuffer
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

		// When getting dependents, the creation parameters of GPU objects will be produced and patched appropriately.
		// `patch_t` uses CRPT to inherit from `patch_impl_t` to provide default `operator==` and `update_hash()` definition.
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

				inline bool valid(const ILogicalDevice* device) const {return usage!=IGPUBuffer::E_USAGE_FLAGS::EUF_NONE;}

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
		struct patch_impl_t<asset::ICPUPipelineLayout>
		{
			public:
				PATCH_IMPL_BOILERPLATE(asset::ICPUPipelineLayout);

				inline bool valid(const ILogicalDevice* device) const {return !invalid;}

				using shader_stage_t = asset::IShader::E_SHADER_STAGE;
				std::array<core::bitflag<shader_stage_t>,asset::CSPIRVIntrospector::MaxPushConstantsSize> pushConstantBytes = {shader_stage_t::ESS_UNKNOWN};
				bool invalid = true;
				
			protected:
				inline std::pair<bool,this_t> combine(const this_t& other) const
				{
					this_t retval = *this;
					for (auto byte=0; byte!=pushConstantBytes.size(); byte++)
						retval.pushConstantBytes[byte] |= other.pushConstantBytes[byte];
					return {true,retval};
				}
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
			inline std::pair<bool,this_t> combine(const this_t& other) const
			{
				//assert(base_t::valid() && other.valid());
				return base_t::combine(other);
			}

			// actual new methods
			inline bool operator==(const patch_t<AssetType>& other) const
			{
				return memcmp(this,&other,sizeof(base_t))==0;
			}
		};
#define NBL_API
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

				// `cacheMistrustLevel` is how deep from `asset` do we start trusting the cache to contain correct non stale hashes
				template<asset::Asset AssetType, typename PatchGetter>
				inline core::blake3_hash_t hash(const AssetType* asset, PatchGetter patchGet, const uint32_t cacheMistrustLevel=0)
				{
					if (!asset)
						return NoContentHash;

					// this is the only time a call to `patchGet` happens, which allows it to mutate its state only once
					const patch_t<AssetType>* patch = patchGet(asset);
					// failed to provide us with a patch, so fail the hash
					if (!patch)// || !patch->valid()) we assume any patch gotten is valid (to not have a dependancy on the device)
						return NoContentHash;

					// consult cache
					const lookup_t<AssetType> lookup = {asset,patch};
					auto foundIt = find(lookup);
					auto& container = std::get<container_t<AssetType>>(m_containers);
					const bool found = foundIt!=container.end();
					// if found and we trust then return the cached hash
					if (cacheMistrustLevel==0 && found)
						return foundIt->second;
					// proceed with full hash computation
					//! We purposefully don't hash asset pointer, we hash the contents instead
					const auto nextMistrustLevel = cacheMistrustLevel ? (cacheMistrustLevel-1):0;
					const auto retval = static_cast<core::blake3_hash_t>(
						hash_impl<AssetType,PatchGetter>::_call({
							this,asset,patch,patchGet,nextMistrustLevel
						})
					);
					// failed to hash (missing required deps), so return invalid hash
					// but don't eject stale entry, this may have been a mistake
					if (retval==NoContentHash)
						return NoContentHash;

					if (found) // replace stale entry
						foundIt->second = retval;
					else // insert new entry
					{
						auto [insertIt,inserted] = container.emplace(
							key_t<AssetType>{
								.asset = core::smart_refctd_ptr<const AssetType>(asset),
								.patch = *patch
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
				template<typename PatchGetter>
				inline void eraseStale(const PatchGetter& patchGet)
				{
					auto rehash = [&]<typename AssetType>() -> void
					{
						auto& container = std::get<container_t<AssetType>>(m_containers);
						core::erase_if(container,[this](const auto& entry)->bool
							{
								// backup because `hash(lookup)` call will update it
								const auto oldHash = entry.second;
								const auto& key = entry.first;
								// can re-use cached hashes for dependants if we start ejecting in the correct order
								const auto newHash = hash(key.asset.get(),patchGet,/*.cacheMistrustLevel = */1);
								return newHash!=oldHash || newHash==NoContentHash;
							}
						);
					};
					// to make the process more efficient we start ejecting from "lowest level" assets
					rehash.operator()<asset::ICPUSampler>();
					rehash.operator()<asset::ICPUDescriptorSetLayout>();
					rehash.operator()<asset::ICPUPipelineLayout>();
					// shaders and images depend on buffers for data sourcing
					rehash.operator()<asset::ICPUBuffer>();
				//	rehash.operator()<ICPUBufferView>();
				//	rehash.operator()<ICPUImage>();
				//	rehash.operator()<ICPUImageView>();
				//	rehash.operator()<ICPUBottomLevelAccelerationStructure>();
				//	rehash.operator()<ICPUTopLevelAccelerationStructure>();
					// only once all the descriptor types have been hashed, we can hash sets
				//	rehash.operator()<ICPUDescriptorSet>();
					// naturally any pipeline depends on shaders and pipeline cache
					rehash.operator()<asset::ICPUShader>();
					rehash.operator()<asset::ICPUPipelineCache>();
					rehash.operator()<asset::ICPUComputePipeline>();
					// graphics pipeline needs a renderpass
					rehash.operator()<asset::ICPURenderpass>();
					rehash.operator()<asset::ICPUGraphicsPipeline>();
				//	rehash.operator()<ICPUFramebuffer>();
				}
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

			private:
				inline ~CHashCache() = default;

				//
				template<asset::Asset AssetType, typename PatchGetter>
				struct hash_impl;
				template<asset::Asset AssetType, typename PatchGetter>
				struct hash_impl_args final
				{
					template<asset::Asset DepAssetType>
					inline core::blake3_hash_t depHash(const DepAssetType* dep)
					{
						return _this->hash(dep,patchGet,nextMistrustLevel);
					}

					CHashCache* _this;
					const AssetType* const asset;
					const patch_t<AssetType>* patch;
					const PatchGetter& patchGet;
					const uint32_t nextMistrustLevel;
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
#if 0
			virtual inline std::span<const uint32_t> getSharedOwnershipQueueFamilies(const size_t groupCopyID, const asset::ICPUImage* buffer, const patch_t<asset::ICPUImage>& patch) const
			{
				return {};
			}

			// most plain PNG, JPG, etc. loaders don't produce images with mipmaps
			virtual inline uint16_t getMipLevelCount(const size_t groupCopyID, const asset::ICPUImage* image, const patch_t<asset::ICPUImage>& patch) const
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
					asset::ICPUBuffer/*,
					asset::ICPUImage,
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


template<typename PatchGetter>
struct CAssetConverter::CHashCache::hash_impl<asset::ICPUSampler,PatchGetter>
{
	static inline core::blake3_hasher _call(hash_impl_args<asset::ICPUSampler,PatchGetter>&& args)
	{
		auto patchedParams = args.asset->getParams();
		patchedParams.AnisotropicFilter = args.patch->anisotropyLevelLog2;
		return core::blake3_hasher().update(&patchedParams,sizeof(patchedParams));
	}
};

template<typename PatchGetter>
struct CAssetConverter::CHashCache::hash_impl<asset::ICPUShader,PatchGetter>
{
	static inline core::blake3_hasher _call(hash_impl_args<asset::ICPUShader,PatchGetter>&& args)
	{
		const auto* asset = args.asset;

		core::blake3_hasher hasher;
		hasher << args.patch->stage;
		const auto type = asset->getContentType();
		hasher << type;
		// if not SPIR-V then own path matters
		if (type!=asset::ICPUShader::E_CONTENT_TYPE::ECT_SPIRV)
			hasher << asset->getFilepathHint();
		const auto* content = asset->getContent();
		if (!content || content->getContentHash()==NoContentHash)
			return {};
		// we're not using the buffer directly, just its contents
		hasher << content->getContentHash();
		return hasher;
	}
};

template<typename PatchGetter>
struct CAssetConverter::CHashCache::hash_impl<asset::ICPUBuffer,PatchGetter>
{
	static inline core::blake3_hasher _call(hash_impl_args<asset::ICPUBuffer,PatchGetter>&& args)
	{
		auto patchedParams = args.asset->getCreationParams();
		assert(args.patch->usage.hasFlags(patchedParams.usage));
		patchedParams.usage = args.patch->usage;
		return core::blake3_hasher().update(&patchedParams,sizeof(patchedParams)) << args.asset->getContentHash();
	}
};

template<typename PatchGetter>
struct CAssetConverter::CHashCache::hash_impl<asset::ICPUDescriptorSetLayout,PatchGetter>
{
	static inline core::blake3_hasher _call(hash_impl_args<asset::ICPUDescriptorSetLayout,PatchGetter>&& args)
	{
		const auto* asset = args.asset;

		core::blake3_hasher hasher;
		using storage_range_index_t = asset::ICPUDescriptorSetLayout::CBindingRedirect::storage_range_index_t;
		for (uint32_t t=0u; t<static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT); t++)
		{
			const auto type = static_cast<asset::IDescriptor::E_TYPE>(t);
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
		const auto& immutableSamplerRedirects = asset->getImmutableSamplerRedirect();
		const auto count = immutableSamplerRedirects.getBindingCount();
		for (auto i=0u; i<count; i++)
		{
			const storage_range_index_t storageRangeIx(i);
			hasher << immutableSamplerRedirects.getBinding(storageRangeIx).data;
		}
		for (const auto& immutableSampler : asset->getImmutableSamplers())
		{
			const auto samplerHash = args.depHash(immutableSampler.get());
			if (samplerHash==NoContentHash)
				return {};
			hasher << samplerHash;
		}
		return hasher;
	}
};

template<typename PatchGetter>
struct CAssetConverter::CHashCache::hash_impl<asset::ICPUPipelineLayout,PatchGetter>
{
	static inline core::blake3_hasher _call(hash_impl_args<asset::ICPUPipelineLayout,PatchGetter>&& args)
	{
		core::blake3_hasher hasher;
		hasher << std::span(args.patch->pushConstantBytes);
		for (auto i = 0; i < asset::ICPUPipelineLayout::DESCRIPTOR_SET_COUNT; i++)
		{
			auto dep = args.asset->getDescriptorSetLayout(i);
			if (dep) // remember each layout is optional
			{
				const auto dsLayoutHash = args.depHash(dep);
				if (dsLayoutHash==NoContentHash)
					return {};
				hasher << dsLayoutHash;
			}
			else
				hasher << NoContentHash;
		}
		return hasher;
	}
};

template<typename PatchGetter>
struct CAssetConverter::CHashCache::hash_impl<asset::ICPUPipelineCache,PatchGetter>
{
	static inline core::blake3_hasher _call(hash_impl_args<asset::ICPUPipelineCache,PatchGetter>&& args)
	{
		core::blake3_hasher hasher;
		for (const auto& entry : args.asset->getEntries())
		{
			hasher << entry.first.deviceAndDriverUUID;
			if (entry.first.meta)
				hasher.update(entry.first.meta->data(),entry.first.meta->size());
		}
		hasher << args.asset->getContentHash();
		return hasher;
	}
};

template<typename PatchGetter>
struct CAssetConverter::CHashCache::hash_impl<asset::ICPUComputePipeline,PatchGetter>
{
	static inline core::blake3_hasher _call(hash_impl_args<asset::ICPUComputePipeline,PatchGetter>&& args)
	{
		const auto* asset = args.asset;

		core::blake3_hasher hasher;
		{
			const auto layoutHash = args.depHash(asset->getLayout());
			if (layoutHash==NoContentHash)
				return {};
			hasher << layoutHash;
		}
		const auto& specInfo = asset->getSpecInfo();
		hasher << specInfo.entryPoint;
		{
			const auto shaderHash = args.depHash(specInfo.shader);
			if (shaderHash==NoContentHash)
				return {};
			hasher << shaderHash;
		}
		if (specInfo.entries)
		for (const auto& specConstant : *specInfo.entries)
		{
			hasher << specConstant.first;
			hasher.update(specConstant.second.data,specConstant.second.size);
		}
		hasher << specInfo.requiredSubgroupSize;
		hasher << specInfo.requireFullSubgroups;
		return hasher;
	}
};

template<typename PatchGetter>
struct CAssetConverter::CHashCache::hash_impl<asset::ICPURenderpass,PatchGetter>
{
	static inline core::blake3_hasher _call(hash_impl_args<asset::ICPURenderpass,PatchGetter>&& args)
	{
		const auto* asset = args.asset;

		core::blake3_hasher hasher;
		hasher << asset->getDepthStencilAttachmentCount();
		hasher << asset->getColorAttachmentCount();
		hasher << asset->getSubpassCount();
		hasher << asset->getDependencyCount();
		hasher << asset->getViewMaskMSB();
		const asset::ICPURenderpass::SCreationParams& params = asset->getCreationParameters();
		{
			auto hashLayout = [&](const asset::E_FORMAT format, const asset::IImage::SDepthStencilLayout& layout)->void
			{
				if (!asset::isStencilOnlyFormat(format))
					hasher << layout.depth;
				if (!asset::isDepthOnlyFormat(format))
					hasher << layout.stencil;
			};

			for (auto i=0; i<asset->getDepthStencilAttachmentCount(); i++)
			{
				auto entry = params.depthStencilAttachments[i];
				if (!entry.valid())
					return {};
				hasher << entry.format;
				hasher << entry.samples;
				hasher << entry.mayAlias;
				auto hashOp = [&](const auto& op)->void
				{
					if (!asset::isStencilOnlyFormat(entry.format))
						hasher << op.depth;
					if (!asset::isDepthOnlyFormat(entry.format))
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
					return {};
				hasher.update(&entry,sizeof(entry));
			}
			// subpasses
			using SubpassDesc = asset::ICPURenderpass::SCreationParams::SSubpassDescription;
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
                        if (inputIt->aspectMask==asset::IImage::EAF_COLOR_BIT)
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
			hasher.update(params.dependencies,sizeof(asset::ICPURenderpass::SCreationParams::SSubpassDependency)*asset->getDependencyCount());
		}
		hasher.update(params.viewCorrelationGroup,sizeof(params.viewCorrelationGroup));

		return hasher;
	}
};

template<typename PatchGetter>
struct CAssetConverter::CHashCache::hash_impl<asset::ICPUGraphicsPipeline,PatchGetter>
{
	static inline core::blake3_hasher _call(hash_impl_args<asset::ICPUGraphicsPipeline,PatchGetter>&& args)
	{
		const auto* asset = args.asset;

		core::blake3_hasher hasher;
		{
			const auto layoutHash = args.depHash(asset->getLayout());
			if (layoutHash==NoContentHash)
				return {};
			hasher << layoutHash;
		}

		{
			const auto renderpassHash = args.depHash(asset->getRenderpass());
			if (renderpassHash==NoContentHash)
				return {};
			hasher << renderpassHash;
		}

		using shader_stage_t = asset::ICPUShader::E_SHADER_STAGE;
		auto hashStage = [&](const shader_stage_t stage, const bool required=false)->bool
		{
			const auto& specInfo = asset->getSpecInfo(stage);
			if (!specInfo.shader) // fail only if required
				return required;

			hasher << specInfo.entryPoint;
			{
				const auto shaderHash = args.depHash(specInfo.shader);
				// having a non-required shader at a stage but that shader failing hash, fails us
				if (shaderHash==NoContentHash)
					return true;
				hasher << shaderHash;
			}
			if (specInfo.entries)
			for (const auto& specConstant : *specInfo.entries)
			{
				hasher << specConstant.first;
				hasher.update(specConstant.second.data,specConstant.second.size);
			}
			hasher << specInfo.requiredSubgroupSize;
			return false;
		};
		if (hashStage(shader_stage_t::ESS_VERTEX,true) ||
			hashStage(shader_stage_t::ESS_TESSELLATION_CONTROL) ||
			hashStage(shader_stage_t::ESS_TESSELLATION_EVALUATION) ||
			hashStage(shader_stage_t::ESS_GEOMETRY) ||
			hashStage(shader_stage_t::ESS_FRAGMENT))
			return {};

		const auto& params = asset->getCachedCreationParams();
		{
			for (auto i=0; i<asset::SVertexInputParams::MAX_VERTEX_ATTRIB_COUNT; i++)
			if (params.vertexInput.enabledAttribFlags&(0x1u<<i))
			{
				const auto& attribute = params.vertexInput.attributes[i];
				hasher.update(&attribute,sizeof(asset::SVertexInputAttribParams));
				hasher.update(&params.vertexInput.bindings+attribute.binding,sizeof(asset::SVertexInputBindingParams));
			}
			const auto& ass = params.primitiveAssembly;
			hasher << ass.primitiveType;
			hasher << ass.primitiveRestartEnable;
			if (ass.primitiveType==asset::E_PRIMITIVE_TOPOLOGY::EPT_PATCH_LIST)
				hasher << ass.tessPatchVertCount;
			const auto& raster = params.rasterization;
			if (!raster.rasterizerDiscard)
			{
				hasher << raster.viewportCount;
				hasher << raster.samplesLog2;
				hasher << raster.polygonMode;
				//if (raster.polygonMode==asset::E_POLYGON_MODE::EPM_FILL) // do wireframes and point draw with face culling?
				{
					hasher << raster.faceCullingMode;
					hasher << raster.frontFaceIsCCW;
				}
				const auto& rpassParam = asset->getRenderpass()->getCreationParameters();
				const auto& depthStencilRef = rpassParam.subpasses[params.subpassIx].depthStencilAttachment.render;
				if (depthStencilRef.used())
				{
					const auto attFormat = rpassParam.depthStencilAttachments[depthStencilRef.attachmentIndex].format;
					if (!asset::isStencilOnlyFormat(attFormat))
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
					if (raster.stencilTestEnable() && !asset::isDepthOnlyFormat(attFormat))
					{
						if ((raster.faceCullingMode&asset::E_FACE_CULL_MODE::EFCM_FRONT_BIT)==0)
							hasher << raster.frontStencilOps;
						if ((raster.faceCullingMode&asset::E_FACE_CULL_MODE::EFCM_BACK_BIT)==0)
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

		return hasher;
	}
};

}
#endif