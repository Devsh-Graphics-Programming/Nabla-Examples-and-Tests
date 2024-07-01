// Copyright (C) 2024-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
#ifndef _NBL_VIDEO_C_ASSET_CONVERTER_INCLUDED_
#define _NBL_VIDEO_C_ASSET_CONVERTER_INCLUDED_

#if 1
#include "nabla.h"
#else
#include "nbl/video/utilities/IUtilities.h"
#include "nbl/video/asset_traits.h"
#endif
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
* Therefore if you don't want some resource to be deduplicated you need to "explicitly" set `CAssetConverter::input_t<>::uniqueCopyGroupID`.
*/
class CAssetConverter : public core::IReferenceCounted
{
	public:
		// meta tuple
		using supported_asset_types = std::tuple<
			asset::ICPUShader/*,
			asset::ICPUDescriptorSetLayout,
			asset::ICPUPipelineLayout*/
		>;

		struct SCreationParams
		{
			inline bool valid() const
			{
				if (!device)
					return false;

				if (pipelineCache && pipelineCache->getOriginDevice()!=device) 
					return false;

				return true;
			}

			// required not null
			ILogicalDevice* device = nullptr;
			// optional
			core::smart_refctd_ptr<const asset::ISPIRVOptimizer> optimizer = {};
			core::smart_refctd_ptr<asset::IShaderCompiler::CCache> compilerCache = {};
			core::smart_refctd_ptr<IGPUPipelineCache> pipelineCache = {};
		};
		static inline core::smart_refctd_ptr<CAssetConverter> create(SCreationParams&& params)
		{
			if (!params.valid())
				return nullptr;
			return core::smart_refctd_ptr<CAssetConverter>(new CAssetConverter(std::move(params)),core::dont_grab);
		}

		// when getting dependents, the creation parameters of GPU objects will be produced and patched appropriately
		template<asset::Asset AssetType>
		struct patch_t;
		// This is the object we're trying to de-duplicate and hash-cons
		template<asset::Asset AssetType>
		struct input_t final
		{
			inline bool operator==(const input_t<AssetType>& rhs) const = default;

			inline size_t hash() const
			{
				return ptrdiff_t(asset)^ptrdiff_t(uniqueCopyGroupID);
			}

			inline bool valid() const
			{
				return asset;
			}

			const typename asset_traits<AssetType>::asset_t* asset = nullptr;
			// Normally all references to the same IAsset* would spawn the same IBackendObject*, set this different integers to get more copies.
			// Each unique integer value "spawns" a new copy, note that the group ID is the same size as a pointer, so you can e.g.
			// cast a pointer of the user (parent reference) to size_t and use that for a unique copy for the user.
			size_t uniqueCopyGroupID = 0; // NOTE: this may have to move a bit farther down
		};
		// And these are the results of the conversion.
		template<asset::Asset AssetType>
		struct cached_t final
		{
			private:
				using video_t = typename asset_traits<AssetType>::video_t;
				constexpr static inline bool RefCtd = core::ReferenceCounted<video_t>;

			public:
				inline cached_t() = default;
				inline cached_t(const cached_t<AssetType>& other) : cached_t() {operator=(other);}
				inline cached_t(cached_t<AssetType>&&) = default;

				// special wrapping to make smart_refctd_ptr copyable
				inline cached_t<AssetType>& operator=(const cached_t<AssetType>& rhs)
				{
					if constexpr (RefCtd)
						value = core::smart_refctd_ptr<video_t>(rhs.value.get());
					else
						value = rhs;
					return *this;
				}
				inline cached_t<AssetType>& operator=(cached_t<AssetType>&&) = default;

				inline const auto& get() const
				{
					if constexpr (RefCtd)
						return value.get();
					else
						return value;
				}

				using type = std::conditional_t<RefCtd,core::smart_refctd_ptr<video_t>,video_t>;
				type value = {};
		};
		template<>
		struct patch_t<asset::ICPUShader>
		{
			using this_t = patch_t<asset::ICPUShader>;

			inline patch_t() = default;
			inline patch_t(const asset::ICPUShader* shader) : stage(shader->getStage()) {}

			inline bool valid() const {return nbl::hlsl::bitCount<uint32_t>(stage)!=1;}

			inline this_t combine(const this_t& other) const
			{
				if (stage!=other.stage)
				{
					if (stage==IGPUShader::ESS_UNKNOWN)
						return other; // return the other whether valid or not
					else if (other.stage!=IGPUShader::ESS_UNKNOWN)
						return {}; // invalid
					// other is UNKNOWN so fallthrough and return us whether valid or not
				}
				return *this;
			}

			IGPUShader::E_SHADER_STAGE stage = IGPUShader::ESS_UNKNOWN;
		};
		template<>
		struct patch_t<asset::ICPUDescriptorSetLayout>
		{
			using this_t = patch_t<asset::ICPUDescriptorSetLayout>;

			inline patch_t() = default;
			inline patch_t(const asset::ICPUDescriptorSetLayout* layout) {}

			inline bool valid() const {return true;}

			inline this_t combine(const this_t& other) const
			{
				return *this;
			}
		};
		template<>
		struct patch_t<asset::ICPUPipelineLayout>
		{
			using this_t = patch_t<asset::ICPUPipelineLayout>;

			inline patch_t() = default;
			inline patch_t(const asset::ICPUPipelineLayout* pplnLayout)
			{
				const auto pc = pplnLayout->getPushConstantRanges();
				for (auto it=pc.begin(); it!=pc.end(); it++)
				for (auto byte=it->offset; byte<it->offset+it->size; byte++)
					pushConstantBytes[byte] = it->stageFlags;
			}

			inline bool valid() const {return true;}

			inline this_t combine(const this_t& other) const
			{
				this_t retval = *this;
				for (auto byte=0; byte!=pushConstantBytes.size(); byte++)
					retval.pushConstantBytes[byte] |= other.pushConstantBytes[byte];
				return retval;
			}

			std::array<core::bitflag<IGPUShader::E_SHADER_STAGE>,asset::CSPIRVIntrospector::MaxPushConstantsSize> pushConstantBytes = {IGPUShader::ESS_UNKNOWN};
		};
#define NBL_API
		// Typed Cache (for a particular AssetType)
		template<asset::Asset AssetType>
        class CCache final : core::Uncopyable
        {
			public:
				inline CCache() = default;
				inline CCache(CCache&&) = default;
				inline ~CCache() = default;

				inline CCache& operator=(CCache&&) = default;

				// fastest lookup
				inline const auto& find(const core::blake3_hash_t& hash) const
				{
					return m_container.find(hash)->second.get();
				}

				struct lookup_t final
				{
					public:
						// constructor for when you want to override the creation parameters for the GPU Object while keeping the asset const
						inline lookup_t(const input_t<AssetType>& _input, const patch_t<AssetType>& _patch) : input(_input), patch(_patch) {}
						// the default constructor that deducts the patch params from a const asset which we don't want to change (so far)
						inline lookup_t(const input_t<AssetType>& _input) : input(_input), patch(input.asset) {}

						inline bool operator==(const lookup_t&) const = default;

						inline bool valid() const
						{
							return input.valid() && patch.valid();
						}

						inline core::blake3_hash_t hash() const
						{
							core::blake3_hash_t retval;
							{
								blake3_hasher hasher;
								blake3_hasher_init(&hasher);
								core::blake3_hasher_update(hasher,input.uniqueCopyGroupID);
								hash_impl(&hasher);
								blake3_hasher_finalize(&hasher,retval.data,sizeof(retval));
							}
							return retval;
						}

						input_t<AssetType> input;
						patch_t<AssetType> patch;

					private:
						NBL_API void hash_impl(blake3_hasher* hasher) const;
				};
				// slower but more useful version
                inline auto find(const lookup_t& key) const
                {
					if (key.valid())
						return m_container.end();

                    return find(key.hash());
                }

				inline void merge(const CCache<AssetType>& other)
				{
					m_container.insert(other.m_container.begin(),other.m_container.end());
				}

            private:
				// The blake3 hash is quite fat (256bit), so we don't actually store a full asset ref for comparison.
				// Assuming a uniform distribution of keys and perfect hashing, we'd expect a collision on average every 2^256 asset loads.
				// Or if you actually calculate the P(X>1) for any reasonable number of asset loads (k trials), the Poisson CDF will be pratically 0.
				std::unordered_map<core::blake3_hash_t,cached_t<AssetType>> m_container;
        };

		// A meta class to encompass all the Assets you might want to convert at once
        struct SInputs
        {
			// you need to tell us twice (first time by explicitly listing in `span_t` and having `uniqueCopyGroupID!=0`)
			// if an asset needs multiple copies (second time here to know who uses which copy)
			virtual inline size_t getDependantUniqueCopyGroupID(const asset::IAsset* user, const asset::IAsset* dependant) const
			{
				//assert(std::find(get<dependant->getAssetType()>(assets),retval)!=::end());
				return 0;
			}

			// Typed Range of Inputs of the same type
            template<asset::Asset AssetType>
            using span_t = std::span<const typename CCache<AssetType>::lookup_t>;

			// can be `nullptr` and even equal to `this`
			CAssetConverter* readCache = nullptr;

			// A type-sorted non-polymorphic list of "root assets"
			core::tuple_transform_t<span_t,supported_asset_types> assets = {};
        };
        struct SResults
        {
			public:
				inline SResults(SResults&&) = default;
				inline SResults(const SResults&) = delete;
				inline ~SResults() = default;
				inline SResults& operator=(const SResults&) = delete;
				inline SResults& operator=(SResults&&) = default;

				//
				inline bool reserveSuccess() const {return m_success;}

				// What queues you'll need to run the submit
				inline core::bitflag<IQueue::FAMILY_FLAGS> getRequiredQueueFlags() const {return m_queueFlags;}

				// for every entry in the input array, we have this mapped 1:1
				template<asset::Asset AssetType>
				using vector_t = core::vector<cached_t<AssetType>>;

				// until `convert` is called, this will only contain valid entries for items already found in `SInput::readCache`
				template<asset::Asset AssetType>
				std::span<const cached_t<AssetType>> getGPUObjects() const {return std::get<vector_t<AssetType>>(m_gpuObjects);}

			private:
				friend class CAssetConverter;

				inline SResults() = default;

				//
				core::tuple_transform_t<vector_t,supported_asset_types> m_gpuObjects = {};
				//
//				m_reservations;
				//
				core::bitflag<IQueue::FAMILY_FLAGS> m_queueFlags = IQueue::FAMILY_FLAGS::NONE;
				//
				bool m_success = true;
        };
		// First Pass: Explore the DAG of Assets and "gather" patch infos for creating/retrieving equivalent GPU Objects.
		NBL_API SResults reserve(const SInputs& inputs);

		//
		struct SConvertParams
		{
			// By default the compute queue will own everything after all transfer operations are complete.
			virtual inline SIntendedSubmitInfo& getFinalOwnerSubmit(IDeviceMemoryBacked* imageOrBuffer, const core::blake3_hash_t& createdFrom)
			{
				return compute;
			}

			// submits the buffered up cals 
			NBL_API ISemaphore::future_t<bool> autoSubmit();

			// can be `nullptr`, but for most usages should be equal to `this`
			CAssetConverter* writeCache = nullptr;
			SIntendedSubmitInfo transfer = {};
			SIntendedSubmitInfo compute = {};
			IUtilities* utilities = nullptr;
		};
		// Second Pass: Actually create the GPU Objects
		NBL_API bool convert(SResults& reservations, SConvertParams& params);
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

        SCreationParams m_params;
		core::tuple_transform_t<CCache,supported_asset_types> m_caches;
};

}
#endif