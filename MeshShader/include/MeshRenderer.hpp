#pragma once

#include "nbl/builtin/hlsl/math/linalg/fast_affine.hlsl"
#include "nbl/examples/geometry/SPushConstants.hlsl"

namespace nbl::examples
{

	enum class MeshletObjectTypes {
		Cube,
		Rectangle,
		Disk,
		Sphere,
		Cylinder,
		Cone,
		Icosphere,

		COUNT
	};
		//this is buffer data
	struct MeshletObjectData {
		uint32_t vertCount;
		uint32_t primCount;
		uint32_t objectType;
		uint32_t positionView;
		uint32_t normalView;
		uint32_t indexView;
	};
	struct MeshDataBuffer {
		//if gpuGeometry is nullptr or std::nullopt or whatever, then mesh object type is invalid, the CPU memory failed to transfer to GPU for whatever reason
		core::smart_refctd_ptr<const video::IGPUPolygonGeometry> gpuGeometry{};

		static constexpr std::size_t MaxObjectCount = static_cast<std::size_t>(MeshletObjectTypes::COUNT);
		static constexpr std::size_t MaxInstanceCount = 8; //for each object

		MeshletObjectData meshData[MaxObjectCount];
		hlsl::float32_t4x4 transforms[MaxInstanceCount];

		//remove index type to avoid branch in shader
		//asset::E_INDEX_TYPE indexType = asset::EIT_UNKNOWN;
	};


class MeshDebugRenderer final : public core::IReferenceCounted {
#define EXPOSE_NABLA_NAMESPACES \
		using namespace nbl::core; \
		using namespace nbl::system; \
		using namespace nbl::asset; \
		using namespace nbl::video

public:
	//
	constexpr static inline uint16_t VertexAttrubUTBDescBinding = 0;

	//
	struct SViewParams
	{
		SViewParams(const hlsl::float32_t4x4& _viewProj, std::array<uint32_t, MeshDataBuffer::MaxObjectCount> const& objectCounts);

		hlsl::float32_t4x4 viewProj;
		std::array<uint32_t, MeshDataBuffer::MaxObjectCount> objectCounts;
		//hlsl::float32_t3x3 normal;
	};
	constexpr static inline auto MissingView = hlsl::examples::geometry_creator_scene::SPushConstants::DescriptorCount;

	//
	struct SInstance
	{
		struct SPushConstants
		{
			NBL_CONSTEXPR_STATIC_INLINE uint32_t DescriptorCount = (0x1 << 16) - 1;

			hlsl::float32_t4x4 viewProj;
			uint32_t objectCount[MeshDataBuffer::MaxObjectCount];
		};
		inline SPushConstants computePushConstants(const SViewParams& viewParams) const	{
			SPushConstants ret{
				.viewProj = viewParams.viewProj
			};
			memcpy(ret.objectCount, viewParams.objectCounts.data(), viewParams.objectCounts.size() * sizeof(uint32_t));
			return ret;
		}

		hlsl::float32_t3x4 world;
	};

	static std::array<const core::smart_refctd_ptr<nbl::asset::IShader>, 3> CreateTestShader(asset::IAssetManager* assMan, video::IGPURenderpass* renderpass, const uint32_t subpassIX);

	//
	static core::smart_refctd_ptr<MeshDebugRenderer> create(asset::IAssetManager* assMan, video::IGPURenderpass* renderpass, const uint32_t subpassIX);

	//
	static inline core::smart_refctd_ptr<MeshDebugRenderer> create(asset::IAssetManager* assMan, video::IGPURenderpass* renderpass, const uint32_t subpassIX, const std::span<const video::IGPUPolygonGeometry* const> geometries)
	{
		auto retval = create(assMan,renderpass,subpassIX);
		if (retval)
			retval->addGeometries(geometries);
		return retval;
	}

	//
	struct SInitParams {

		core::smart_refctd_ptr<video::IGPUDescriptorSet> meshDescriptor;
		core::smart_refctd_ptr<video::SubAllocatedDescriptorSet> subAllocDS;//vertex and normal views
		core::smart_refctd_ptr<video::IGPUPipelineLayout> layout;
		core::smart_refctd_ptr<video::IGPUMeshPipeline> pipeline;
	};
	inline SInitParams& getInitParams() {return m_params;}

	//im not going to go thru every example to fix them up to use this static function instead, so im leaving the old one
	//device should be const* but im not going to fix it right now 
	//(scope creep)
		
	bool addGeometries(const std::span<const video::IGPUPolygonGeometry* const> geometries);

	void removeGeometry(const uint32_t ix, const video::ISemaphore::SWaitInfo& info);

	inline const auto& getGeometries() const {return m_geoms;}

	void render(video::IGPUCommandBuffer* cmdbuf, const SViewParams& viewParams) const;

	SInstance m_instance;

	//mesh layout
	//PVP vertices at set 0 binding 0
	//mesh data at set 1 binding 0
	//they should be in the same set but tiny bit slower (1 additional API call) for a tiny bit easier programming
	nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSetLayout> mesh_layout{};

	MeshDataBuffer m_geoms;
protected:
	inline MeshDebugRenderer(SInitParams&& _params) : m_params(std::move(_params)) {}
	inline ~MeshDebugRenderer()	{
		// clean shutdown, can also make SubAllocatedDescriptorSet resillient against that, and issue `device->waitIdle` if not everything is freed
		const_cast<video::ILogicalDevice*>(m_params.layout->getOriginDevice())->waitIdle();
		clearGeometries({});
	}
	void clearGeometries(const video::ISemaphore::SWaitInfo& info);

	inline void immediateDealloc(video::SubAllocatedDescriptorSet::value_type index)
	{
		video::IGPUDescriptorSet::SDropDescriptorSet dummy[1];
		m_params.subAllocDS->multi_deallocate(dummy,VertexAttrubUTBDescBinding,1,&index);
	}

	SInitParams m_params;
#undef EXPOSE_NABLA_NAMESPACES
};

}