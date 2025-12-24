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

	constexpr static inline auto MissingView = hlsl::examples::geometry_creator_scene::SPushConstants::DescriptorCount;

	//
	struct SInstance
	{
		struct SPushConstants
		{
			NBL_CONSTEXPR_STATIC_INLINE uint32_t DescriptorCount = (0x1 << 16) - 1;

			nbl::hlsl::float32_t4x4 viewProj;
			uint32_t vertCount;
		};

		hlsl::float32_t3x4 world;
	};

	static std::array<const core::smart_refctd_ptr<nbl::asset::IShader>, 2> CreateTestShader(asset::IAssetManager* assMan, video::IGPURenderpass* renderpass, const uint32_t subpassIX);

	//
	static core::smart_refctd_ptr<MeshDebugRenderer> create(asset::IAssetManager* assMan, video::IGPURenderpass* renderpass, const uint32_t subpassIX);
	//
	struct SInitParams {

		core::smart_refctd_ptr<video::IGPUDescriptorSet> meshDescriptor;
		core::smart_refctd_ptr<video::IGPUPipelineLayout> pipe_layout; //when im looking at it from outside the class i need to know what kind of layout this is
		core::smart_refctd_ptr<video::IGPUMeshPipeline> pipeline;
	};
	inline SInitParams& getInitParams() {return m_params;}

	//im not going to go thru every example to fix them up to use this static function instead, so im leaving the old one
	//device should be const* but im not going to fix it right now 
	//(scope creep)
		
	bool addGeometries();

	void removeGeometry(const uint32_t ix, const video::ISemaphore::SWaitInfo& info);

	inline const auto& getGeometries() const {return m_geoms;}

	void render(video::IGPUCommandBuffer* cmdbuf, nbl::hlsl::float32_t4x4 const& mvp) const;

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
		const_cast<video::ILogicalDevice*>(m_params.pipe_layout->getOriginDevice())->waitIdle();
		clearGeometries({});
	}
	void clearGeometries(const video::ISemaphore::SWaitInfo& info);

	SInitParams m_params;
#undef EXPOSE_NABLA_NAMESPACES
};

}