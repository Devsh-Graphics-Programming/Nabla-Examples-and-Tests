// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

// TODO move this into nabla

#ifndef _NBL_EXT_DRAW_AABB_H_
#define _NBL_EXT_DRAW_AABB_H_

#include "nbl/video/declarations.h"
#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "../app_resources/common.hlsl"

namespace nbl::ext::drawdebug
{
class DrawAABB final : public core::IReferenceCounted
{
public:
    struct SCreationParameters
    {
        asset::SPushConstantRange pushConstantRange;
    };

    // creates an instance that draws one AABB via push constant
    static core::smart_refctd_ptr<DrawAABB> create(SCreationParameters&& params);

    // creates an instance that draws multiple AABBs using streaming buffer
    // TODO

    // creates default pipeline layout for push constant version
    static core::smart_refctd_ptr<video::IGPUPipelineLayout> createDefaultPipelineLayout(video::ILogicalDevice* device, const asset::SPushConstantRange& pcRange);

    static bool createDefaultPipeline(core::smart_refctd_ptr<video::IGPUGraphicsPipeline>* pipeline, video::ILogicalDevice* device, video::IGPUPipelineLayout* layout, video::IGPURenderpass* renderpass, video::IGPUGraphicsPipeline::SShaderSpecInfo& vertex, video::IGPUGraphicsPipeline::SShaderSpecInfo& fragment);

    inline const SCreationParameters& getCreationParameters() const { return m_creationParams; }

    // records draw command for single AABB, user has to set pipeline outside
    bool renderSingle(video::IGPUCommandBuffer* commandBuffer);

    static std::array<hlsl::float32_t3, 24> getVerticesFromAABB(const core::aabbox3d<float>& aabb);

    void addAABB(const core::aabbox3d<float>& aabb, const hlsl::float32_t3& color = { 1,0,0 });

protected:
	DrawAABB(SCreationParameters&& _params);
	~DrawAABB() override;

private:
    SCreationParameters m_creationParams;

    std::vector<InstanceData> m_instances;
    std::array<hlsl::float32_t3, 24> m_unitVertices;
    constexpr static inline core::aabbox3d<float> UnitAABB = core::aabbox3d<float>({ 0, 0, 0 }, { 1, 1, 1 });
};
}

#endif
