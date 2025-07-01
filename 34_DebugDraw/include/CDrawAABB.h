// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

// TODO move this into nabla

#include "nbl/video/declarations.h"
#include "nbl/builtin/hlsl/cpp_compat.hlsl"

#ifndef _NBL_EXT_DRAW_AABB_H_
#define _NBL_EXT_DRAW_AABB_H_

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

    inline const SCreationParameters& getCreationParameters() const { return m_creationParams; }

    // records draw command for single AABB, user has to set pipeline outside
    bool renderSingle(video::IGPUCommandBuffer* commandBuffer);

    static std::array<hlsl::float32_t3, 24> getVerticesFromAABB(const core::aabbox3d<float>& aabb);

protected:
	DrawAABB(SCreationParameters&& _params);
	~DrawAABB() override;

private:
    SCreationParameters m_creationParams;
};
}

#endif
