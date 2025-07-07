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
    struct SCachedCreationParameters
    {
        using streaming_buffer_t = video::StreamingTransientDataBufferST<core::allocator<uint8_t>>;

        static constexpr inline auto RequiredAllocateFlags = core::bitflag<video::IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS>(video::IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);
        static constexpr inline auto RequiredUsageFlags = core::bitflag(asset::IBuffer::EUF_STORAGE_BUFFER_BIT) | asset::IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;

        core::smart_refctd_ptr<video::IUtilities> utilities;

        //! optional, default MDI buffer allocated if not provided
        core::smart_refctd_ptr<streaming_buffer_t> streamingBuffer = nullptr;
    };
    
    struct SCreationParameters : SCachedCreationParameters
    {
        core::smart_refctd_ptr<asset::IAssetManager> assetManager = nullptr;
        system::path localInputCWD; // TODO replace when working from nbl/ext

        core::smart_refctd_ptr<video::IGPUPipelineLayout> pipelineLayout;
        core::smart_refctd_ptr<video::IGPURenderpass> renderpass = nullptr;
    };

    // creates an instance that draws one AABB via push constant
    static core::smart_refctd_ptr<DrawAABB> create(SCreationParameters&& params);

    // creates an instance that draws multiple AABBs using streaming buffer
    // TODO

    // creates default pipeline layout for push constant version
    static core::smart_refctd_ptr<video::IGPUPipelineLayout> createDefaultPipelineLayout(video::ILogicalDevice* device, const asset::SPushConstantRange& pcRange);

    // creates default pipeline layout for streaming version
    static core::smart_refctd_ptr<video::IGPUPipelineLayout> createDefaultPipelineLayout(video::ILogicalDevice* device);

    static core::smart_refctd_ptr<video::IGPUGraphicsPipeline> createDefaultPipeline(video::ILogicalDevice* device, video::IGPUPipelineLayout* layout, video::IGPURenderpass* renderpass, video::IGPUGraphicsPipeline::SShaderSpecInfo& vertex, video::IGPUGraphicsPipeline::SShaderSpecInfo& fragment);

    inline const SCachedCreationParameters& getCreationParameters() const { return m_cachedCreationParams; }

    // records draw command for single AABB, user has to set pipeline outside
    bool renderSingle(video::IGPUCommandBuffer* commandBuffer);

    bool render(video::IGPUCommandBuffer* commandBuffer, video::ISemaphore::SWaitInfo waitInfo, float* cameraMat3x4);

    static std::array<hlsl::float32_t3, 24> getVerticesFromAABB(const core::aabbox3d<float>& aabb);

    void addAABB(const core::aabbox3d<float>& aabb, const hlsl::float32_t3& color = { 1,0,0 });

    void clearAABBs();

protected:
	DrawAABB(SCreationParameters&& _params, core::smart_refctd_ptr<video::IGPUGraphicsPipeline> pipeline);
	~DrawAABB() override;

private:
    static core::smart_refctd_ptr<video::IGPUGraphicsPipeline> createPipeline(SCreationParameters& params);
    static bool createStreamingBuffer(SCreationParameters& params);

    std::vector<InstanceData> m_instances;
    std::array<hlsl::float32_t3, 24> m_unitAABBVertices;

    SCachedCreationParameters m_cachedCreationParams;

    core::smart_refctd_ptr<video::IGPUGraphicsPipeline> m_pipeline;
};
}

#endif
