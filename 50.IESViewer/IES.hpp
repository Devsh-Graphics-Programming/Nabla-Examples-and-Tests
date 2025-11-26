#ifndef _THIS_EXAMPLE_IES_HPP_
#define _THIS_EXAMPLE_IES_HPP_

// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/examples/examples.hpp"

NBL_EXPOSE_NAMESPACES

struct IES
{
    enum E_MODE : uint32_t
    {
        EM_CDC,         //! Candlepower Distribution Curve
        EM_IES_C,       //! IES Candela
        EM_SPERICAL_C,  //! Sperical coordinates
        EM_DIRECTION,   //! Sample direction
        EM_PASS_T_MASK, //! Test mask

        EM_SIZE
    };

    struct
    {
        smart_refctd_ptr<IGPUImageView> candela = nullptr, spherical = nullptr, direction = nullptr, mask = nullptr;
    } views;

    struct
    {
        smart_refctd_ptr<IGPUBuffer> vAngles = nullptr, hAngles = nullptr, data = nullptr;
    } buffers;

    SAssetBundle bundle;
    std::string key;

    float zDegree = 0.f;
    E_MODE mode = EM_CDC;

    const asset::CIESProfile* getProfile() const;
    video::IGPUImage* getActiveImage() const;

    static const char* modeToRS(E_MODE mode);
    static const char* symmetryToRS(CIESProfile::properties_t::LuminairePlanesSymmetry symmetry);

    template<IImage::LAYOUT newLayout, bool undefined = false>
    requires(newLayout == IImage::LAYOUT::GENERAL or newLayout == IImage::LAYOUT::READ_ONLY_OPTIMAL)
    static inline bool barrier(IGPUCommandBuffer* const cb, const std::span<video::IGPUImage*> images)
    {
        if (images.empty())
            return false;

        if (not cb)
            return false;

        using image_memory_barrier_t = IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier>;
        const IGPUImage::SSubresourceRange range =
        {
            .aspectMask = IGPUImage::E_ASPECT_FLAGS::EAF_COLOR_BIT,
            .baseMipLevel = 0u,
            .levelCount = 1u,
            .baseArrayLayer = 0u,
            .layerCount = 1u
        };

        std::vector<image_memory_barrier_t> imageBarriers(images.size());

        for (uint32_t i = 0; i < imageBarriers.size(); ++i)
        {
            auto& it = imageBarriers[i] =
            {
                .barrier = {.dep = {}},
                .image = images[i],
                .subresourceRange = range,
                .oldLayout = IImage::LAYOUT::UNDEFINED,
                .newLayout = newLayout
            };

            if constexpr (newLayout == IImage::LAYOUT::GENERAL)
            {
                // READ_ONLY_OPTIMAL -> GENERAL, RW
                it.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT;
                it.barrier.dep.srcAccessMask = ACCESS_FLAGS::SAMPLED_READ_BIT;
                it.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
                it.barrier.dep.dstAccessMask = ACCESS_FLAGS::STORAGE_WRITE_BIT;
                it.oldLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;
            }
            else if (newLayout == IImage::LAYOUT::READ_ONLY_OPTIMAL)
            {
                // GENERAL -> READ_ONLY_OPTIMAL, RO
                it.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
                it.barrier.dep.srcAccessMask = ACCESS_FLAGS::STORAGE_WRITE_BIT;
                it.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT;
                it.barrier.dep.dstAccessMask = ACCESS_FLAGS::SAMPLED_READ_BIT;
                it.oldLayout = IImage::LAYOUT::GENERAL;
            }

            if constexpr (undefined)
                it.oldLayout = IImage::LAYOUT::UNDEFINED; // transition for init
        }

        return cb->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .memBarriers = {}, .bufBarriers = {}, .imgBarriers = imageBarriers });
    }

    template<IImage::LAYOUT newLayout, bool undefined = false>
    requires(newLayout == IImage::LAYOUT::GENERAL or newLayout == IImage::LAYOUT::READ_ONLY_OPTIMAL)
    static inline bool barrier(IGPUCommandBuffer* const cb, video::IGPUImage* image)
    {
        if (not image)
            return false;

        auto in = std::to_array({ image });
        return barrier<newLayout, undefined>(cb, in);
    }
};

#endif // _THIS_EXAMPLE_IES_HPP_