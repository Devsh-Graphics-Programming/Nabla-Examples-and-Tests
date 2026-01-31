#ifndef _THIS_EXAMPLE_IES_HPP_
#define _THIS_EXAMPLE_IES_HPP_

// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/examples/examples.hpp"
#include "nbl/system/to_string.h"

NBL_EXPOSE_NAMESPACES

struct IES
{
    enum E_MODE : uint32_t
    {
        EM_CDC,					//! Candlepower Distribution Curve
		EM_OCTAHEDRAL_MAP,      //! Candela Octahedral Map

        EM_SIZE
    };

    struct
    {
        smart_refctd_ptr<IGPUImageView> candelaOctahedralMap = nullptr;
    } views;

    struct
    {
        smart_refctd_ptr<IGPUBuffer> vAngles = nullptr, hAngles = nullptr, data = nullptr;		// allocation per ies
		SBufferBinding<IGPUBuffer> textureInfo;													// shared allocation for all ies
    } buffers;

    SAssetBundle bundle;
    std::string key;

    float zDegree = 0.f;

    const asset::CIESProfile* getProfile() const;
    video::IGPUImage* getActiveImage(E_MODE mode) const;

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
                it.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::ALL_GRAPHICS_BITS;
                it.barrier.dep.srcAccessMask = ACCESS_FLAGS::SHADER_READ_BITS;
                it.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
                it.barrier.dep.dstAccessMask = ACCESS_FLAGS::STORAGE_WRITE_BIT;
                it.oldLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;
            }
            else if (newLayout == IImage::LAYOUT::READ_ONLY_OPTIMAL)
            {
                // GENERAL -> READ_ONLY_OPTIMAL, RO
                it.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
                it.barrier.dep.srcAccessMask = ACCESS_FLAGS::STORAGE_WRITE_BIT;
                it.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::ALL_GRAPHICS_BITS;
                it.barrier.dep.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS;
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

namespace nbl::system::impl
{
template<>
struct to_string_helper<IES::E_MODE>
{
    static std::string __call(const IES::E_MODE mode)
    {
        switch (mode)
        {
        case IES::EM_CDC:
            return "Candlepower Distribution Curve";
        case IES::EM_OCTAHEDRAL_MAP:
            return "Candela Octahedral Map";
        default:
            return "ERROR (mode)";
        }
    }
};

template<>
struct to_string_helper<nbl::asset::CIESProfile::properties_t::LuminairePlanesSymmetry>
{
    static std::string __call(const nbl::asset::CIESProfile::properties_t::LuminairePlanesSymmetry symmetry)
    {
        switch (symmetry)
        {
        case nbl::asset::CIESProfile::properties_t::ISOTROPIC:
            return "ISOTROPIC";
        case nbl::asset::CIESProfile::properties_t::QUAD_SYMETRIC:
            return "QUAD_SYMETRIC";
        case nbl::asset::CIESProfile::properties_t::HALF_SYMETRIC:
            return "HALF_SYMETRIC";
        case nbl::asset::CIESProfile::properties_t::OTHER_HALF_SYMMETRIC:
            return "OTHER_HALF_SYMMETRIC";
        case nbl::asset::CIESProfile::properties_t::NO_LATERAL_SYMMET:
            return "NO_LATERAL_SYMMET";
        default:
            return "ERROR (symmetry)";
        }
    }
};

template<>
struct to_string_helper<nbl::asset::CIESProfile::properties_t::PhotometricType>
{
    static std::string __call(const nbl::asset::CIESProfile::properties_t::PhotometricType type)
    {
        switch (type)
        {
        case nbl::asset::CIESProfile::properties_t::TYPE_C:
            return "TYPE_C";
        case nbl::asset::CIESProfile::properties_t::TYPE_B:
            return "TYPE_B";
        case nbl::asset::CIESProfile::properties_t::TYPE_A:
            return "TYPE_A";
        case nbl::asset::CIESProfile::properties_t::TYPE_NONE:
        default:
            return "TYPE_NONE";
        }
    }
};

template<>
struct to_string_helper<nbl::asset::CIESProfile::properties_t::Version>
{
    static std::string __call(const nbl::asset::CIESProfile::properties_t::Version version)
    {
        switch (version)
        {
        case nbl::asset::CIESProfile::properties_t::V_1995:
            return "V_1995";
        case nbl::asset::CIESProfile::properties_t::V_2002:
            return "V_2002";
        default:
            return "V_UNKNOWN";
        }
    }
};
}

#endif // _THIS_EXAMPLE_IES_HPP_
