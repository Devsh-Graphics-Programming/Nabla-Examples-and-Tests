// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXAMPLES_COMMON_GEOMETRY_AABB_UTILITIES_INCLUDED_
#define _NBL_EXAMPLES_COMMON_GEOMETRY_AABB_UTILITIES_INCLUDED_

#include "nbl/asset/ICPUPolygonGeometry.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace nbl::examples::geometry
{

inline bool isFinite(const double value)
{
    return std::isfinite(value);
}

inline bool isFinite(const hlsl::float64_t3& value)
{
    return isFinite(value.x) && isFinite(value.y) && isFinite(value.z);
}

template<typename Scalar>
inline bool isValidAABB(const hlsl::shapes::AABB<3, Scalar>& aabb)
{
    return
        std::isfinite(static_cast<double>(aabb.minVx.x)) &&
        std::isfinite(static_cast<double>(aabb.minVx.y)) &&
        std::isfinite(static_cast<double>(aabb.minVx.z)) &&
        std::isfinite(static_cast<double>(aabb.maxVx.x)) &&
        std::isfinite(static_cast<double>(aabb.maxVx.y)) &&
        std::isfinite(static_cast<double>(aabb.maxVx.z)) &&
        (aabb.minVx.x <= aabb.maxVx.x) &&
        (aabb.minVx.y <= aabb.maxVx.y) &&
        (aabb.minVx.z <= aabb.maxVx.z);
}

template<typename Scalar>
inline hlsl::shapes::AABB<3, Scalar> translateAABB(const hlsl::shapes::AABB<3, Scalar>& aabb, const hlsl::vector<Scalar, 3>& translation)
{
    auto out = aabb;
    out.minVx += translation;
    out.maxVx += translation;
    return out;
}

template<typename Scalar>
inline hlsl::shapes::AABB<3, Scalar> scaleAABB(const hlsl::shapes::AABB<3, Scalar>& aabb, const Scalar scale)
{
    auto out = aabb;
    out.minVx *= scale;
    out.maxVx *= scale;
    return out;
}

inline hlsl::shapes::AABB<3, double> computeFiniteUsedPositionAABB(const asset::ICPUPolygonGeometry* geometry)
{
    auto aabb = hlsl::shapes::AABB<3, double>::create();
    if (!geometry)
        return aabb;

    const auto positionView = geometry->getPositionView();
    const uint64_t vertexCount = positionView.getElementCount();
    if (!vertexCount)
        return aabb;

    const auto indexView = geometry->getIndexView();
    const uint64_t vertexRefCount = geometry->getVertexReferenceCount();
    bool hasFiniteVertex = false;
    for (uint64_t i = 0u; i < vertexRefCount; ++i)
    {
        uint64_t vertexIx = i;
        if (indexView)
        {
            const auto* const ptr = static_cast<const uint8_t*>(indexView.getPointer(i));
            if (!ptr)
                continue;
            switch (indexView.composed.format)
            {
            case asset::EF_R16_UINT:
            {
                uint16_t index = 0u;
                memcpy(&index, ptr, sizeof(index));
                vertexIx = index;
            }
            break;
            case asset::EF_R32_UINT:
            {
                uint32_t index = 0u;
                memcpy(&index, ptr, sizeof(index));
                vertexIx = index;
            }
            break;
            default:
                continue;
            }
        }
        if (vertexIx >= vertexCount)
            continue;

        hlsl::float32_t3 decoded = {};
        positionView.decodeElement(vertexIx, decoded);
        const hlsl::float64_t3 point = {
            static_cast<double>(decoded.x),
            static_cast<double>(decoded.y),
            static_cast<double>(decoded.z)
        };
        if (!isFinite(point))
            continue;

        if (!hasFiniteVertex)
        {
            aabb.minVx = point;
            aabb.maxVx = point;
            hasFiniteVertex = true;
            continue;
        }

        aabb.minVx.x = std::min(aabb.minVx.x, point.x);
        aabb.minVx.y = std::min(aabb.minVx.y, point.y);
        aabb.minVx.z = std::min(aabb.minVx.z, point.z);
        aabb.maxVx.x = std::max(aabb.maxVx.x, point.x);
        aabb.maxVx.y = std::max(aabb.maxVx.y, point.y);
        aabb.maxVx.z = std::max(aabb.maxVx.z, point.z);
    }

    if (!hasFiniteVertex)
        return hlsl::shapes::AABB<3, double>::create();
    return aabb;
}

inline hlsl::shapes::AABB<3, double> fallbackUnitAABB()
{
    hlsl::shapes::AABB<3, double> fallback = hlsl::shapes::AABB<3, double>::create();
    fallback.minVx = hlsl::float64_t3(-1.0, -1.0, -1.0);
    fallback.maxVx = hlsl::float64_t3(1.0, 1.0, 1.0);
    return fallback;
}

} // namespace nbl::examples::geometry

#endif
