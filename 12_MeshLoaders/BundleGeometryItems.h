#ifndef _NBL_EXAMPLES_12_MESHLOADERS_BUNDLE_GEOMETRY_ITEMS_H_INCLUDED_
#define _NBL_EXAMPLES_12_MESHLOADERS_BUNDLE_GEOMETRY_ITEMS_H_INCLUDED_

#include "include/common.hpp"

#include "nbl/asset/interchange/SGeometryWriterCommon.h"

namespace meshloaders
{

struct BundleGeometryItem
{
    core::smart_refctd_ptr<const ICPUPolygonGeometry> geometry;
    hlsl::float32_t3x4 world = hlsl::math::linalg::identity<hlsl::float32_t3x4>();
};

inline bool collectBundleGeometryItems(
    const asset::SAssetBundle& bundle,
    core::vector<BundleGeometryItem>& out,
    const bool preserveTransforms)
{
    if (bundle.getContents().empty())
        return false;

    switch (bundle.getAssetType())
    {
    case IAsset::E_TYPE::ET_GEOMETRY:
    case IAsset::E_TYPE::ET_GEOMETRY_COLLECTION:
    case IAsset::E_TYPE::ET_SCENE:
        break;
    default:
        return false;
    }

    const auto identity = hlsl::math::linalg::identity<hlsl::float32_t3x4>();
    const bool useItemTransforms = preserveTransforms && bundle.getAssetType() == IAsset::E_TYPE::ET_SCENE;
    for (const auto& root : bundle.getContents())
    {
        if (!root)
            continue;
        const auto items = asset::SGeometryWriterCommon::collectPolygonGeometryWriteItems(root.get());
        for (const auto& item : items)
        {
            if (!item.geometry)
                continue;
            out.push_back({
                .geometry = core::smart_refctd_ptr<const ICPUPolygonGeometry>(const_cast<ICPUPolygonGeometry*>(item.geometry)),
                .world = useItemTransforms ? item.transform : identity
                });
        }
    }
    return !out.empty();
}

}

#endif
