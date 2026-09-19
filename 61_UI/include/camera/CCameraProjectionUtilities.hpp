// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_THIS_EXAMPLE_CAMERA_PROJECTION_UTILITIES_HPP_INCLUDED_
#define _NBL_THIS_EXAMPLE_CAMERA_PROJECTION_UTILITIES_HPP_INCLUDED_

#include "nbl/ext/Cameras/IPlanarProjection.hpp"

struct CCameraProjectionUtilities final
{
    /// @brief Apply a camera-provided dynamic perspective FOV to one planar projection entry.
    static inline bool syncDynamicPerspectiveProjection(nbl::ext::cameras::ICamera* camera, nbl::ext::cameras::IPlanarProjection::CProjection& projection)
    {
        if (!camera)
            return false;

        const auto& params = projection.getParameters();
        if (params.m_type != nbl::ext::cameras::IPlanarProjection::CProjection::Perspective)
            return false;

        float dynamicFov = 0.0f;
        if (!camera->tryGetDynamicPerspectiveFov(dynamicFov))
            return false;

        projection.setPerspective(params.m_zNear, params.m_zFar, dynamicFov);
        return true;
    }
};

#endif // _NBL_THIS_EXAMPLE_CAMERA_PROJECTION_UTILITIES_HPP_INCLUDED_
