#ifndef _NBL_CCUBE_PROJECTION_HPP_
#define _NBL_CCUBE_PROJECTION_HPP_

#include "ILinearProjection.hpp"

namespace nbl::hlsl
{

struct CCubeProjectionBase
{
    using base_t = ILinearProjection<std::array<float64_t4x4, 6u>>;
};

//! Class providing linear cube projections with projection matrix per face of a cube, each projection matrix represents a single view-port
class CCubeProjection : public CCubeProjectionBase::base_t
{
public:
    using base_t = typename CCubeProjectionBase::base_t;
    using range_t = typename base_t::range_t;

    CCubeProjection(range_t&& matrices = {}) : base_t(std::move(matrices)) {}

    range_t& getCubeFaceProjectionMatrices()
    {
        return base_t::getViewportMatrices();
    }
};

} // nbl::hlsl namespace

#endif // _NBL_CCUBE_PROJECTION_HPP_