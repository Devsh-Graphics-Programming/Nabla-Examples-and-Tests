#ifndef _NBL_CCUBE_PROJECTION_HPP_
#define _NBL_CCUBE_PROJECTION_HPP_

#include "ILinearProjection.hpp"

namespace nbl::hlsl
{

struct CCubeProjectionBase
{
    using projection_t = typename IProjection<float64_t4x4>;
    using projection_range_t = std::array<typename projection_t, 6u>;
    using base_t = ILinearProjection<typename projection_range_t>;
};

//! Class providing linear cube projections with projection matrix per face of a cube, each projection matrix represents a single view-port
class CCubeProjection : public CCubeProjectionBase::base_t
{
public:
    using base_t = typename CCubeProjectionBase::base_t;
    using projection_range_t = typename base_t::range_t;

    CCubeProjection(projection_range_t&& projections = {}) : base_t(std::move(projections)) {}

    projection_range_t& getCubeFaceProjections()
    {
        return base_t::getViewportProjections();
    }
};

} // nbl::hlsl namespace

#endif // _NBL_CCUBE_PROJECTION_HPP_