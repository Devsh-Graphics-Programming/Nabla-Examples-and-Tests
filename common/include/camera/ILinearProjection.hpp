#ifndef _NBL_ILINEAR_PROJECTION_HPP_
#define _NBL_ILINEAR_PROJECTION_HPP_

#include "IProjection.hpp"

namespace nbl::hlsl
{

//! Interface class for linear projections range storage - a projection matrix represents single view-port
template<typename T>
class ILinearProjection : protected IProjection<T>
{
public:
    using base_t = typename IProjection<T>;
    using range_t = typename base_t::range_t;

    ILinearProjection(range_t&& matrices) : base_t(matrices) {}

protected:
    inline range_t& getViewportMatrices()
    {
        return base_t::m_projMatrices;
    }
};

} // nbl::hlsl namespace

#endif // _NBL_ILINEAR_PROJECTION_HPP_