#ifndef _NBL_CCUBE_PROJECTION_HPP_
#define _NBL_CCUBE_PROJECTION_HPP_

#include "IRange.hpp"
#include "IQuadProjection.hpp"

namespace nbl::hlsl
{

// A projection where each cube face is a quad we project onto
class CCubeProjection final : public IQuadProjection
{
public:
    static inline constexpr auto CubeFaces = 6u;

    inline static core::smart_refctd_ptr<CCubeProjection> create(core::smart_refctd_ptr<ICamera>&& camera)
    {
        if (!camera)
            return nullptr;

        return core::smart_refctd_ptr<CCubeProjection>(new CCubeProjection(core::smart_refctd_ptr(camera)), core::dont_grab);
    }

    virtual std::span<const CProjection> getQuadProjections() const
    {
        return m_cubefaceProjections;
    }

private:
    CCubeProjection(core::smart_refctd_ptr<ICamera>&& camera)
        : IQuadProjection(core::smart_refctd_ptr(camera)) {}
    virtual ~CCubeProjection() = default;

    std::array<CProjection, CubeFaces> m_cubefaceProjections;
};

} // nbl::hlsl namespace

#endif // _NBL_CCUBE_PROJECTION_HPP_