#ifndef _NBL_CCUBE_PROJECTION_HPP_
#define _NBL_CCUBE_PROJECTION_HPP_

#include "IRange.hpp"
#include "IPerspectiveProjection.hpp"

namespace nbl::hlsl
{

/**
* @brief A projection where each cube face is a perspective quad we project onto.
*
* Represents a cube projection given direction vector where each face of 
* the cube is treated as a quad. The projection onto the cube is done using
* these quads and each face has its own unique pre-transform and
* view-port linear matrix.
*/
class CCubeProjection final : public IPerspectiveProjection, public IProjection
{
public:
    //! Represents six face identifiers of a cube.
    enum CubeFaces : uint8_t
    {
        //! Cube face in the +X base direction
        PositiveX = 0,

        //! Cube face in the -X base direction
        NegativeX,

        //! Cube face in the +Y base direction
        PositiveY,

        //! Cube face in the -Y base direction
        NegativeY,

        //! Cube face in the +Z base direction
        PositiveZ,

        //! Cube face in the -Z base direction
        NegativeZ,

        CubeFacesCount
    };

    inline static core::smart_refctd_ptr<CCubeProjection> create(core::smart_refctd_ptr<ICamera>&& camera)
    {
        if (!camera)
            return nullptr;

        return core::smart_refctd_ptr<CCubeProjection>(new CCubeProjection(core::smart_refctd_ptr(camera)), core::dont_grab);
    }

    virtual std::span<const ILinearProjection::CProjection> getLinearProjections() const override
    {
        return { reinterpret_cast<const ILinearProjection::CProjection*>(m_quads.data()), m_quads.size() };
    }

    void transformCube()
    {
        // TODO: update m_quads
    }

    virtual ProjectionType getProjectionType() const override { return ProjectionType::Cube; }

    virtual void project(const projection_vector_t& vecToProjectionSpace, projection_vector_t& output) const override
    {
        auto direction = normalize(vecToProjectionSpace);

        // TODO: project onto cube using quads representing faces
    }

    virtual bool unproject(const projection_vector_t& vecFromProjectionSpace, projection_vector_t& output) const override
    {
        // TODO: return back direction vector?
    }

    template<CubeFaces FaceIx>
    requires (FaceIx != CubeFacesCount)
    inline const CProjection& getProjectionQuad()
    {
        return m_quads[FaceIx];
    }

private:
    CCubeProjection(core::smart_refctd_ptr<ICamera>&& camera)
        : IPerspectiveProjection(core::smart_refctd_ptr(camera)) {}
    virtual ~CCubeProjection() = default;

    std::array<CProjection, CubeFacesCount> m_quads;
};

} // nbl::hlsl namespace

#endif // _NBL_CCUBE_PROJECTION_HPP_