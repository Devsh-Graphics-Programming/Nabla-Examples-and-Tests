#ifndef _NBL_I_LINEAR_PROJECTION_HPP_
#define _NBL_I_LINEAR_PROJECTION_HPP_

#include "IProjection.hpp"
#include "ICamera.hpp"

namespace nbl::hlsl
{

/**
 * @brief Interface class for any custom linear projection transformation (matrix elements are already evaluated scalars)
 * referencing a camera, great for Perspective, Orthographic, Oblique, Axonometric and Shear projections
 */
class ILinearProjection : virtual public core::IReferenceCounted
{
protected:
    ILinearProjection(core::smart_refctd_ptr<ICamera>&& camera)
        : m_camera(core::smart_refctd_ptr(camera)) {}
    virtual ~ILinearProjection() = default;

    core::smart_refctd_ptr<ICamera> m_camera;
public:
    //! underlying type for linear world TRS matrix
    using model_matrix_t = typename decltype(m_camera)::pointee::CGimbal::model_matrix_t;

    //! underlying type for linear concatenated matrix
    using concatenated_matrix_t = float64_t4x4;

    //! underlying type for linear inverse of concatenated matrix
    using inv_concatenated_matrix_t = std::optional<float64_t4x4>;

    struct CProjection : public IProjection
    {
        using IProjection::IProjection;
        using projection_matrix_t = concatenated_matrix_t;
        using inv_projection_matrix_t = inv_concatenated_matrix_t;

        CProjection() : CProjection(projection_matrix_t(1)) {}
        CProjection(const projection_matrix_t& matrix) { setProjectionMatrix(matrix); }

        //! Returns P (Projection matrix)
        inline const projection_matrix_t& getProjectionMatrix() const { return m_projectionMatrix; }

        //! Returns P⁻¹ (Inverse of Projection matrix) *if it exists*
        inline const inv_projection_matrix_t& getInvProjectionMatrix() const { return m_invProjectionMatrix; }

        inline const std::optional<bool>& isProjectionLeftHanded() const { return m_isProjectionLeftHanded; }
        inline bool isProjectionSingular() const { return m_isProjectionSingular; }
        virtual ProjectionType getProjectionType() const override { return ProjectionType::Linear; }

        virtual void project(const projection_vector_t& vecToProjectionSpace, projection_vector_t& output) const override
        {
            output = mul(m_projectionMatrix, vecToProjectionSpace);
        }

        virtual bool unproject(const projection_vector_t& vecFromProjectionSpace, projection_vector_t& output) const override
        {
            if (m_isProjectionSingular)
                return false;

            output = mul(m_invProjectionMatrix.value(), vecFromProjectionSpace);

            return true;
        }

    protected:
        inline void setProjectionMatrix(const projection_matrix_t& matrix)
        {
            m_projectionMatrix = matrix;
            const auto det = hlsl::determinant(m_projectionMatrix);

            // we will allow you to lose a dimension since such a projection itself *may* 
            // be valid, however then you cannot un-project because the inverse doesn't exist!
            m_isProjectionSingular = not det;

            if (m_isProjectionSingular)
            {
                m_isProjectionLeftHanded = std::nullopt;
                m_invProjectionMatrix = std::nullopt;
            }
            else
            {
                m_isProjectionLeftHanded = det < 0.0;
                m_invProjectionMatrix = inverse(m_projectionMatrix);
            }
        }

    private:
        projection_matrix_t m_projectionMatrix;
        inv_projection_matrix_t m_invProjectionMatrix;
        std::optional<bool> m_isProjectionLeftHanded;
        bool m_isProjectionSingular;
    };

    virtual std::span<const CProjection> getLinearProjections() const = 0;
    
    inline bool setCamera(core::smart_refctd_ptr<ICamera>&& camera)
    {
        if (camera)
        {
            m_camera = camera;
            return true;
        }

        return false;
    }

    inline ICamera* getCamera()
    {
        return m_camera.get();
    }

    /**
    * @brief Computes Model View (MV) matrix
    * @param "model" is world TRS matrix
    * @return Returns MV matrix
    */
    inline concatenated_matrix_t getMV(const model_matrix_t& model) const
    {
        const auto& v = m_camera->getGimbal().getViewMatrix();
        return mul(getMatrix3x4As4x4(v), getMatrix3x4As4x4(model));
    }

    /**
    * @brief Computes Model View Projection (MVP) matrix
    * @param "projection" is linear projection
    * @param "model" is world TRS matrix
    * @return Returns MVP matrix
    */
    inline concatenated_matrix_t getMVP(const CProjection& projection, const model_matrix_t& model) const
    {
        const auto& v = m_camera->getGimbal().getViewMatrix();
        const auto& p = projection.getProjectionMatrix();
        auto mv = mul(getMatrix3x4As4x4(v), getMatrix3x4As4x4(model));
        return mul(p, mv);
    }

    /**
    * @brief Computes Model View Projection (MVP) matrix
    * @param "projection" is linear projection 
    * @param "mv" is Model View (MV) matrix 
    * @return Returns MVP matrix
    */
    inline concatenated_matrix_t getMVP(const CProjection& projection, const concatenated_matrix_t& mv) const
    {
        const auto& p = projection.getProjectionMatrix();
        return mul(p, mv);
    }

    /**
    * @brief Computes Inverse of Model View ((MV)⁻¹) matrix
    * @param "mv" is Model View (MV) matrix
    * @return Returns ((MV)⁻¹) matrix *if it exists*, otherwise returns std::nullopt
    */
    inline inv_concatenated_matrix_t getMVInverse(const model_matrix_t& model) const
    {
        const auto mv = getMV(model);
        if (auto det = determinant(mv); det)
            return inverse(mv);
        return std::nullopt;
    }

    /**
    * @brief Computes Inverse of Model View Projection ((MVP)⁻¹) matrix
    * @param "projection" is linear projection 
    * @param "model" is world TRS matrix
    * @return Returns ((MVP)⁻¹) matrix *if it exists*, otherwise returns std::nullopt
    */
    inline inv_concatenated_matrix_t getMVPInverse(const CProjection& projection, const model_matrix_t& model) const
    {
        const auto mvp = getMVP(projection, model);
        if (auto det = determinant(mvp); det)
            return inverse(mvp);
        return std::nullopt;
    }
};

} // nbl::hlsl namespace

#endif // _NBL_I_LINEAR_PROJECTION_HPP_