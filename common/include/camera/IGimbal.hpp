#ifndef _NBL_IGIMBAL_HPP_
#define _NBL_IGIMBAL_HPP_

#include "CCameraMathUtilities.hpp"
#include "CVirtualGimbalEvent.hpp"

namespace nbl::core
{
    struct CReferenceTransform
    {
        hlsl::float64_t4x4 frame;
        hlsl::camera_quaternion_t<hlsl::float64_t> orientation = hlsl::CCameraMathUtilities::makeIdentityQuaternion<hlsl::float64_t>();
    };

    template<typename T>
    requires is_any_of_v<T, hlsl::float32_t, hlsl::float64_t>
    class IGimbal
    {
    public:
        using precision_t = T;
        using quaternion_t = hlsl::camera_quaternion_t<precision_t>;
        template<uint32_t N>
        using vector_t = hlsl::camera_vector_t<precision_t, N>;
        //! underlying type for world matrix (TRS)
        using model_matrix_t = hlsl::matrix<precision_t, 3, 4>;

        struct VirtualImpulse
        {
            vector_t<3u> dVirtualTranslate { 0.0f }, dVirtualRotation { 0.0f }, dVirtualScale { 1.0f };
        };

        //! Accumulates one frame of virtual events into a translation/rotation/scale impulse.
        template <uint32_t AllowedEvents>
        VirtualImpulse accumulate(std::span<const CVirtualGimbalEvent> virtualEvents, const vector_t<3u>& gRightOverride, const vector_t<3u>& gUpOverride, const vector_t<3u>& gForwardOverride)
        {
            VirtualImpulse impulse;

            for (const auto& event : virtualEvents)
            {
                assert(event.magnitude >= 0);

                // translation events
                if constexpr (AllowedEvents & CVirtualGimbalEvent::MoveRight)
                    if (event.type == CVirtualGimbalEvent::MoveRight)
                        impulse.dVirtualTranslate.x += static_cast<precision_t>(event.magnitude);

                if constexpr (AllowedEvents & CVirtualGimbalEvent::MoveLeft)
                    if (event.type == CVirtualGimbalEvent::MoveLeft)
                        impulse.dVirtualTranslate.x -= static_cast<precision_t>(event.magnitude);

                if constexpr (AllowedEvents & CVirtualGimbalEvent::MoveUp)
                    if (event.type == CVirtualGimbalEvent::MoveUp)
                        impulse.dVirtualTranslate.y += static_cast<precision_t>(event.magnitude);

                if constexpr (AllowedEvents & CVirtualGimbalEvent::MoveDown)
                    if (event.type == CVirtualGimbalEvent::MoveDown)
                        impulse.dVirtualTranslate.y -= static_cast<precision_t>(event.magnitude);

                if constexpr (AllowedEvents & CVirtualGimbalEvent::MoveForward)
                    if (event.type == CVirtualGimbalEvent::MoveForward)
                        impulse.dVirtualTranslate.z += static_cast<precision_t>(event.magnitude);

                if constexpr (AllowedEvents & CVirtualGimbalEvent::MoveBackward)
                    if (event.type == CVirtualGimbalEvent::MoveBackward)
                        impulse.dVirtualTranslate.z -= static_cast<precision_t>(event.magnitude);

                // rotation events
                if constexpr (AllowedEvents & CVirtualGimbalEvent::TiltUp)
                    if (event.type == CVirtualGimbalEvent::TiltUp)
                        impulse.dVirtualRotation.x += static_cast<precision_t>(event.magnitude);

                if constexpr (AllowedEvents & CVirtualGimbalEvent::TiltDown)
                    if (event.type == CVirtualGimbalEvent::TiltDown)
                        impulse.dVirtualRotation.x -= static_cast<precision_t>(event.magnitude);

                if constexpr (AllowedEvents & CVirtualGimbalEvent::PanRight)
                    if (event.type == CVirtualGimbalEvent::PanRight)
                        impulse.dVirtualRotation.y += static_cast<precision_t>(event.magnitude);

                if constexpr (AllowedEvents & CVirtualGimbalEvent::PanLeft)
                    if (event.type == CVirtualGimbalEvent::PanLeft)
                        impulse.dVirtualRotation.y -= static_cast<precision_t>(event.magnitude);

                if constexpr (AllowedEvents & CVirtualGimbalEvent::RollRight)
                    if (event.type == CVirtualGimbalEvent::RollRight)
                        impulse.dVirtualRotation.z += static_cast<precision_t>(event.magnitude);

                if constexpr (AllowedEvents & CVirtualGimbalEvent::RollLeft)
                    if (event.type == CVirtualGimbalEvent::RollLeft)
                        impulse.dVirtualRotation.z -= static_cast<precision_t>(event.magnitude);

                // scaling events
                if constexpr (AllowedEvents & CVirtualGimbalEvent::ScaleXInc)
                    if (event.type == CVirtualGimbalEvent::ScaleXInc)
                        impulse.dVirtualScale.x *= static_cast<precision_t>(event.magnitude);

                if constexpr (AllowedEvents & CVirtualGimbalEvent::ScaleXDec)
                    if (event.type == CVirtualGimbalEvent::ScaleXDec)
                        impulse.dVirtualScale.x *= static_cast<precision_t>(event.magnitude);

                if constexpr (AllowedEvents & CVirtualGimbalEvent::ScaleYInc)
                    if (event.type == CVirtualGimbalEvent::ScaleYInc)
                        impulse.dVirtualScale.y *= static_cast<precision_t>(event.magnitude);

                if constexpr (AllowedEvents & CVirtualGimbalEvent::ScaleYDec)
                    if (event.type == CVirtualGimbalEvent::ScaleYDec)
                        impulse.dVirtualScale.y *= static_cast<precision_t>(event.magnitude);

                if constexpr (AllowedEvents & CVirtualGimbalEvent::ScaleZInc)
                    if (event.type == CVirtualGimbalEvent::ScaleZInc)
                        impulse.dVirtualScale.z *= static_cast<precision_t>(event.magnitude);

                if constexpr (AllowedEvents & CVirtualGimbalEvent::ScaleZDec)
                    if (event.type == CVirtualGimbalEvent::ScaleZDec)
                        impulse.dVirtualScale.z *= static_cast<precision_t>(event.magnitude);
            }

            return impulse;
        }

        template <uint32_t AllowedEvents>
        VirtualImpulse accumulate(std::span<const CVirtualGimbalEvent> virtualEvents)
        {
            return accumulate<AllowedEvents>(virtualEvents, getXAxis(), getYAxis(), getZAxis());
        }

        struct SCreationParameters
        {
            vector_t<3u> position;
            quaternion_t orientation = hlsl::CCameraMathUtilities::makeIdentityQuaternion<precision_t>();
        };

        IGimbal(const IGimbal&) = default;
        IGimbal(IGimbal&&) noexcept = default;
        IGimbal& operator=(const IGimbal&) = default;
        IGimbal& operator=(IGimbal&&) noexcept = default;

        IGimbal(SCreationParameters&& parameters)
            : m_position(parameters.position), m_orientation(parameters.orientation)
        {
            updateOrthonormalOrientationBase();
        }

        void begin()
        {
            m_isManipulating = true;
            m_counter = 0u;
        }

        inline void setPosition(const vector_t<3u>& position)
        {
            assert(m_isManipulating);

            if (m_position != position)
                m_counter++;

            m_position = position;
        }

        inline void setScale(const vector_t<3u>& scale)
        {
            m_scale = scale;
        }

        inline void setOrientation(const quaternion_t& orientation)
        {
            assert(m_isManipulating);

            if (m_orientation.data != orientation.data)
                m_counter++;

            m_orientation = hlsl::CCameraMathUtilities::normalizeQuaternion(orientation);
            updateOrthonormalOrientationBase();
        }

        inline void transform(const CReferenceTransform& reference, const VirtualImpulse& impulse)
        {
            setOrientation(reference.orientation * hlsl::CCameraMathUtilities::makeQuaternionFromEulerRadiansYXZ(impulse.dVirtualRotation));
            setPosition(hlsl::mul(hlsl::float64_t4(impulse.dVirtualTranslate, 1), reference.frame).xyz);
        }

        inline void rotate(const vector_t<3u>& axis, float dRadians)
        {
            assert(m_isManipulating);

            if(dRadians)
                m_counter++;

            const auto dRotation = hlsl::CCameraMathUtilities::makeQuaternionFromAxisAngle(axis, static_cast<precision_t>(dRadians));
            m_orientation = hlsl::CCameraMathUtilities::normalizeQuaternion(dRotation * m_orientation);
            updateOrthonormalOrientationBase();
        }

        inline void move(vector_t<3u> delta)
        {
            assert(m_isManipulating);

            auto newPosition = m_position + delta;

            if (newPosition != m_position)
                m_counter++;

            m_position = newPosition;
        }

        inline void strafe(precision_t distance)
        {
            move(getXAxis() * distance);
        }

        inline void climb(precision_t distance)
        {
            move(getYAxis() * distance);
        }

        inline void advance(precision_t distance)
        {
            move(getZAxis() * distance);
        }

        inline void end()
        {
            m_isManipulating = false;
        }

        //! Position of gimbal in world space
        inline const auto& getPosition() const { return m_position; }

        //! Orientation of gimbal
        inline const auto& getOrientation() const { return m_orientation; }

        //! Scale transform component
        inline const auto& getScale() const { return m_scale; }

        //! World matrix (TRS)
        template<typename TRS = model_matrix_t>
        requires is_any_of_v<TRS, model_matrix_t, hlsl::matrix<T, 4u, 4u>>
        const TRS operator()() const
        { 
            const auto& position = getPosition();
            const auto& rotation = getOrthonornalMatrix();
            const auto& scale = getScale();

            if constexpr (std::is_same_v<TRS, model_matrix_t>)
            {
                return
                {
                    hlsl::camera_vector_t<precision_t, 4>(rotation[0] * scale.x, position.x),
                    hlsl::camera_vector_t<precision_t, 4>(rotation[1] * scale.y, position.y),
                    hlsl::camera_vector_t<precision_t, 4>(rotation[2] * scale.z, position.z)
                };
            }
            else
            {
                return
                {
                    hlsl::camera_vector_t<precision_t, 4>(rotation[0] * scale.x, T(0)),
                    hlsl::camera_vector_t<precision_t, 4>(rotation[1] * scale.y, T(0)),
                    hlsl::camera_vector_t<precision_t, 4>(rotation[2] * scale.z, T(0)),
                    hlsl::camera_vector_t<precision_t, 4>(position, T(1))
                };
            }
        }

        //! Orthonormal [getXAxis(), getYAxis(), getZAxis()] orientation matrix
        inline const auto& getOrthonornalMatrix() const { return m_orthonormal; }

        //! Base "right" vector in orthonormal orientation basis (X-axis)
        inline const auto& getXAxis() const { return m_orthonormal[0u]; }

        //! Base "up" vector in orthonormal orientation basis (Y-axis)
        inline const auto& getYAxis() const { return m_orthonormal[1u]; }

        //! Base "forward" vector in orthonormal orientation basis (Z-axis)
        inline const auto& getZAxis() const { return m_orthonormal[2u]; }

        //! Target vector in local space, alias for getZAxis()
        inline const auto getLocalTarget() const { return getZAxis(); }

        //! Target vector in world space
        inline const auto getWorldTarget() const { return getPosition() + getLocalTarget(); }

        //! Counts how many times a valid manipulation has been performed, the counter resets when begin() is called
        inline const auto& getManipulationCounter() { return m_counter; }

        //! Returns true if gimbal records a manipulation 
        inline bool isManipulating() const { return m_isManipulating; }

        bool extractReferenceTransform(CReferenceTransform* out, const hlsl::float64_t4x4* referenceFrame = nullptr)
        {
            if (not out)
                return false;

            if (referenceFrame)
            {
                if (!hlsl::CCameraMathUtilities::tryBuildRigidFrameFromTransform(*referenceFrame, out->frame, out->orientation))
                    return false;
            }
            else
            {
                out->orientation = getOrientation();
                out->frame = hlsl::CCameraMathUtilities::composeTransformMatrix(getPosition(), out->orientation);
            }

            return true;
        }

    private:
        inline void updateOrthonormalOrientationBase()
        {
            m_orthonormal = hlsl::CCameraMathUtilities::getQuaternionBasisMatrix(m_orientation);
        }

        //! Position of a gimbal in world space
        vector_t<3u> m_position;

        //! Normalized orientation of gimbal
        quaternion_t m_orientation;

        //! Scale transform component
        vector_t<3u> m_scale = { 1.f, 1.f , 1.f };

        //! Orthonormal basis reconstructed from the current orientation.
        hlsl::matrix<precision_t, 3, 3> m_orthonormal;

        //! Counter that increments for each performed manipulation, resets with each begin() call
        size_t m_counter = {};

        //! Tracks whether gimbal is currently in manipulation mode
        bool m_isManipulating = false;

    };
} // namespace nbl::core

#endif // _NBL_IGIMBAL_HPP_
