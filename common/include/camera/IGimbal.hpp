#ifndef _NBL_IGIMBAL_HPP_
#define _NBL_IGIMBAL_HPP_

#include "glm/glm/ext/matrix_transform.hpp" // TODO: TEMPORARY!!! whatever used will be moved to cpp
#include "glm/glm/gtc/quaternion.hpp"
#include "nbl/builtin/hlsl/matrix_utils/transformation_matrix_utils.hlsl"

// TODO: DIFFERENT NAMESPACE
namespace nbl::hlsl
{
    struct CVirtualGimbalEvent
    {
        enum VirtualEventType : uint32_t
        {
            None = 0,

            // Individual events
            MoveForward = core::createBitmask({ 0 }),
            MoveBackward = core::createBitmask({ 1 }),
            MoveLeft = core::createBitmask({ 2 }),
            MoveRight = core::createBitmask({ 3 }),
            MoveUp = core::createBitmask({ 4 }),
            MoveDown = core::createBitmask({ 5 }),
            TiltUp = core::createBitmask({ 6 }),
            TiltDown = core::createBitmask({ 7 }),
            PanLeft = core::createBitmask({ 8 }),
            PanRight = core::createBitmask({ 9 }),
            RollLeft = core::createBitmask({ 10 }),
            RollRight = core::createBitmask({ 11 }),
            ScaleXInc = core::createBitmask({ 12 }),
            ScaleXDec = core::createBitmask({ 13 }),
            ScaleYInc = core::createBitmask({ 14 }),
            ScaleYDec = core::createBitmask({ 15 }),
            ScaleZInc = core::createBitmask({ 16 }),
            ScaleZDec = core::createBitmask({ 17 }),

            EventsCount = 18,

            // Grouped bitmasks
            Translate = MoveForward | MoveBackward | MoveLeft | MoveRight | MoveUp | MoveDown,
            Rotate = TiltUp | TiltDown | PanLeft | PanRight | RollLeft | RollRight,
            Scale = ScaleXInc | ScaleXDec | ScaleYInc | ScaleYDec | ScaleZInc | ScaleZDec,

            All = Translate | Rotate | Scale
        };

        using manipulation_encode_t = float64_t;
        
        VirtualEventType type = None;
        manipulation_encode_t magnitude = {};

        static constexpr std::string_view virtualEventToString(VirtualEventType event)
        {
            switch (event)
            {
                case MoveForward: return "MoveForward";
                case MoveBackward: return "MoveBackward";
                case MoveLeft: return "MoveLeft";
                case MoveRight: return "MoveRight";
                case MoveUp: return "MoveUp";
                case MoveDown: return "MoveDown";
                case TiltUp: return "TiltUp";
                case TiltDown: return "TiltDown";
                case PanLeft: return "PanLeft";
                case PanRight: return "PanRight";
                case RollLeft: return "RollLeft";
                case RollRight: return "RollRight";
                case ScaleXInc: return "ScaleXInc";
                case ScaleXDec: return "ScaleXDec";
                case ScaleYInc: return "ScaleYInc";
                case ScaleYDec: return "ScaleYDec";
                case ScaleZInc: return "ScaleZInc";
                case ScaleZDec: return "ScaleZDec";
                case Translate: return "Translate";
                case Rotate: return "Rotate";
                case Scale: return "Scale";
                case None: return "None";
                default: return "Unknown";
            }
        }

        static constexpr VirtualEventType stringToVirtualEvent(std::string_view event)
        {
            if (event == "MoveForward") return MoveForward;
            if (event == "MoveBackward") return MoveBackward;
            if (event == "MoveLeft") return MoveLeft;
            if (event == "MoveRight") return MoveRight;
            if (event == "MoveUp") return MoveUp;
            if (event == "MoveDown") return MoveDown;
            if (event == "TiltUp") return TiltUp;
            if (event == "TiltDown") return TiltDown;
            if (event == "PanLeft") return PanLeft;
            if (event == "PanRight") return PanRight;
            if (event == "RollLeft") return RollLeft;
            if (event == "RollRight") return RollRight;
            if (event == "ScaleXInc") return ScaleXInc;
            if (event == "ScaleXDec") return ScaleXDec;
            if (event == "ScaleYInc") return ScaleYInc;
            if (event == "ScaleYDec") return ScaleYDec;
            if (event == "ScaleZInc") return ScaleZInc;
            if (event == "ScaleZDec") return ScaleZDec;
            if (event == "Translate") return Translate;
            if (event == "Rotate") return Rotate;
            if (event == "Scale") return Scale;
            if (event == "None") return None;
            return None;
        }

        static inline constexpr auto VirtualEventsTypeTable = []()
        {
            std::array<VirtualEventType, EventsCount> output;

            for (uint16_t i = 0u; i < EventsCount; ++i)
                output[i] = static_cast<VirtualEventType>(core::createBitmask({ i }));

            return output;
        }();
    };

    struct CReferenceTransform
    {
        float64_t4x4 frame;
        glm::quat orientation;
    };

    template<typename T>
    requires is_any_of_v<T, float32_t, float64_t>
    class IGimbal
    {
    public:
        using precision_t = T;
        //! underlying type for world matrix (TRS)
        using model_matrix_t = matrix<precision_t, 3, 4>;

        struct VirtualImpulse
        {
            vector<precision_t, 3u> dVirtualTranslate { 0.0f }, dVirtualRotation { 0.0f }, dVirtualScale { 1.0f };
        };

        //! Accumulates virtual impulse given allowed virtual event bitmap. Input virtual events are already deltas with respect to some base frame, the utility filters the events & outputs the impulse
        template <uint32_t AllowedEvents>
        VirtualImpulse accumulate(std::span<const CVirtualGimbalEvent> virtualEvents, const vector<precision_t, 3u>& gRightOverride, const vector<precision_t, 3u>& gUpOverride, const vector<precision_t, 3u>& gForwardOverride)
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
            vector<precision_t, 3u> position;
            glm::quat orientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
        };

        IGimbal(const IGimbal&) = default;
        IGimbal(IGimbal&&) noexcept = default;
        IGimbal& operator=(const IGimbal&) = default;
        IGimbal& operator=(IGimbal&&) noexcept = default;

        IGimbal(SCreationParameters&& parameters)
            : m_position(parameters.position), m_orientation(parameters.orientation), m_id(reinterpret_cast<uintptr_t>(this))
        {
            updateOrthonormalOrientationBase();
        }

        inline const uintptr_t getID() const { return m_id; }

        void begin()
        {
            m_isManipulating = true;
            m_counter = 0u;
        }

        inline void setPosition(const vector<precision_t, 3u>& position)
        {
            assert(m_isManipulating); // TODO: log error and return without doing nothing

            if (m_position != position)
                m_counter++;

            m_position = position;
        }

        inline void setScale(const vector<precision_t, 3u>& scale)
        {
            m_scale = scale;
        }

        inline void setOrientation(const glm::quat& orientation)
        {
            assert(m_isManipulating); // TODO: log error and return without doing nothing

            if(m_orientation != orientation)
                m_counter++;

            m_orientation = glm::normalize(orientation);
            updateOrthonormalOrientationBase();
        }

        inline void transform(const CReferenceTransform& reference, const VirtualImpulse& impulse)
        {
            setOrientation(reference.orientation * glm::quat(glm::radians(impulse.dVirtualRotation)));
            setPosition(mul(float64_t4(impulse.dVirtualTranslate, 1), reference.frame).xyz);
        }

        inline void rotate(const vector<precision_t, 3u>& axis, float dRadians)
        {
            assert(m_isManipulating); // TODO: log error and return without doing nothing

            if(dRadians)
                m_counter++;

            glm::quat dRotation = glm::angleAxis(dRadians, axis);
            m_orientation = glm::normalize(dRotation * m_orientation);
            updateOrthonormalOrientationBase();
        }

        inline void move(vector<precision_t, 3u> delta)
        {
            assert(m_isManipulating); // TODO: log error and return without doing nothing

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
        requires is_any_of_v<TRS, model_matrix_t, matrix<T, 4u, 4u>>
        const TRS operator()() const
        { 
            const auto& position = getPosition();
            const auto& rotation = getOrthonornalMatrix();
            const auto& scale = getScale();

            if constexpr (is_same_v<TRS, model_matrix_t>)
            {
                return
                {
                    vector<precision_t, 4>(rotation[0] * scale.x, position.x),
                    vector<precision_t, 4>(rotation[1] * scale.y, position.y),
                    vector<precision_t, 4>(rotation[2] * scale.z, position.z)
                };
            }
            else
            {
                return
                {
                    vector<precision_t, 4>(rotation[0] * scale.x, T(0)),
                    vector<precision_t, 4>(rotation[1] * scale.y, T(0)),
                    vector<precision_t, 4>(rotation[2] * scale.z, T(0)),
                    vector<precision_t, 4>(position, T(1))
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

        bool extractReferenceTransform(CReferenceTransform* out, const float64_t4x4 const* referenceFrame = nullptr)
        {
            if (not out)
                return false;

            if (referenceFrame)
            {
                out->frame = *referenceFrame;
                if (not isOrthoBase(float64_t3(out->frame[0]), float64_t3(out->frame[1]), float64_t3(out->frame[2])))
                    return false;
            }
            else
            {
                out->frame = getMatrix3x3As4x4(getOrthonornalMatrix());
                out->frame[3] = float64_t4(getPosition(), 1);
            }

            out->orientation = glm::quat_cast(glm::dmat3{ out->frame[0], out->frame[1], out->frame[2] });

            return true;
        }

    private:
        inline void updateOrthonormalOrientationBase()
        {
            m_orthonormal = matrix<precision_t, 3, 3>(glm::mat3_cast(glm::normalize(m_orientation)));
        }

        //! Position of a gimbal in world space
        vector<precision_t, 3u> m_position;

        //! Normalized orientation of gimbal
        //! TODO: precision + replace with our "quat at home"
        glm::quat m_orientation;

        //! Scale transform component
        vector<precision_t, 3u> m_scale = { 1.f, 1.f , 1.f };

        //! Orthonormal base composed from "m_orientation" representing gimbal's "forward", "up" & "right" vectors in local space - basically it spans orientation space
        matrix<precision_t, 3, 3> m_orthonormal;

        //! Counter that increments for each performed manipulation, resets with each begin() call
        size_t m_counter = {};

        //! Tracks whether gimbal is currently in manipulation mode
        bool m_isManipulating = false;

        //! The fact ImGUIZMO has global context I don't like, however for IDs we can do a life-tracking trick and cast addresses which are unique & we don't need any global associative container to track them!
        const uintptr_t m_id;
    };
} // namespace nbl::hlsl

#endif // _NBL_IGIMBAL_HPP_