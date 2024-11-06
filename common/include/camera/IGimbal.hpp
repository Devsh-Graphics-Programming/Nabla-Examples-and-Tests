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
            Scale = ScaleXInc | ScaleXDec | ScaleYInc | ScaleYDec | ScaleZInc | ScaleZDec
        };

        struct CRequestInfo
        {
            CRequestInfo() : type(None) {}
            CRequestInfo(VirtualEventType _type) : type(_type) {}
            ~CRequestInfo() = default;

            VirtualEventType type;
            bool active = false;
        };

        using manipulation_encode_t = float64_t;
        
        VirtualEventType type;
        manipulation_encode_t magnitude;

        static inline constexpr auto VirtualEventsTypeTable = []()
        {
            std::array<VirtualEventType, EventsCount> output;

            for (uint16_t i = 0u; i < EventsCount; ++i)
                output[i] = static_cast<VirtualEventType>(i);

            return output;
        }();
    };

    template<typename T>
    requires is_any_of_v<T, float32_t, float64_t>
    class IGimbal
    {
    public:
        using precision_t = T;

        struct VirtualImpulse
        {
            vector<precision_t, 3u> dVirtualTranslate { 0.0f }, dVirtualRotation { 0.0f }, dVirtualScale { 0.0f };
        };

        template <uint32_t AllowedEvents>
        VirtualImpulse accumulate(std::span<const CVirtualGimbalEvent> virtualEvents, const vector<precision_t, 3u>& gRightOverride, const vector<precision_t, 3u>& gUpOverride, const vector<precision_t, 3u>& gForwardOverride)
        {
            VirtualImpulse impulse;

            const auto& gRight = gRightOverride, gUp = gUpOverride, gForward = gForwardOverride;

            for (const auto& event : virtualEvents)
            {
                // translation events
                if constexpr (AllowedEvents & CVirtualGimbalEvent::MoveForward)
                    if (event.type == CVirtualGimbalEvent::MoveForward)
                        impulse.dVirtualTranslate += gForward * static_cast<precision_t>(event.magnitude);

                if constexpr (AllowedEvents & CVirtualGimbalEvent::MoveBackward)
                    if (event.type == CVirtualGimbalEvent::MoveBackward)
                        impulse.dVirtualTranslate -= gForward * static_cast<precision_t>(event.magnitude);

                if constexpr (AllowedEvents & CVirtualGimbalEvent::MoveRight)
                    if (event.type == CVirtualGimbalEvent::MoveRight)
                        impulse.dVirtualTranslate += gRight * static_cast<precision_t>(event.magnitude);

                if constexpr (AllowedEvents & CVirtualGimbalEvent::MoveLeft)
                    if (event.type == CVirtualGimbalEvent::MoveLeft)
                        impulse.dVirtualTranslate -= gRight * static_cast<precision_t>(event.magnitude);

                if constexpr (AllowedEvents & CVirtualGimbalEvent::MoveUp)
                    if (event.type == CVirtualGimbalEvent::MoveUp)
                        impulse.dVirtualTranslate += gUp * static_cast<precision_t>(event.magnitude);

                if constexpr (AllowedEvents & CVirtualGimbalEvent::MoveDown)
                    if (event.type == CVirtualGimbalEvent::MoveDown)
                        impulse.dVirtualTranslate -= gUp * static_cast<precision_t>(event.magnitude);

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
                        impulse.dVirtualScale.x += static_cast<precision_t>(event.magnitude);

                if constexpr (AllowedEvents & CVirtualGimbalEvent::ScaleXDec)
                    if (event.type == CVirtualGimbalEvent::ScaleXDec)
                        impulse.dVirtualScale.x -= static_cast<precision_t>(event.magnitude);

                if constexpr (AllowedEvents & CVirtualGimbalEvent::ScaleYInc)
                    if (event.type == CVirtualGimbalEvent::ScaleYInc)
                        impulse.dVirtualScale.y += static_cast<precision_t>(event.magnitude);

                if constexpr (AllowedEvents & CVirtualGimbalEvent::ScaleYDec)
                    if (event.type == CVirtualGimbalEvent::ScaleYDec)
                        impulse.dVirtualScale.y -= static_cast<precision_t>(event.magnitude);

                if constexpr (AllowedEvents & CVirtualGimbalEvent::ScaleZInc)
                    if (event.type == CVirtualGimbalEvent::ScaleZInc)
                        impulse.dVirtualScale.z += static_cast<precision_t>(event.magnitude);

                if constexpr (AllowedEvents & CVirtualGimbalEvent::ScaleZDec)
                    if (event.type == CVirtualGimbalEvent::ScaleZDec)
                        impulse.dVirtualScale.z -= static_cast<precision_t>(event.magnitude);
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

        inline void setOrientation(const glm::quat& orientation)
        {
            assert(m_isManipulating); // TODO: log error and return without doing nothing

            if(m_orientation != orientation)
                m_counter++;

            m_orientation = glm::normalize(orientation);
            updateOrthonormalOrientationBase();
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

        void end()
        {
            m_isManipulating = false;
        }

        // Position of gimbal
        inline const auto& getPosition() const { return m_position; }

        // Orientation of gimbal
        inline const auto& getOrientation() const { return m_orientation; }

        // Orthonormal [getXAxis(), getYAxis(), getZAxis()] orientation matrix
        inline const auto& getOrthonornalMatrix() const { return m_orthonormal; }

        // Base "right" vector in orthonormal orientation basis (X-axis)
        inline const auto& getXAxis() const { return m_orthonormal[0u]; }

        // Base "up" vector in orthonormal orientation basis (Y-axis)
        inline const auto& getYAxis() const { return m_orthonormal[1u]; }

        // Base "forward" vector in orthonormal orientation basis (Z-axis)
        inline const auto& getZAxis() const { return m_orthonormal[2u]; }

        inline const auto& getManipulationCounter() { return m_counter; }
        inline bool isManipulating() const { return m_isManipulating; }

    private:
        inline void updateOrthonormalOrientationBase()
        {
            m_orthonormal = matrix<precision_t, 3, 3>(glm::mat3_cast(glm::normalize(m_orientation)));
        }

        vector<precision_t, 3u> m_position;
        glm::quat m_orientation; // TODO: precision
        matrix<precision_t, 3, 3> m_orthonormal;
        
        // Counts *performed* manipulations, a manipulation with 0 delta is not counted!
        size_t m_counter = {};

        // Records manipulation state
        bool m_isManipulating = false;

        // the fact ImGUIZMO has global context I don't like, however for IDs we can do a life-tracking trick and cast addresses which are unique & we don't need any global associative container to track them!
        const uintptr_t m_id;
    };
} // namespace nbl::hlsl

#endif // _NBL_IGIMBAL_HPP_