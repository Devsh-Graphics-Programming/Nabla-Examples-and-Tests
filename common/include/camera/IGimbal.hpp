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
        //! Virtual event representing a gimbal manipulation
        enum VirtualEventType : uint8_t
        {
            // Strafe forward
            MoveForward = 0,

            // Strafe backward
            MoveBackward,

            // Strafe left
            MoveLeft,

            // Strafe right
            MoveRight,

            // Strafe up
            MoveUp,

            // Strafe down
            MoveDown,

            // Tilt the camera upward (pitch)
            TiltUp,

            // Tilt the camera downward (pitch)
            TiltDown,

            // Rotate the camera left around the vertical axis (yaw)
            PanLeft,

            // Rotate the camera right around the vertical axis (yaw)
            PanRight,

            // Roll the camera counterclockwise around the forward axis (roll)
            RollLeft,

            // Roll the camera clockwise around the forward axis (roll)
            RollRight,

            // TODO: scale events

            EventsCount
        };

        using manipulation_encode_t = float64_t;
        using keys_to_virtual_events_t = std::array<ui::E_KEY_CODE, EventsCount>;

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