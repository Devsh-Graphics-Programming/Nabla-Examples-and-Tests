#ifndef _NBL_I_CAMERA_CONTROLLER_HPP_
#define _NBL_I_CAMERA_CONTROLLER_HPP_

#include "IProjection.hpp"
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

        EventsCount
    };

    using manipulation_encode_t = float64_t;
    using keys_to_virtual_events_t = std::array<ui::E_KEY_CODE, CVirtualGimbalEvent::EventsCount>;

    VirtualEventType type;
    manipulation_encode_t magnitude;

    static inline constexpr auto VirtualEventsTypeTable = []()
    {
        std::array<VirtualEventType, EventsCount> output;

        for (uint16_t i = 0u; i < EventsCount; ++i)
        {
            output[i] = static_cast<VirtualEventType>(i);
        }

        return output;
    }();
};

template<typename T>
class ICameraController : virtual public core::IReferenceCounted
{
public:
    using matrix_precision_t = typename T;

    class CGimbal
    {
    public:
        struct SCreationParameters
        {
            float32_t3 position;
            glm::quat orientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
            bool withView = true;
        };

        // Gimbal's view matrix consists of an orthonormal basis (https://en.wikipedia.org/wiki/Orthonormal_basis) 
        // for orientation and a translation component that positions the world relative to the gimbal's position.
        // Any camera type is supposed to manipulate a position & orientation of a gimbal 
        // with "virtual events" which model its view bound to the camera
        struct SView
        {
            matrix<matrix_precision_t, 3, 4> matrix = {};
            bool isLeftHandSystem = true;
        };

        CGimbal(SCreationParameters&& parameters)
            : m_position(parameters.position), m_orientation(parameters.orientation)
        {
            updateOrthonormalOrientationBase();

            if (parameters.withView)
            {
                m_view = std::optional<SView>(SView{}); // RVO
                updateView();
            } 
        }

        void begin()
        {
            m_isManipulating = true;
            m_counter = 0u;
        }

        inline void setPosition(const float32_t3& position)
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

        inline void rotate(const float32_t3& axis, float dRadians)
        {
            assert(m_isManipulating); // TODO: log error and return without doing nothing

            if(dRadians)
                m_counter++;

            glm::quat dRotation = glm::angleAxis(dRadians, axis);
            m_orientation = glm::normalize(dRotation * m_orientation);
            updateOrthonormalOrientationBase();
        }

        inline void move(float32_t3 delta)
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

            if (m_counter > 0u)
                updateView();

            m_counter = 0u;
        }

        // Position of gimbal
        inline const float32_t3& getPosition() const { return m_position; }

        // Orientation of gimbal
        inline const glm::quat& getOrientation() const { return m_orientation; }

        // Orthonormal [getXAxis(), getYAxis(), getZAxis()] orientation matrix
        inline const float32_t3x3& getOrthonornalMatrix() const { return m_orthonormal; }

        // Base "right" vector in orthonormal orientation basis (X-axis)
        inline const float32_t3& getXAxis() const { return m_orthonormal[0u]; }

        // Base "up" vector in orthonormal orientation basis (Y-axis)
        inline const float32_t3& getYAxis() const { return m_orthonormal[1u]; }

        // Base "forward" vector in orthonormal orientation basis (Z-axis)
        inline const float32_t3& getZAxis() const { return m_orthonormal[2u]; }

        // Optional view of a gimbal
        inline const std::optional<SView>& getView() const { return m_view; }

        inline const size_t& getManipulationCounter() { return m_counter; }
        inline bool isManipulating() const { return m_isManipulating; }

    private:
        inline void updateOrthonormalOrientationBase()
        {
            m_orthonormal = matrix<matrix_precision_t, 3, 3>(glm::mat3_cast(glm::normalize(m_orientation)));
        }

        inline void updateView()
        {
            if (m_view.has_value()) // TODO: this could be templated + constexpr actually if gimbal doesn't init this on runtime depending on sth
            {
                auto& view = m_view.value();
                const auto& gRight = getXAxis(), gUp = getYAxis(), gForward = getZAxis();

                // TODO: I think I will provide convert utility allowing to go from one hand system to another, its just a matter to take care of m_view->matrix[2u] to perform a LH/RH flip
                // in general this should not know about projections which are now supposed to be independent and store reference to a camera (or own it)
                view.isLeftHandSystem = hlsl::determinant(m_orthonormal) < 0.0f;

                auto isNormalized = [](const auto& v, float epsilon) -> bool
                {
                    return glm::epsilonEqual(glm::length(v), 1.0f, epsilon);
                };

                auto isOrthogonal = [](const auto& a, const auto& b, float epsilon) -> bool
                {
                    return glm::epsilonEqual(glm::dot(a, b), 0.0f, epsilon);
                };

                auto isOrthoBase = [&](const auto& x, const auto& y, const auto& z, float epsilon = 1e-6f) -> bool
                {
                    return isNormalized(x, epsilon) && isNormalized(y, epsilon) && isNormalized(z, epsilon) &&
                        isOrthogonal(x, y, epsilon) && isOrthogonal(x, z, epsilon) && isOrthogonal(y, z, epsilon);
                };

                assert(isOrthoBase(gRight, gUp, gForward));

                view.matrix[0u] = vector<matrix_precision_t, 4u>(gRight, -glm::dot(gRight, m_position));
                view.matrix[1u] = vector<matrix_precision_t, 4u>(gUp, -glm::dot(gUp, m_position));
                view.matrix[2u] = vector<matrix_precision_t, 4u>(gForward, -glm::dot(gForward, m_position));
            }
        }

        float32_t3 m_position;
        glm::quat m_orientation;
        matrix<matrix_precision_t, 3, 3> m_orthonormal;

        // For a camera implementation at least one gimbal models its view but not all gimbals (if multiple) are expected to do so
        std::optional<SView> m_view = std::nullopt;
        
        // Counts *performed* manipulations, a manipulation with 0 delta is not counted!
        size_t m_counter = {};

        // Records manipulation state
        bool m_isManipulating = false;
    };

    ICameraController() {}

    // Binds key codes to virtual events, the mapKeys lambda will be executed with controller CVirtualGimbalEvent::keys_to_virtual_events_t table 
    void updateKeysToEvent(const std::function<void(CVirtualGimbalEvent::keys_to_virtual_events_t&)>& mapKeys)
    {
        mapKeys(m_keysToVirtualEvents);
    }

    // Manipulates camera with view gimbal & virtual events
    virtual void manipulate(std::span<const CVirtualGimbalEvent> virtualEvents) = 0;

    // TODO: *maybe* would be good to have a class interface for virtual event generators,
    // eg keyboard, mouse but maybe custom stuff too eg events from gimbal drag & drop

    // Processes keyboard events to generate virtual manipulation events, note that it doesn't make the manipulation itself!
    void processKeyboard(CVirtualGimbalEvent* output, uint32_t& count, std::span<const ui::SKeyboardEvent> events)
    {
        if (!output)
        {
            count = CVirtualGimbalEvent::EventsCount;
            return;
        }

        count = 0u;

        if (events.empty())
            return;

        const auto timestamp = getEventGenerationTimestamp();

        for (const auto virtualEventType : CVirtualGimbalEvent::VirtualEventsTypeTable)
        {
            const auto code = m_keysToVirtualEvents[virtualEventType];
            bool& keyDown = m_keysDown[virtualEventType];

            using virtual_key_state_t = std::tuple<ui::E_KEY_CODE /*physical key representing virtual key*/, bool /*is pressed*/, float64_t /*delta action*/>;

            auto updateVirtualState = [&]() -> virtual_key_state_t
            {
                virtual_key_state_t state = { ui::E_KEY_CODE::EKC_NONE, false, 0.f };

                for (const auto& ev : events) // TODO: improve the search
                {
                    if (ev.keyCode == code)
                    {
                        if (ev.action == nbl::ui::SKeyboardEvent::ECA_PRESSED && !keyDown)
                            keyDown = true;
                        else if (ev.action == nbl::ui::SKeyboardEvent::ECA_RELEASED)
                            keyDown = false;

                        const auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(timestamp - ev.timeStamp).count();
                        assert(dt >= 0);

                        state = std::make_tuple(code, keyDown, dt);
                        break;
                    }
                }

                return state;
            };

            const auto&& [physicalKey, isDown, dtAction] = updateVirtualState();

            if (physicalKey != ui::E_KEY_CODE::EKC_NONE)
                if (isDown)
                {
                    auto* virtualEvent = output + count;
                    assert(virtualEvent); // TODO: maybe just log error and return 0 count

                    virtualEvent->type = virtualEventType;
                    virtualEvent->magnitude = static_cast<float64_t>(dtAction);
                    ++count;
                }
        }
    }

    // Processes mouse events to generate virtual manipulation events, note that it doesn't make the manipulation itself!
    // Limited to Pan & Tilt rotation events, camera type implements how event magnitudes should be interpreted
    void processMouse(CVirtualGimbalEvent* output, uint32_t& count, std::span<const ui::SMouseEvent> events)
    {
        if (!output)
        {
            count = 2u;
            return;
        }

        count = 0u;

        if (events.empty())
            return;

        const auto timestamp = getEventGenerationTimestamp();
        double dYaw = {}, dPitch = {};

        for (const auto& ev : events)
            if (ev.type == nbl::ui::SMouseEvent::EET_MOVEMENT)
            {
                dYaw += ev.movementEvent.relativeMovementX;
                dPitch += ev.movementEvent.relativeMovementY;
            }

        if (dPitch)
        {
            auto* pitch = output + count;
            assert(pitch); // TODO: maybe just log error and return 0 count
            pitch->type = dPitch > 0.f ? CVirtualGimbalEvent::TiltUp : CVirtualGimbalEvent::TiltDown;
            pitch->magnitude = std::abs(dPitch);
            count++;
        }

        if (dYaw)
        {
            auto* yaw = output + count;
            assert(yaw); // TODO: maybe just log error and return 0 count
            yaw->type = dYaw > 0.f ? CVirtualGimbalEvent::PanRight : CVirtualGimbalEvent::PanLeft;
            yaw->magnitude = std::abs(dYaw);
            count++;
        }
    }

protected:
    virtual void initKeysToEvent() = 0;

private:
    CVirtualGimbalEvent::keys_to_virtual_events_t m_keysToVirtualEvents = { { ui::E_KEY_CODE::EKC_NONE } };
    bool m_keysDown[CVirtualGimbalEvent::EventsCount] = {};

    // exactly what our Nabla events do, actually I don't want users to pass timestamp since I know when it should be best to make a request -> just before generating events!
    // TODO: need to think about this
    inline std::chrono::microseconds getEventGenerationTimestamp() { return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()); }
};

#if 0 // TOOD: update
template<typename R>
concept GimbalRange = GeneralPurposeRange<R> && requires 
{
    requires ProjectionMatrix<typename std::ranges::range_value_t<R>::projection_t>;
    requires std::same_as<std::ranges::range_value_t<R>, typename ICameraController<typename std::ranges::range_value_t<R>::projection_t>::CGimbal>;
};

template<GimbalRange Range>
class IGimbalRange : public IRange<typename Range>
{
public:
    using base_t = IRange<typename Range>;
    using range_t = typename base_t::range_t;
    using gimbal_t = typename base_t::range_value_t;

    IGimbalRange(range_t&& gimbals) : base_t(std::move(gimbals)) {}
    inline const range_t& getGimbals() const { return base_t::m_range; }

protected:
    inline range_t& getGimbals() const { return base_t::m_range; }
};

// TODO NOTE: eg. "follow camera" should use GimbalRange<std::array<ICameraController<T>::CGimbal, 2u>>, 
// one per camera itself and one for target it follows
#endif

} // nbl::hlsl namespace

#endif // _NBL_I_CAMERA_CONTROLLER_HPP_