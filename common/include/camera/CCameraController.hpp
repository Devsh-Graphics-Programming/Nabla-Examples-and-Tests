#ifndef _NBL_C_CAMERA_CONTROLLER_HPP_
#define _NBL_C_CAMERA_CONTROLLER_HPP_

#include "IGimbalController.hpp"
#include "CGeneralPurposeGimbal.hpp"
#include "ICamera.hpp"

// TODO: DIFFERENT NAMESPACE
namespace nbl::hlsl
{
    //! Controls any type of camera with available controllers using virtual gimbal events in Local mode
    template<typename T>
    class CCameraController final : public IGimbalController
    {
    public:
        using IGimbalController::IGimbalController;
        using precision_t = T;

        using interface_camera_t = ICamera<precision_t>;
        using interface_gimbal_t = IGimbal<precision_t>;
        
        CCameraController(core::smart_refctd_ptr<interface_camera_t> camera)
            : m_camera(std::move(camera)), m_target(interface_gimbal_t::SCreationParameters({ .position = m_camera->getGimbal().getWorldTarget() })) {}
        ~CCameraController() {}

        const keyboard_to_virtual_events_t& getKeyboardMappingPreset() const override { return m_camera->getKeyboardMappingPreset(); }
        const mouse_to_virtual_events_t& getMouseMappingPreset() const override { return m_camera->getMouseMappingPreset(); }
        const imguizmo_to_virtual_events_t& getImguizmoMappingPreset() const override { return m_camera->getImguizmoMappingPreset(); }

        //! Manipulate the camera view gimbal by requesting a manipulation to its world target represented by a target gimbal, 
        //! on success it may change both the camera view gimbal & target gimbal
        bool manipulateTargetGimbal(SUpdateParameters parameters, std::chrono::microseconds nextPresentationTimestamp)
        {
            // TODO & note to self:
            // thats a little bit tricky -> a request of m_target manipulation which represents camera world target is a step where we consider only change of its 
            // position and translate that onto virtual orientation events we fire camera->manipulate with. Note that *we can fail* the manipulation because each 
            // camera type has some constraints on how it works right, however.. if any manipulation happens it means "target vector" changes and it doesn't matter
            // what camera type is bound to the camera controller! If so, on success we can simply update m_target gimbal accordingly and represent it nicely on the 
            // screen with gizmo (as we do for camera view gimbal in Drag & Drop mode) or whatever, otherwise we do nothing because it means we failed the gimbal view 
            // manipulation hence "target vector" did not change (its really the orientation which changes right, but an orientation change means target vector changes)

            // and whats nice is we can do it with ANY controller now
        }

        //! Manipulate the camera view gimbal directly, 
        //! on success it may change both the camera view gimbal & target gimbal
        bool manipulateViewGimbal(SUpdateParameters parameters, std::chrono::microseconds nextPresentationTimestamp)
        {
            // TODO & note to self:
            // and here there is a small difference because we don't map any target gimbal position change to virtual orientation events to request camera view
            // gimbal manipulation but we directly try to manipulate the view gimbal of our camera and if we success then we simply update our m_target gimbal accordingly 

            // and whats nice is we can do it with ANY controller now

            std::vector<CVirtualGimbalEvent> virtualEvents(0x45);
            uint32_t vCount;

            beginInputProcessing(nextPresentationTimestamp);
            {
                process(nullptr, vCount);

                if (virtualEvents.size() < vCount)
                    virtualEvents.resize(vCount);

                process(virtualEvents.data(), vCount, parameters);
            }
            endInputProcessing();

            bool manipulated = m_camera->manipulate({ virtualEvents.data(), vCount }, interface_camera_t::Local);

            if (manipulated)
            {
                // TODO: *any* manipulate success? -> update m_target
            }

            return manipulated;
        }

        inline const interface_camera_t* getCamera() { return m_camera.get(); }

    private:
        core::smart_refctd_ptr<interface_camera_t> m_camera;

        //! Represents the camera world target vector as gimbal we can manipulate
        CGeneralPurposeGimbal<precision_t> m_target;
    };

} // nbl::hlsl namespace

#endif // _NBL_C_CAMERA_CONTROLLER_HPP_
