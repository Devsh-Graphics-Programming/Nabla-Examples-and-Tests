// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _I_CAMERA_HPP_
#define _I_CAMERA_HPP_

#include "camera/IGimbal.hpp"

namespace nbl::hlsl // TODO: DIFFERENT NAMESPACE
{

template<typename T>
class ICamera : virtual public core::IReferenceCounted
{ 
public:
	using precision_t = T;

    // Gimbal with view parameters representing a camera in world space
    class CGimbal : public IGimbal<precision_t>
    {
    public:
        using base_t = IGimbal<precision_t>;

        CGimbal(typename base_t::SCreationParameters&& parameters) : base_t(std::move(parameters)) {}
        ~CGimbal() = default;

        struct SView
        {
            matrix<precision_t, 3, 4> matrix = {};
            bool isLeftHandSystem = true;
        };

        inline void updateView()
        {
            if (base_t::getManipulationCounter())
            {
                const auto& gRight = base_t::getXAxis(), gUp = base_t::getYAxis(), gForward = base_t::getZAxis();

                // TODO: I think this should be set as request state, depending on the value we do m_view.matrix[2u] flip accordingly
                // m_view.isLeftHandSystem;

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

                const auto& position = base_t::getPosition();
                m_view.matrix[0u] = vector<precision_t, 4u>(gRight, -glm::dot(gRight, position));
                m_view.matrix[1u] = vector<precision_t, 4u>(gUp, -glm::dot(gUp, position));
                m_view.matrix[2u] = vector<precision_t, 4u>(gForward, -glm::dot(gForward, position));
            }
        }

        // Getter for gimbal's view
        inline const SView& getView() const { return m_view; }

    private:
        SView m_view;
    };

    ICamera() {}
	~ICamera() = default;

	// Returns a gimbal which *models the camera view*, note that a camera type implementation may have multiple gimbals under the hood
	virtual const CGimbal& getGimbal() = 0u;

    // Manipulates camera with view gimbal & virtual events
    virtual void manipulate(std::span<const CVirtualGimbalEvent> virtualEvents) = 0;

	// VirtualEventType bitmask for a camera view gimbal manipulation requests filtering
	virtual const uint32_t getAllowedVirtualEvents() = 0u;
};

}

#endif // _I_CAMERA_HPP_