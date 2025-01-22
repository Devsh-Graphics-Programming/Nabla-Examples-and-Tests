// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _CAMERA_IMPL_
#define _CAMERA_IMPL_

#include <nabla.h>
#include <iostream>
#include <cstdio>
#include <fstream>
#include <chrono>

class Camera 
{ 
public:
	Camera() = default;
	Camera(const nbl::core::vectorSIMDf& position, const nbl::core::vectorSIMDf& lookat, const nbl::core::matrix4SIMD& projection, float moveSpeed = 1.0f, float rotateSpeed = 1.0f, const nbl::core::vectorSIMDf& upVec = nbl::core::vectorSIMDf(0.0f, 1.0f, 0.0f), const nbl::core::vectorSIMDf& backupUpVec = nbl::core::vectorSIMDf(0.5f, 1.0f, 0.0f)) 
		: position(position)
		, initialPosition(position)
		, target(lookat)
		, initialTarget(lookat)
		, firstUpdate(true)
		, moveSpeed(moveSpeed)
		, rotateSpeed(rotateSpeed)
		, upVector(upVec)
		, backupUpVector(backupUpVec)
	{
		initDefaultKeysMap();
		allKeysUp();
		setProjectionMatrix(projection);
		recomputeViewMatrix();
	}

	~Camera() = default;

	enum E_CAMERA_MOVE_KEYS : uint8_t
	{
		ECMK_MOVE_FORWARD = 0,
		ECMK_MOVE_BACKWARD,
		ECMK_MOVE_LEFT,
		ECMK_MOVE_RIGHT,
		ECMK_COUNT,
	};

	inline void mapKeysToWASD()
	{
		keysMap[ECMK_MOVE_FORWARD] = nbl::ui::EKC_W;
		keysMap[ECMK_MOVE_BACKWARD] = nbl::ui::EKC_S;
		keysMap[ECMK_MOVE_LEFT] = nbl::ui::EKC_A;
		keysMap[ECMK_MOVE_RIGHT] = nbl::ui::EKC_D;
	}

	inline void mapKeysToArrows()
	{
		keysMap[ECMK_MOVE_FORWARD] = nbl::ui::EKC_UP_ARROW;
		keysMap[ECMK_MOVE_BACKWARD] = nbl::ui::EKC_DOWN_ARROW;
		keysMap[ECMK_MOVE_LEFT] = nbl::ui::EKC_LEFT_ARROW;
		keysMap[ECMK_MOVE_RIGHT] = nbl::ui::EKC_RIGHT_ARROW;
	}

	inline void mapKeysCustom(std::array<nbl::ui::E_KEY_CODE, ECMK_COUNT>& map) { keysMap = map; }

	inline const nbl::core::matrix4SIMD& getProjectionMatrix() const { return projMatrix; }
	inline const nbl::core::matrix3x4SIMD& getViewMatrix() const {	return viewMatrix; }
	inline const nbl::core::matrix4SIMD& getConcatenatedMatrix() const { return concatMatrix; }

	inline void setProjectionMatrix(const nbl::core::matrix4SIMD& projection)
	{
		projMatrix = projection;

		const auto hlslMatMap = *reinterpret_cast<const nbl::hlsl::float32_t4x4*>(&projMatrix); // TEMPORARY TILL THE CAMERA CLASS IS REFACTORED TO WORK WITH HLSL MATRICIES!
		{
			leftHanded = nbl::hlsl::determinant(hlslMatMap) < 0.f;
		}
		concatMatrix = nbl::core::matrix4SIMD::concatenateBFollowedByAPrecisely(projMatrix, nbl::core::matrix4SIMD(viewMatrix));
	}
	
	inline void setPosition(const nbl::core::vectorSIMDf& pos)
	{
		position.set(pos);
		recomputeViewMatrix();
	}
	
	inline const nbl::core::vectorSIMDf& getPosition() const { return position; }

	inline void setTarget(const nbl::core::vectorSIMDf& pos) 
	{
		target.set(pos);
		recomputeViewMatrix();
	}

	inline const nbl::core::vectorSIMDf& getTarget() const { return target; }

	inline void setUpVector(const nbl::core::vectorSIMDf& up) { upVector = up; }
	
	inline void setBackupUpVector(const nbl::core::vectorSIMDf& up) { backupUpVector = up; }

	inline const nbl::core::vectorSIMDf& getUpVector() const { return upVector; }
	
	inline const nbl::core::vectorSIMDf& getBackupUpVector() const { return backupUpVector; }

	inline const float getMoveSpeed() const { return moveSpeed; }

	inline void setMoveSpeed(const float _moveSpeed) { moveSpeed = _moveSpeed; }

	inline const float getRotateSpeed() const { return rotateSpeed; }

	inline void setRotateSpeed(const float _rotateSpeed) { rotateSpeed = _rotateSpeed; }

	inline void recomputeViewMatrix() 
	{
		nbl::core::vectorSIMDf pos = position;
		nbl::core::vectorSIMDf localTarget = nbl::core::normalize(target - pos);

		// if upvector and vector to the target are the same, we have a
		// problem. so solve this problem:
		nbl::core::vectorSIMDf up = nbl::core::normalize(upVector);
		nbl::core::vectorSIMDf cross = nbl::core::cross(localTarget, up);
		bool upVectorNeedsChange = nbl::core::lengthsquared(cross)[0] == 0;
		if (upVectorNeedsChange)
			up = nbl::core::normalize(backupUpVector);

		if (leftHanded)
			viewMatrix = nbl::core::matrix3x4SIMD::buildCameraLookAtMatrixLH(pos, target, up);
		else
			viewMatrix = nbl::core::matrix3x4SIMD::buildCameraLookAtMatrixRH(pos, target, up);
		concatMatrix = nbl::core::matrix4SIMD::concatenateBFollowedByAPrecisely(projMatrix, nbl::core::matrix4SIMD(viewMatrix));
	}

	inline bool getLeftHanded() const { return leftHanded; }

public:

	// return whether camera is moved by mouse
	bool mouseProcess(const nbl::ui::IMouseEventChannel::range_t& events)
	{
		bool cameraMoved = false;
		for (auto eventIt=events.begin(); eventIt!=events.end(); eventIt++)
		{
			auto ev = *eventIt;

			if(ev.type == nbl::ui::SMouseEvent::EET_CLICK && ev.clickEvent.mouseButton == nbl::ui::EMB_LEFT_BUTTON)
				if(ev.clickEvent.action == nbl::ui::SMouseEvent::SClickEvent::EA_PRESSED) 
					mouseDown = true;
				else if (ev.clickEvent.action == nbl::ui::SMouseEvent::SClickEvent::EA_RELEASED)
					mouseDown = false;

			if(ev.type == nbl::ui::SMouseEvent::EET_MOVEMENT && mouseDown) 
			{
				nbl::core::vectorSIMDf pos = getPosition();
				nbl::core::vectorSIMDf localTarget = getTarget() - pos;

				// Get Relative Rotation for localTarget in Radians
				float relativeRotationX, relativeRotationY;
				relativeRotationY = atan2(localTarget.X, localTarget.Z);
				const double z1 = nbl::core::sqrt(localTarget.X*localTarget.X + localTarget.Z*localTarget.Z);
				relativeRotationX = atan2(z1, localTarget.Y) - nbl::core::PI<float>()/2;
				
				constexpr float RotateSpeedScale = 0.003f; 
				relativeRotationX -= ev.movementEvent.relativeMovementY * rotateSpeed * RotateSpeedScale * -1.0f;
				float tmpYRot = ev.movementEvent.relativeMovementX * rotateSpeed * RotateSpeedScale * -1.0f;

				if (leftHanded)
					relativeRotationY -= tmpYRot;
				else
					relativeRotationY += tmpYRot;

				const double MaxVerticalAngle = nbl::core::radians<float>(88.0f);

				if (relativeRotationX > MaxVerticalAngle*2 && relativeRotationX < 2 * nbl::core::PI<float>()-MaxVerticalAngle)
					relativeRotationX = 2 * nbl::core::PI<float>()-MaxVerticalAngle;
				else
					if (relativeRotationX > MaxVerticalAngle && relativeRotationX < 2 * nbl::core::PI<float>()-MaxVerticalAngle)
						relativeRotationX = MaxVerticalAngle;

				localTarget.set(0,0, nbl::core::max(1.f, nbl::core::length(pos)[0]), 1.f);

				nbl::core::matrix3x4SIMD mat;
				mat.setRotation(nbl::core::quaternion(relativeRotationX, relativeRotationY, 0));
				mat.transformVect(localTarget);
				
				setTarget(localTarget + pos);

				cameraMoved = true;
			}
		}
		return cameraMoved;
	}

	// return whether camera is moved by keyboard
	bool keyboardProcess(const nbl::ui::IKeyboardEventChannel::range_t& events)
	{
		for(uint32_t k = 0; k < E_CAMERA_MOVE_KEYS::ECMK_COUNT; ++k)
			perActionDt[k] = 0.0;

		/*
		* If a Key was already being held down from previous frames
		* Compute with this assumption that the key will be held down for this whole frame as well,
		* And If an UP event was sent It will get subtracted it from this value. (Currently Disabled Because we Need better Oracle)
		*/

		bool cameraMoved = false;
		for(uint32_t k = 0; k < E_CAMERA_MOVE_KEYS::ECMK_COUNT; ++k) 
			if(keysDown[k]) 
			{
				auto timeDiff = std::chrono::duration_cast<std::chrono::milliseconds>(nextPresentationTimeStamp - lastVirtualUpTimeStamp).count();
				assert(timeDiff >= 0);
				perActionDt[k] += timeDiff;
				cameraMoved = true;
			}

		for (auto eventIt=events.begin(); eventIt!=events.end(); eventIt++)
		{
			const auto ev = *eventIt;
			
			// accumulate the periods for which a key was down
			const auto timeDiff = std::chrono::duration_cast<std::chrono::milliseconds>(nextPresentationTimeStamp - ev.timeStamp).count();
			assert(timeDiff >= 0);

			// handle camera movement
			for (const auto logicalKey : { ECMK_MOVE_FORWARD, ECMK_MOVE_BACKWARD, ECMK_MOVE_LEFT, ECMK_MOVE_RIGHT })
			{
				const auto code = keysMap[logicalKey];

				if (ev.keyCode == code)
				{
					if (ev.action == nbl::ui::SKeyboardEvent::ECA_PRESSED && !keysDown[logicalKey]) 
					{
						perActionDt[logicalKey] += timeDiff;
						keysDown[logicalKey] = true;
					}
					else if (ev.action == nbl::ui::SKeyboardEvent::ECA_RELEASED) 
					{
						// perActionDt[logicalKey] -= timeDiff; 
						keysDown[logicalKey] = false;
					}
				}
			}

			// handle reset to default state
			if (ev.keyCode == nbl::ui::EKC_HOME)
				if (ev.action == nbl::ui::SKeyboardEvent::ECA_RELEASED)
				{
					position = initialPosition;
					target = initialTarget;
					recomputeViewMatrix();
					cameraMoved = true;
				}
		}

		return cameraMoved;
	}

	void beginInputProcessing(std::chrono::microseconds _nextPresentationTimeStamp)
	{
		nextPresentationTimeStamp = _nextPresentationTimeStamp;
		return;
	}
	
	void endInputProcessing(std::chrono::microseconds _nextPresentationTimeStamp)
	{
		nbl::core::vectorSIMDf pos = getPosition();
		nbl::core::vectorSIMDf localTarget = getTarget() - pos;

		if (!firstUpdate)
		{
			nbl::core::vectorSIMDf movedir = localTarget;
			movedir.makeSafe3D();
			movedir = nbl::core::normalize(movedir);

			constexpr float MoveSpeedScale = 0.02f; 

			pos += movedir * perActionDt[E_CAMERA_MOVE_KEYS::ECMK_MOVE_FORWARD] * moveSpeed * MoveSpeedScale;
			pos -= movedir * perActionDt[E_CAMERA_MOVE_KEYS::ECMK_MOVE_BACKWARD] * moveSpeed * MoveSpeedScale;

			// strafing
		
			// if upvector and vector to the target are the same, we have a
			// problem. so solve this problem:
			nbl::core::vectorSIMDf up = nbl::core::normalize(upVector);
			nbl::core::vectorSIMDf cross = nbl::core::cross(localTarget, up);
			bool upVectorNeedsChange = nbl::core::lengthsquared(cross)[0] == 0;
			if (upVectorNeedsChange)
			{
				up = nbl::core::normalize(backupUpVector);
			}

			nbl::core::vectorSIMDf strafevect = localTarget;
			if (leftHanded)
				strafevect = nbl::core::cross(strafevect, up);
			else
				strafevect = nbl::core::cross(up, strafevect);

			strafevect = nbl::core::normalize(strafevect);

			pos += strafevect * perActionDt[E_CAMERA_MOVE_KEYS::ECMK_MOVE_LEFT] * moveSpeed * MoveSpeedScale;
			pos -= strafevect * perActionDt[E_CAMERA_MOVE_KEYS::ECMK_MOVE_RIGHT] * moveSpeed * MoveSpeedScale;
		}
		else
			firstUpdate = false;

		setPosition(pos);
		setTarget(localTarget+pos);

		lastVirtualUpTimeStamp = nextPresentationTimeStamp;
	}

private:

	inline void initDefaultKeysMap() { mapKeysToWASD(); }
	
	inline void allKeysUp() 
	{
		for (uint32_t i=0; i< E_CAMERA_MOVE_KEYS::ECMK_COUNT; ++i)
			keysDown[i] = false;

		mouseDown = false;
	}

private:
	nbl::core::vectorSIMDf initialPosition, initialTarget, position, target, upVector, backupUpVector; // TODO: make first 2 const + add default copy constructor
	nbl::core::matrix3x4SIMD viewMatrix;
	nbl::core::matrix4SIMD concatMatrix, projMatrix;

	float moveSpeed, rotateSpeed;
	bool leftHanded, firstUpdate = true, mouseDown = false;
	
	std::array<nbl::ui::E_KEY_CODE, ECMK_COUNT> keysMap = { {nbl::ui::EKC_NONE} }; // map camera E_CAMERA_MOVE_KEYS to corresponding Nabla key codes, by default camera uses WSAD to move
	// TODO: make them use std::array
	bool keysDown[E_CAMERA_MOVE_KEYS::ECMK_COUNT] = {};
	double perActionDt[E_CAMERA_MOVE_KEYS::ECMK_COUNT] = {}; // durations for which the key was being held down from lastVirtualUpTimeStamp(=last "guessed" presentation time) to nextPresentationTimeStamp

	std::chrono::microseconds nextPresentationTimeStamp, lastVirtualUpTimeStamp;
};

#endif // _CAMERA_IMPL_