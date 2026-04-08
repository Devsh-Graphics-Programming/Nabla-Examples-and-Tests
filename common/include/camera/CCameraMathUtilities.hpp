#ifndef _C_CAMERA_MATH_UTILITIES_HPP_
#define _C_CAMERA_MATH_UTILITIES_HPP_

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

#include "nbl/builtin/hlsl/numbers.hlsl"
#include "nbl/builtin/hlsl/matrix_utils/transformation_matrix_utils.hlsl"

namespace nbl::hlsl
{

//! Camera-oriented math aliases and helpers built on top of Nabla `nbl::hlsl` types.
template<typename T>
inline T wrapAngleRad(T angle)
{
    constexpr T Pi = numbers::pi<T>;
    constexpr T TwoPi = Pi * static_cast<T>(2);

    angle = std::fmod(angle + Pi, TwoPi);
    if (angle < static_cast<T>(0))
        angle += TwoPi;
    return angle - Pi;
}

template<typename T>
inline T getWrappedAngleDistanceRadians(const T a, const T b)
{
    return std::abs(wrapAngleRad(a - b));
}

template<typename T>
inline T getWrappedAngleDistanceDegrees(const T a, const T b)
{
    constexpr T HalfTurn = static_cast<T>(180);
    constexpr T FullTurn = static_cast<T>(360);

    T angle = std::fmod(a - b + HalfTurn, FullTurn);
    if (angle < static_cast<T>(0))
        angle += FullTurn;
    return std::abs(angle - HalfTurn);
}

template<typename T>
inline bool isFiniteScalar(const T value)
{
    return std::isfinite(value);
}

template<typename T, uint32_t N>
using camera_vector_t = vector<T, N>;

template<typename T, uint32_t N, uint32_t M>
using camera_matrix_t = matrix<T, N, M>;

template<typename T>
using camera_quaternion_t = math::quaternion<T>;

template<typename T>
inline camera_quaternion_t<T> makeIdentityQuaternion();

template<typename T>
struct SRigidTransformComponents
{
    camera_vector_t<T, 3> translation = camera_vector_t<T, 3>(T(0));
    camera_quaternion_t<T> orientation = camera_quaternion_t<T>::create();
    camera_vector_t<T, 3> scale = camera_vector_t<T, 3>(T(1));
};

template<typename T>
inline bool tryExtractRigidTransformComponents(
    const camera_matrix_t<T, 4, 4>& transform,
    SRigidTransformComponents<T>& outComponents);

template<typename T>
inline camera_quaternion_t<T> makeIdentityQuaternion()
{
    return camera_quaternion_t<T>::create();
}

template<typename T>
inline camera_quaternion_t<T> makeQuaternionFromComponents(const T x, const T y, const T z, const T w)
{
    camera_quaternion_t<T> output;
    output.data = camera_vector_t<T, 4>(x, y, z, w);
    return output;
}

template<typename T>
inline camera_quaternion_t<T> normalizeQuaternion(const camera_quaternion_t<T>& q)
{
    return normalize(q);
}

template<typename T>
inline bool isFiniteQuaternion(const camera_quaternion_t<T>& q)
{
    return std::isfinite(q.data.x) &&
        std::isfinite(q.data.y) &&
        std::isfinite(q.data.z) &&
        std::isfinite(q.data.w);
}

template<typename T>
inline camera_quaternion_t<T> makeQuaternionFromAxisAngle(const camera_vector_t<T, 3>& axis, const T radians)
{
    return camera_quaternion_t<T>::create(axis, radians);
}

template<typename T>
inline camera_quaternion_t<T> makeQuaternionFromEulerRadians(const camera_vector_t<T, 3>& eulerRadians)
{
    return camera_quaternion_t<T>::create(eulerRadians.x, eulerRadians.y, eulerRadians.z);
}

template<typename T>
inline camera_quaternion_t<T> makeQuaternionFromEulerDegrees(const camera_vector_t<T, 3>& eulerDegrees)
{
    return makeQuaternionFromEulerRadians(camera_vector_t<T, 3>(
        radians(eulerDegrees.x),
        radians(eulerDegrees.y),
        radians(eulerDegrees.z)));
}

template<typename T>
inline camera_quaternion_t<T> makeQuaternionFromBasis(
    const camera_vector_t<T, 3>& right,
    const camera_vector_t<T, 3>& up,
    const camera_vector_t<T, 3>& forward)
{
    const auto safeNormalize = [](const camera_vector_t<T, 3>& v, const camera_vector_t<T, 3>& fallback)
    {
        const auto len = length(v);
        if (!std::isfinite(len) || len <= std::numeric_limits<T>::epsilon())
            return fallback;
        return v / len;
    };

    const auto canonicalForward = safeNormalize(forward, camera_vector_t<T, 3>(T(0), T(0), T(1)));

    auto canonicalRight = right - canonicalForward * dot(right, canonicalForward);
    canonicalRight = safeNormalize(
        canonicalRight,
        safeNormalize(cross(up, canonicalForward), camera_vector_t<T, 3>(T(1), T(0), T(0))));

    auto canonicalUp = cross(canonicalForward, canonicalRight);
    canonicalUp = safeNormalize(
        canonicalUp,
        safeNormalize(up - canonicalForward * dot(up, canonicalForward), camera_vector_t<T, 3>(T(0), T(1), T(0))));

    canonicalRight = safeNormalize(cross(canonicalUp, canonicalForward), canonicalRight);
    canonicalUp = safeNormalize(cross(canonicalForward, canonicalRight), canonicalUp);

    const camera_matrix_t<T, 3, 3> basis { canonicalRight, canonicalUp, canonicalForward };
    const auto desiredRight = canonicalRight;
    const auto desiredUp = canonicalUp;
    const auto desiredForward = canonicalForward;

    const auto scoreCandidate = [&](const camera_quaternion_t<T>& candidate)
    {
        if (!isFiniteQuaternion(candidate))
            return std::numeric_limits<T>::infinity();

        const auto normalizedCandidate = normalizeQuaternion(candidate);
        const auto rebuiltRight = normalizedCandidate.transformVector(camera_vector_t<T, 3>(T(1), T(0), T(0)), true);
        const auto rebuiltUp = normalizedCandidate.transformVector(camera_vector_t<T, 3>(T(0), T(1), T(0)), true);
        const auto rebuiltForward = normalizedCandidate.transformVector(camera_vector_t<T, 3>(T(0), T(0), T(1)), true);

        const T rightError = length(rebuiltRight - desiredRight);
        const T upError = length(rebuiltUp - desiredUp);
        const T forwardError = length(rebuiltForward - desiredForward);
        return rightError + upError + forwardError;
    };

    const auto quaternionFromMatrixFallback = [&](const camera_matrix_t<T, 3, 3>& m)
    {
        const T m00 = m[0][0];
        const T m11 = m[1][1];
        const T m22 = m[2][2];
        const T trace = m00 + m11 + m22;

        camera_quaternion_t<T> output = makeIdentityQuaternion<T>();
        if (trace > T(0))
        {
            const T scale = hlsl::sqrt(trace + T(1));
            const T invScale = T(0.5) / scale;
            output.data.x = (m[2][1] - m[1][2]) * invScale;
            output.data.y = (m[0][2] - m[2][0]) * invScale;
            output.data.z = (m[1][0] - m[0][1]) * invScale;
            output.data.w = scale * T(0.5);
        }
        else if (m00 >= m11 && m00 >= m22)
        {
            const T scale = hlsl::sqrt(T(1) + m00 - m11 - m22);
            const T invScale = T(0.5) / scale;
            output.data.x = scale * T(0.5);
            output.data.y = (m[0][1] + m[1][0]) * invScale;
            output.data.z = (m[2][0] + m[0][2]) * invScale;
            output.data.w = (m[2][1] - m[1][2]) * invScale;
        }
        else if (m11 >= m22)
        {
            const T scale = hlsl::sqrt(T(1) + m11 - m00 - m22);
            const T invScale = T(0.5) / scale;
            output.data.x = (m[0][1] + m[1][0]) * invScale;
            output.data.y = scale * T(0.5);
            output.data.z = (m[1][2] + m[2][1]) * invScale;
            output.data.w = (m[0][2] - m[2][0]) * invScale;
        }
        else
        {
            const T scale = hlsl::sqrt(T(1) + m22 - m00 - m11);
            const T invScale = T(0.5) / scale;
            output.data.x = (m[2][0] + m[0][2]) * invScale;
            output.data.y = (m[1][2] + m[2][1]) * invScale;
            output.data.z = scale * T(0.5);
            output.data.w = (m[1][0] - m[0][1]) * invScale;
        }
        return normalizeQuaternion(output);
    };

    const camera_matrix_t<T, 3, 3> transposedBasis = hlsl::transpose(basis);
    const camera_quaternion_t<T> castCandidates[] = {
        normalizeQuaternion(hlsl::_static_cast<camera_quaternion_t<T>>(basis)),
        normalizeQuaternion(hlsl::_static_cast<camera_quaternion_t<T>>(transposedBasis))
    };
    const camera_quaternion_t<T> fallbackCandidates[] = {
        quaternionFromMatrixFallback(basis),
        quaternionFromMatrixFallback(transposedBasis)
    };

    camera_quaternion_t<T> bestCandidate = makeIdentityQuaternion<T>();
    T bestScore = std::numeric_limits<T>::infinity();
    bool foundFiniteCandidate = false;

    for (const auto& candidate : castCandidates)
    {
        const T score = scoreCandidate(candidate);
        if (score < bestScore)
        {
            bestScore = score;
            bestCandidate = candidate;
            foundFiniteCandidate = true;
        }
    }

    if (!foundFiniteCandidate)
    {
        for (const auto& candidate : fallbackCandidates)
        {
            const T score = scoreCandidate(candidate);
            if (score < bestScore)
            {
                bestScore = score;
                bestCandidate = candidate;
                foundFiniteCandidate = true;
            }
        }
    }

    if (!foundFiniteCandidate)
        return makeIdentityQuaternion<T>();

    return normalizeQuaternion(bestCandidate);
}

template<typename T>
inline bool isFiniteVec3(const camera_vector_t<T, 3>& value)
{
    return isFiniteScalar(value.x) && isFiniteScalar(value.y) && isFiniteScalar(value.z);
}

template<typename T>
inline bool nearlyEqualScalar(const T a, const T b, const T epsilon)
{
    return std::abs(a - b) <= epsilon;
}

template<typename VecA, typename VecB, typename T>
inline bool nearlyEqualVec3(const VecA& a, const VecB& b, const T epsilon)
{
    const camera_vector_t<T, 3> delta(
        static_cast<T>(a.x - b.x),
        static_cast<T>(a.y - b.y),
        static_cast<T>(a.z - b.z));
    return length(delta) <= epsilon;
}

template<typename T>
inline camera_vector_t<T, 3> safeNormalizeVec3(const camera_vector_t<T, 3>& value, const camera_vector_t<T, 3>& fallback)
{
    const auto len = length(value);
    if (!isFiniteScalar(len) || len <= std::numeric_limits<T>::epsilon())
        return fallback;
    return value / len;
}

template<typename T>
inline camera_vector_t<T, 3> makeSphericalOffsetFromOrbit(const T orbitU, const T orbitV, const T distance)
{
    return camera_vector_t<T, 3>(
        hlsl::cos(orbitV) * hlsl::cos(orbitU) * distance,
        hlsl::cos(orbitV) * hlsl::sin(orbitU) * distance,
        hlsl::sin(orbitV) * distance);
}

template<typename T>
inline T getPlanarRadiusXZ(const camera_vector_t<T, 3>& offset)
{
    return length(camera_vector_t<T, 2>(offset.x, offset.z));
}

template<typename T>
inline T getPathDistance(const T radius, const T height)
{
    return length(camera_vector_t<T, 2>(radius, height));
}

template<typename T>
inline camera_vector_t<T, 3> makePathOffsetFromState(const T angle, const T radius, const T height)
{
    return camera_vector_t<T, 3>(hlsl::cos(angle) * radius, height, hlsl::sin(angle) * radius);
}

template<typename T>
inline bool sanitizePathState(T& angle, T& radius, T& height, const T minRadius)
{
    if (!isFiniteScalar(angle) || !isFiniteScalar(radius) || !isFiniteScalar(height))
        return false;

    radius = std::max(minRadius, radius);
    return isFiniteScalar(radius);
}

template<typename T>
inline bool tryScalePathStateDistance(
    const T desiredDistance,
    const T minRadius,
    T& radius,
    T& height,
    T* outAppliedDistance = nullptr)
{
    if (!isFiniteScalar(desiredDistance) || !isFiniteScalar(radius) || !isFiniteScalar(height))
        return false;

    const T currentDistance = getPathDistance(radius, height);
    constexpr T Epsilon = std::numeric_limits<T>::epsilon();
    if (currentDistance > Epsilon)
    {
        const T scale = desiredDistance / currentDistance;
        radius = std::max(minRadius, radius * scale);
        height *= scale;
    }
    else
    {
        radius = std::max(minRadius, desiredDistance);
        height = T(0);
    }

    if (outAppliedDistance)
        *outAppliedDistance = getPathDistance(radius, height);
    return isFiniteScalar(radius) && isFiniteScalar(height);
}

template<typename T>
inline bool tryBuildPathStateFromPosition(
    const camera_vector_t<T, 3>& targetPosition,
    const camera_vector_t<T, 3>& position,
    const T minRadius,
    T& outAngle,
    T& outRadius,
    T& outHeight)
{
    const auto offset = position - targetPosition;
    const auto radius = getPlanarRadiusXZ(offset);
    if (!isFiniteScalar(radius) || !isFiniteScalar(offset.y))
        return false;

    outAngle = hlsl::atan2(offset.z, offset.x);
    outRadius = std::max(minRadius, radius);
    outHeight = offset.y;
    return isFiniteScalar(outAngle) && isFiniteScalar(outRadius) && isFiniteScalar(outHeight);
}

template<typename T>
inline bool tryBuildLookAtOrientation(
    const camera_vector_t<T, 3>& position,
    const camera_vector_t<T, 3>& targetPosition,
    const camera_vector_t<T, 3>& preferredUp,
    camera_quaternion_t<T>& outOrientation)
{
    const auto toTarget = targetPosition - position;
    const auto toTargetLength = length(toTarget);
    if (!isFiniteScalar(toTargetLength) || toTargetLength <= std::numeric_limits<T>::epsilon())
        return false;

    const auto forward = toTarget / toTargetLength;
    auto up = safeNormalizeVec3(preferredUp, camera_vector_t<T, 3>(T(0), T(0), T(1)));
    auto right = cross(up, forward);
    if (!isFiniteVec3(right) || length(right) <= std::numeric_limits<T>::epsilon())
    {
        const auto fallbackUp = std::abs(forward.z) < T(0.99) ?
            camera_vector_t<T, 3>(T(0), T(0), T(1)) :
            camera_vector_t<T, 3>(T(0), T(1), T(0));
        right = cross(fallbackUp, forward);
        if (!isFiniteVec3(right) || length(right) <= std::numeric_limits<T>::epsilon())
            return false;
    }

    right = normalize(right);
    up = normalize(cross(forward, right));
    if (!isOrthoBase(right, up, forward))
        return false;

    outOrientation = makeQuaternionFromBasis(right, up, forward);
    return true;
}

template<typename T>
inline bool tryExtractRigidPoseFromTransform(
    const camera_matrix_t<T, 4, 4>& transform,
    camera_vector_t<T, 3>& outTranslation,
    camera_quaternion_t<T>& outOrientation)
{
    SRigidTransformComponents<T> components;
    if (!tryExtractRigidTransformComponents(transform, components))
        return false;

    outTranslation = components.translation;
    outOrientation = components.orientation;
    return true;
}

template<typename T>
inline bool tryBuildSphericalPoseFromOrbit(
    const camera_vector_t<T, 3>& targetPosition,
    const T orbitU,
    const T orbitV,
    const T distance,
    const T minDistance,
    const T maxDistance,
    camera_vector_t<T, 3>& outPosition,
    camera_quaternion_t<T>& outOrientation,
    T* outAppliedDistance = nullptr)
{
    if (!isFiniteScalar(orbitU) || !isFiniteScalar(orbitV) || !isFiniteScalar(distance))
        return false;

    const T appliedDistance = std::clamp(distance, minDistance, maxDistance);
    const auto spherePosition = makeSphericalOffsetFromOrbit(orbitU, orbitV, appliedDistance);
    const auto forward = safeNormalizeVec3(-spherePosition, camera_vector_t<T, 3>(T(0), T(0), T(1)));
    auto up = safeNormalizeVec3(
        camera_vector_t<T, 3>(
            -hlsl::sin(orbitV) * hlsl::cos(orbitU),
            -hlsl::sin(orbitV) * hlsl::sin(orbitU),
            hlsl::cos(orbitV)),
        camera_vector_t<T, 3>(T(0), T(0), T(1)));
    auto right = safeNormalizeVec3(cross(up, forward), camera_vector_t<T, 3>(T(1), T(0), T(0)));
    up = safeNormalizeVec3(cross(forward, right), up);
    right = safeNormalizeVec3(cross(up, forward), right);
    if (!isOrthoBase(right, up, forward))
        return false;

    outPosition = targetPosition + spherePosition;
    outOrientation = makeQuaternionFromBasis(right, up, forward);
    if (outAppliedDistance)
        *outAppliedDistance = appliedDistance;
    return true;
}

template<typename T>
inline bool tryBuildOrbitFromPosition(
    const camera_vector_t<T, 3>& targetPosition,
    const camera_vector_t<T, 3>& position,
    const T minDistance,
    const T maxDistance,
    T& outOrbitU,
    T& outOrbitV,
    T& outDistance)
{
    const auto offset = position - targetPosition;
    const auto distance = length(offset);
    if (!isFiniteScalar(distance) || distance <= std::numeric_limits<T>::epsilon())
        return false;

    outDistance = std::clamp(distance, minDistance, maxDistance);
    const auto local = offset / outDistance;
    outOrbitU = hlsl::atan2(local.y, local.x);
    outOrbitV = hlsl::asin(std::clamp(local.z, T(-1), T(1)));
    return isFiniteScalar(outOrbitU) && isFiniteScalar(outOrbitV) && isFiniteScalar(outDistance);
}

template<typename T>
inline camera_vector_t<T, 2> getPitchYawFromForwardVector(const camera_vector_t<T, 3>& forward)
{
    const T planarLength = length(camera_vector_t<T, 2>(forward.x, forward.z));
    return camera_vector_t<T, 2>(
        hlsl::atan2(planarLength, forward.y) - numbers::pi<T> * T(0.5),
        hlsl::atan2(forward.x, forward.z));
}

template<typename T>
inline bool tryBuildPathPoseFromState(
    const camera_vector_t<T, 3>& targetPosition,
    const T pathAngle,
    const T pathRadius,
    const T pathHeight,
    const T minRadius,
    const T minDistance,
    const T maxDistance,
    camera_vector_t<T, 3>& outPosition,
    camera_quaternion_t<T>& outOrientation,
    T* outAppliedDistance = nullptr,
    T* outOrbitU = nullptr,
    T* outOrbitV = nullptr)
{
    if (!isFiniteScalar(pathAngle) || !isFiniteScalar(pathRadius) || !isFiniteScalar(pathHeight))
        return false;

    const T appliedRadius = std::max(minRadius, pathRadius);
    const auto offset = makePathOffsetFromState(pathAngle, appliedRadius, pathHeight);

    T orbitU = T(0);
    T orbitV = T(0);
    T distance = T(0);
    if (!tryBuildOrbitFromPosition(targetPosition, targetPosition + offset, minDistance, maxDistance, orbitU, orbitV, distance))
        return false;
    if (!tryBuildSphericalPoseFromOrbit(targetPosition, orbitU, orbitV, distance, minDistance, maxDistance, outPosition, outOrientation, &distance))
        return false;

    if (outAppliedDistance)
        *outAppliedDistance = distance;
    if (outOrbitU)
        *outOrbitU = orbitU;
    if (outOrbitV)
        *outOrbitV = orbitV;
    return true;
}

template<typename T>
inline camera_vector_t<T, 3> rotateVectorByQuaternion(const camera_quaternion_t<T>& orientation, const camera_vector_t<T, 3>& vectorToRotate)
{
    return normalizeQuaternion(orientation).transformVector(vectorToRotate, true);
}

template<typename T>
inline camera_vector_t<T, 3> getQuaternionEulerRadians(const camera_quaternion_t<T>& orientation)
{
    const auto q = normalizeQuaternion(orientation);
    const T x = q.data.x;
    const T y = q.data.y;
    const T z = q.data.z;
    const T w = q.data.w;

    const T pitch = hlsl::atan2(
        T(2) * (y * z + w * x),
        w * w - x * x - y * y + z * z);
    const T yaw = hlsl::asin(std::clamp(
        T(-2) * (x * z - w * y),
        T(-1),
        T(1)));
    const T roll = hlsl::atan2(
        T(2) * (x * y + w * z),
        w * w + x * x - y * y - z * z);

    return camera_vector_t<T, 3>(pitch, yaw, roll);
}

template<typename T>
inline camera_vector_t<T, 3> getQuaternionEulerDegrees(const camera_quaternion_t<T>& orientation)
{
    const auto eulerRadians = getQuaternionEulerRadians(orientation);
    return camera_vector_t<T, 3>(
        degrees(eulerRadians.x),
        degrees(eulerRadians.y),
        degrees(eulerRadians.z));
}

template<typename T>
inline T getQuaternionAngularDistanceRadians(const camera_quaternion_t<T>& lhs, const camera_quaternion_t<T>& rhs)
{
    const auto lhsNormalized = normalizeQuaternion(lhs);
    const auto rhsNormalized = normalizeQuaternion(rhs);
    const T orientationDot = std::clamp(
        static_cast<T>(std::abs(dot(lhsNormalized.data, rhsNormalized.data))),
        T(0),
        T(1));
    return T(2) * hlsl::acos(orientationDot);
}

template<typename T>
inline T getQuaternionAngularDistanceDegrees(const camera_quaternion_t<T>& lhs, const camera_quaternion_t<T>& rhs)
{
    return degrees(getQuaternionAngularDistanceRadians(lhs, rhs));
}

template<typename T>
inline camera_quaternion_t<T> slerpQuaternion(const camera_quaternion_t<T>& lhs, const camera_quaternion_t<T>& rhs, const T alpha)
{
    return camera_quaternion_t<T>::slerp(normalizeQuaternion(lhs), normalizeQuaternion(rhs), alpha);
}

template<typename T>
inline camera_quaternion_t<T> inverseQuaternion(const camera_quaternion_t<T>& q)
{
    return inverse(q);
}

template<typename T>
inline camera_matrix_t<T, 3, 3> getQuaternionBasisMatrix(const camera_quaternion_t<T>& orientation)
{
    const auto q = normalizeQuaternion(orientation);
    return camera_matrix_t<T, 3, 3>(
        q.transformVector(camera_vector_t<T, 3>(T(1), T(0), T(0)), true),
        q.transformVector(camera_vector_t<T, 3>(T(0), T(1), T(0)), true),
        q.transformVector(camera_vector_t<T, 3>(T(0), T(0), T(1)), true));
}

template<typename T>
inline camera_vector_t<T, 3> getQuaternionEulerRadiansYXZ(const camera_quaternion_t<T>& orientation)
{
    const auto basis = getQuaternionBasisMatrix(orientation);
    const T yaw = hlsl::atan2(basis[2][0], basis[2][2]);
    const T c2 = hlsl::length(camera_vector_t<T, 2>(basis[0][1], basis[1][1]));
    const T pitch = hlsl::atan2(-basis[2][1], c2);
    const T s1 = hlsl::sin(yaw);
    const T c1 = hlsl::cos(yaw);
    const T roll = hlsl::atan2(
        s1 * basis[1][2] - c1 * basis[1][0],
        c1 * basis[0][0] - s1 * basis[0][2]);
    return camera_vector_t<T, 3>(pitch, yaw, roll);
}

template<typename T>
inline camera_vector_t<T, 3> getQuaternionEulerDegreesYXZ(const camera_quaternion_t<T>& orientation)
{
    const auto eulerRadians = getQuaternionEulerRadiansYXZ(orientation);
    return camera_vector_t<T, 3>(
        degrees(eulerRadians.x),
        degrees(eulerRadians.y),
        degrees(eulerRadians.z));
}

template<typename T>
inline camera_vector_t<T, 3> getWrappedEulerDistanceDegrees(
    const camera_vector_t<T, 3>& a,
    const camera_vector_t<T, 3>& b)
{
    return camera_vector_t<T, 3>(
        getWrappedAngleDistanceDegrees(a.x, b.x),
        getWrappedAngleDistanceDegrees(a.y, b.y),
        getWrappedAngleDistanceDegrees(a.z, b.z));
}

template<typename T>
inline T getMaxVectorComponent(const camera_vector_t<T, 3>& value)
{
    return std::max(value.x, std::max(value.y, value.z));
}

template<typename T>
inline camera_matrix_t<T, 4, 4> composeTransformMatrix(
    const camera_vector_t<T, 3>& translation,
    const camera_quaternion_t<T>& orientation,
    const camera_vector_t<T, 3>& scale = camera_vector_t<T, 3>(T(1)))
{
    camera_matrix_t<T, 4, 4> output = camera_matrix_t<T, 4, 4>(1);
    const auto basis = getQuaternionBasisMatrix(orientation);
    output[0] = camera_vector_t<T, 4>(basis[0] * scale.x, T(0));
    output[1] = camera_vector_t<T, 4>(basis[1] * scale.y, T(0));
    output[2] = camera_vector_t<T, 4>(basis[2] * scale.z, T(0));
    output[3] = camera_vector_t<T, 4>(translation, T(1));
    return output;
}

template<typename T>
inline bool tryExtractRigidTransformComponents(
    const camera_matrix_t<T, 4, 4>& transform,
    SRigidTransformComponents<T>& outComponents)
{
    outComponents.translation = camera_vector_t<T, 3>(transform[3].x, transform[3].y, transform[3].z);

    auto right = camera_vector_t<T, 3>(transform[0].x, transform[0].y, transform[0].z);
    auto up = camera_vector_t<T, 3>(transform[1].x, transform[1].y, transform[1].z);
    auto forward = camera_vector_t<T, 3>(transform[2].x, transform[2].y, transform[2].z);

    outComponents.scale = camera_vector_t<T, 3>(length(right), length(up), length(forward));

    if (!isFiniteVec3(outComponents.translation) || !isFiniteVec3(outComponents.scale))
        return false;

    constexpr T Epsilon = std::numeric_limits<T>::epsilon();
    if (outComponents.scale.x <= Epsilon || outComponents.scale.y <= Epsilon || outComponents.scale.z <= Epsilon)
        return false;

    right /= outComponents.scale.x;
    up /= outComponents.scale.y;
    forward /= outComponents.scale.z;
    if (!isOrthoBase(right, up, forward))
        return false;

    outComponents.orientation = makeQuaternionFromBasis(right, up, forward);
    return isFiniteQuaternion(outComponents.orientation);
}

template<typename T>
inline bool tryBuildRigidFrameFromTransform(
    const camera_matrix_t<T, 4, 4>& transform,
    camera_matrix_t<T, 4, 4>& outFrame,
    camera_quaternion_t<T>& outOrientation)
{
    SRigidTransformComponents<T> components;
    if (!tryExtractRigidTransformComponents(transform, components))
        return false;

    outOrientation = components.orientation;
    outFrame = composeTransformMatrix(components.translation, components.orientation);
    return true;
}

template<typename T>
inline bool decomposeTransformMatrix(
    const camera_matrix_t<T, 4, 4>& transform,
    camera_vector_t<T, 3>& outTranslation,
    camera_vector_t<T, 3>& outRotationEulerDegrees,
    camera_vector_t<T, 3>& outScale)
{
    SRigidTransformComponents<T> components;
    if (!tryExtractRigidTransformComponents(transform, components))
        return false;

    outTranslation = components.translation;
    outScale = components.scale;
    outRotationEulerDegrees = getQuaternionEulerDegrees(components.orientation);
    return std::isfinite(outRotationEulerDegrees.x) &&
        std::isfinite(outRotationEulerDegrees.y) &&
        std::isfinite(outRotationEulerDegrees.z);
}

} // namespace nbl::hlsl

#endif // _C_CAMERA_MATH_UTILITIES_HPP_
