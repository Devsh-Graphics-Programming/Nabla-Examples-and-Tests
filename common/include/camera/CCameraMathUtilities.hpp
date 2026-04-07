#ifndef _C_CAMERA_MATH_UTILITIES_HPP_
#define _C_CAMERA_MATH_UTILITIES_HPP_

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

#include "nbl/builtin/hlsl/matrix_utils/transformation_matrix_utils.hlsl"

namespace nbl::hlsl
{

template<typename T>
inline T wrapAngleRad(T angle)
{
    constexpr T Pi = static_cast<T>(3.141592653589793238462643383279502884L);
    constexpr T TwoPi = Pi * static_cast<T>(2);

    angle = std::fmod(angle + Pi, TwoPi);
    if (angle < static_cast<T>(0))
        angle += TwoPi;
    return angle - Pi;
}

template<typename T, uint32_t N>
using camera_vector_t = vector<T, N>;

template<typename T, uint32_t N, uint32_t M>
using camera_matrix_t = matrix<T, N, M>;

template<typename T>
using camera_quaternion_t = math::quaternion<T>;

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
            const T scale = std::sqrt(trace + T(1));
            const T invScale = T(0.5) / scale;
            output.data.x = (m[2][1] - m[1][2]) * invScale;
            output.data.y = (m[0][2] - m[2][0]) * invScale;
            output.data.z = (m[1][0] - m[0][1]) * invScale;
            output.data.w = scale * T(0.5);
        }
        else if (m00 >= m11 && m00 >= m22)
        {
            const T scale = std::sqrt(T(1) + m00 - m11 - m22);
            const T invScale = T(0.5) / scale;
            output.data.x = scale * T(0.5);
            output.data.y = (m[0][1] + m[1][0]) * invScale;
            output.data.z = (m[2][0] + m[0][2]) * invScale;
            output.data.w = (m[2][1] - m[1][2]) * invScale;
        }
        else if (m11 >= m22)
        {
            const T scale = std::sqrt(T(1) + m11 - m00 - m22);
            const T invScale = T(0.5) / scale;
            output.data.x = (m[0][1] + m[1][0]) * invScale;
            output.data.y = scale * T(0.5);
            output.data.z = (m[1][2] + m[2][1]) * invScale;
            output.data.w = (m[0][2] - m[2][0]) * invScale;
        }
        else
        {
            const T scale = std::sqrt(T(1) + m22 - m00 - m11);
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

    const T pitch = std::atan2(
        T(2) * (y * z + w * x),
        w * w - x * x - y * y + z * z);
    const T yaw = std::asin(std::clamp(
        T(-2) * (x * z - w * y),
        T(-1),
        T(1)));
    const T roll = std::atan2(
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
    return T(2) * std::acos(orientationDot);
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
inline bool decomposeTransformMatrix(
    const camera_matrix_t<T, 4, 4>& transform,
    camera_vector_t<T, 3>& outTranslation,
    camera_vector_t<T, 3>& outRotationEulerDegrees,
    camera_vector_t<T, 3>& outScale)
{
    outTranslation = camera_vector_t<T, 3>(transform[3].x, transform[3].y, transform[3].z);

    auto right = camera_vector_t<T, 3>(transform[0].x, transform[0].y, transform[0].z);
    auto up = camera_vector_t<T, 3>(transform[1].x, transform[1].y, transform[1].z);
    auto forward = camera_vector_t<T, 3>(transform[2].x, transform[2].y, transform[2].z);

    outScale = camera_vector_t<T, 3>(length(right), length(up), length(forward));

    if (!std::isfinite(outScale.x) || !std::isfinite(outScale.y) || !std::isfinite(outScale.z))
        return false;

    constexpr T Epsilon = std::numeric_limits<T>::epsilon();
    if (outScale.x <= Epsilon || outScale.y <= Epsilon || outScale.z <= Epsilon)
        return false;

    right /= outScale.x;
    up /= outScale.y;
    forward /= outScale.z;

    const auto orientation = makeQuaternionFromBasis(right, up, forward);
    if (!isFiniteQuaternion(orientation))
        return false;

    outRotationEulerDegrees = getQuaternionEulerDegrees(orientation);
    return std::isfinite(outRotationEulerDegrees.x) &&
        std::isfinite(outRotationEulerDegrees.y) &&
        std::isfinite(outRotationEulerDegrees.z);
}

} // namespace nbl::hlsl

#endif // _C_CAMERA_MATH_UTILITIES_HPP_
