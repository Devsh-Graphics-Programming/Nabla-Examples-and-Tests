#pragma once

#include <nabla.h>
#include <nbl/builtin/hlsl/cpp_compat.hlsl>

using namespace nbl;
using namespace ui;

// holds values for `LineStyle` struct and caculates stipple pattern re values, cant think of better name
struct CPULineStyle
{
	static constexpr int32_t InvalidStipplePatternSize = -1;
	static const uint32_t STIPPLE_PATTERN_MAX_SZ = 15u;

	float32_t4 color;
	float screenSpaceLineWidth;
	float worldSpaceLineWidth;
	// gpu stipple pattern data form
	int32_t stipplePatternSize = 0u;
	float reciprocalStipplePatternLen;
	float stipplePattern[STIPPLE_PATTERN_MAX_SZ];
	float phaseShift;
	bool isRoadStyleFlag;

	void setStipplePatternData(const nbl::core::SRange<float>& stipplePatternCPURepresentation)
		//void prepareGPUStipplePatternData(const nbl::core::vector<float>& stipplePatternCPURepresentation)
	{
		assert(stipplePatternCPURepresentation.size() <= STIPPLE_PATTERN_MAX_SZ);

		if (stipplePatternCPURepresentation.size() == 0)
		{
			stipplePatternSize = 0;
			return;
		}

		nbl::core::vector<float> stipplePatternTransformed;

		// just to make sure we have a consistent definition of what's positive and what's negative
		auto isValuePositive = [](float x)
		{
			return (x >= 0);
		};

		// merge redundant values
		for (auto it = stipplePatternCPURepresentation.begin(); it != stipplePatternCPURepresentation.end();)
		{
			float redundantConsecutiveValuesSum = 0.0f;
			const bool firstValueSign = isValuePositive(*it);
			do
			{
				redundantConsecutiveValuesSum += *it;
				it++;
			} while (it != stipplePatternCPURepresentation.end() && (firstValueSign == isValuePositive(*it)));

			stipplePatternTransformed.push_back(redundantConsecutiveValuesSum);
		}

		if (stipplePatternTransformed.size() == 1)
		{
			stipplePatternSize = stipplePatternTransformed[0] < 0.0f ? InvalidStipplePatternSize : 0;
			return;
		}

		// merge first and last value if their sign matches
		phaseShift = 0.0f;
		const bool firstComponentPositive = isValuePositive(stipplePatternTransformed[0]);
		const bool lastComponentPositive = isValuePositive(stipplePatternTransformed[stipplePatternTransformed.size() - 1]);
		if (firstComponentPositive == lastComponentPositive)
		{
			phaseShift += std::abs(stipplePatternTransformed[stipplePatternTransformed.size() - 1]);
			stipplePatternTransformed[0] += stipplePatternTransformed[stipplePatternTransformed.size() - 1];
			stipplePatternTransformed.pop_back();
		}

		if (stipplePatternTransformed.size() == 1)
		{
			stipplePatternSize = (firstComponentPositive) ? 0 : InvalidStipplePatternSize;
			return;
		}

		// rotate values if first value is negative value
		if (!firstComponentPositive)
		{
			std::rotate(stipplePatternTransformed.rbegin(), stipplePatternTransformed.rbegin() + 1, stipplePatternTransformed.rend());
			phaseShift += std::abs(stipplePatternTransformed[0]);
		}

		// calculate normalized prefix sum
		const uint32_t PREFIX_SUM_MAX_SZ = LineStyle::STIPPLE_PATTERN_MAX_SZ - 1u;
		const uint32_t PREFIX_SUM_SZ = stipplePatternTransformed.size() - 1;

		float prefixSum[PREFIX_SUM_MAX_SZ];
		prefixSum[0] = stipplePatternTransformed[0];

		for (uint32_t i = 1u; i < PREFIX_SUM_SZ; i++)
			prefixSum[i] = abs(stipplePatternTransformed[i]) + prefixSum[i - 1];

		reciprocalStipplePatternLen = 1.0f / (prefixSum[PREFIX_SUM_SZ - 1] + abs(stipplePatternTransformed[PREFIX_SUM_SZ]));

		for (int i = 0; i < PREFIX_SUM_SZ; i++)
			prefixSum[i] *= reciprocalStipplePatternLen;

		stipplePatternSize = PREFIX_SUM_SZ;
		std::memcpy(stipplePattern, prefixSum, sizeof(prefixSum));

		phaseShift = phaseShift * reciprocalStipplePatternLen;
		if (stipplePatternTransformed[0] == 0.0)
		{
			phaseShift -= 1e-3; // TODO: I think 1e-3 phase shift in normalized stipple space is a reasonable value? right?
		}
	}

	LineStyle getAsGPUData() const
	{
		LineStyle ret = {};

		// pack into uint32_t
		for (uint32_t i = 0; i < stipplePatternSize; ++i)
		{
			const bool leftIsDot =
				(i > 1 && stipplePattern[i - 1] == stipplePattern[i - 2]) ||
				(i == 1 && stipplePattern[0] == 0.0);

			const bool rightIsDot =
				(i == stipplePatternSize && stipplePattern[0] == 0.0) ||
				(i + 2 <= stipplePatternSize && stipplePattern[i] == stipplePattern[i + 1]);

			ret.stipplePattern[i] = static_cast<uint32_t>(stipplePattern[i] * (1u << 29u));

			if (leftIsDot)
				ret.stipplePattern[i] |= 1u << 30u;
			if (rightIsDot)
				ret.stipplePattern[i] |= 1u << 31u;
		}

		ret.color = color;
		ret.screenSpaceLineWidth = screenSpaceLineWidth;
		ret.worldSpaceLineWidth = worldSpaceLineWidth;
		ret.stipplePatternSize = stipplePatternSize;
		ret.reciprocalStipplePatternLen = reciprocalStipplePatternLen;
		ret.isRoadStyleFlag = isRoadStyleFlag;

		return ret;
	}

	inline bool isVisible() const { return stipplePatternSize != InvalidStipplePatternSize; }
};

static_assert(sizeof(DrawObject) == 16u);
static_assert(sizeof(MainObject) == 8u);
static_assert(sizeof(Globals) == 112u);
static_assert(sizeof(LineStyle) == 96u);
static_assert(sizeof(ClipProjectionData) == 88u);

// It is not optimized because how you feed a Polyline to our cad renderer is your choice. this is just for convenience
// This is a Nabla Polyline used to feed to our CAD renderer. You can convert your Polyline to this class. or just use it directly.
class CPolyline
{
public:

	// each section consists of multiple connected lines or multiple connected ellipses
	struct SectionInfo
	{
		ObjectType	type;
		uint32_t	index; // can't make this a void* cause of vector resize
		uint32_t	count;
	};

	size_t getSectionsCount() const { return m_sections.size(); }

	const SectionInfo& getSectionInfoAt(const uint32_t idx) const
	{
		return m_sections[idx];
	}

	const QuadraticBezierInfo& getQuadBezierInfoAt(const uint32_t idx) const
	{
		return m_quadBeziers[idx];
	}

	const LinePointInfo& getLinePointAt(const uint32_t idx) const
	{
		return m_linePoints[idx];
	}

	const std::vector<PolylineConnector>& getConnectors() const
	{
		return m_polylineConnector;
	}

	void clearEverything()
	{
		m_sections.clear();
		m_linePoints.clear();
		m_quadBeziers.clear();
	}

	// Reserves memory with worst case
	void reserveMemory(uint32_t noOfLines, uint32_t noOfBeziers)
	{
		m_sections.reserve(noOfLines + noOfBeziers);
		m_linePoints.reserve(noOfLines * 2u);
		m_quadBeziers.reserve(noOfBeziers);
	}

	void addLinePoints(const core::SRange<float64_t2>& linePoints, bool addToPreviousLineSectionIfAvailable = false)
	{
		if (linePoints.size() <= 1u)
			return;

		const bool previousSectionIsLine = m_sections.size() > 0u && m_sections[m_sections.size() - 1u].type == ObjectType::LINE;
		const bool alwaysAddNewSection = !addToPreviousLineSectionIfAvailable;
		const bool addNewSection = alwaysAddNewSection || !previousSectionIsLine;
		if (addNewSection)
		{
			SectionInfo newSection = {};
			newSection.type = ObjectType::LINE;
			newSection.index = static_cast<uint32_t>(m_linePoints.size());
			newSection.count = static_cast<uint32_t>(linePoints.size() - 1u);
			m_sections.push_back(newSection);
		}
		else
		{
			m_sections[m_sections.size() - 1u].count += static_cast<uint32_t>(linePoints.size());
		}

		constexpr LinePointInfo EMPTY_LINE_POINT_INFO = {};
		const uint32_t newLinePointSize = m_linePoints.size() + linePoints.size();
		m_quadBeziers.reserve(newLinePointSize);
		for (uint32_t i = 0u; i < linePoints.size(); i++)
		{
			m_linePoints.emplace_back(EMPTY_LINE_POINT_INFO);
			auto& lastLinePoint = m_linePoints[m_linePoints.size() - 1u];
			lastLinePoint.p = linePoints[i];
		}
	}

	void addEllipticalArcs(const core::SRange<curves::EllipticalArcInfo>& ellipses)
	{
		// TODO[Erfan] Approximate with quadratic beziers
	}

	void addQuadBeziers(const core::SRange<shapes::QuadraticBezier<double>>& quadBeziers)
	{
		bool addNewSection = m_sections.size() == 0u || m_sections[m_sections.size() - 1u].type != ObjectType::QUAD_BEZIER;
		if (addNewSection)
		{
			SectionInfo newSection = {};
			newSection.type = ObjectType::QUAD_BEZIER;
			newSection.index = static_cast<uint32_t>(m_quadBeziers.size());
			newSection.count = static_cast<uint32_t>(quadBeziers.size());
			m_sections.push_back(newSection);
		}
		else
		{
			m_sections[m_sections.size() - 1u].count += static_cast<uint32_t>(quadBeziers.size());
		}

		constexpr QuadraticBezierInfo EMPTY_QUADRATIC_BEZIER_INFO = {};
		const uint32_t newQuadBezierSize = m_quadBeziers.size() + quadBeziers.size();
		m_quadBeziers.reserve(newQuadBezierSize);
		for (uint32_t i = 0u; i < quadBeziers.size(); i++)
		{
			m_quadBeziers.emplace_back(EMPTY_QUADRATIC_BEZIER_INFO);
			auto& lastBezier = m_quadBeziers[m_quadBeziers.size() - 1u];
			lastBezier.p[0] = quadBeziers[i].P0;
			lastBezier.p[1] = quadBeziers[i].P1;
			lastBezier.p[2] = quadBeziers[i].P2;
		}
	}

	// TODO[Przemek]: Add a function here named preprocessPolylineWithStyle -> give it the line style
	/*
	*  this preprocess should:
	*	1. if style has road info try to generate miters:
	*		if tangents are not in the same direction with some error add a PolylineConnector object
		2. go over the list of sections (line and beziers in order) compute the phase shift by computing their arclen and divisions with style length and
			fill the phaseShift part of the QuadraticBezierInfo and LinePointInfo,
			you initially set them to 0 in addLinePoints/addQuadBeziers

		NOTE that PolylineConnectors are special object types, user does not add them and they should not be part of m_sections vector
	*/

	void preprocessPolylineWithStyle(const CPULineStyle& lineStyle)
	{
		core::vector<PolylineConnectorHelperInfo> connectorInfos;
		float phaseShiftTotal = lineStyle.phaseShift;
		for (uint32_t sectionIdx = 0u; sectionIdx < m_sections.size(); sectionIdx++)
		{
			const auto& section = m_sections[sectionIdx];

			if (section.type == ObjectType::LINE)
			{
				// calculate phase shift at each point of each line in section
				const uint32_t linePointCnt = section.count + 1u;
				for (uint32_t i = 0u; i < linePointCnt; i++)
				{
					const uint32_t currIdx = section.index + i;
					auto& linePoint = m_linePoints[currIdx];
					if (i == 0u)
					{
						linePoint.phaseShift = phaseShiftTotal;
						continue;
					}

					const auto& prevLinePoint = m_linePoints[section.index + i - 1u];
					const float64_t2 line = linePoint.p - prevLinePoint.p;
					const double lineLen = glm::length(line);

					const double changeInPhaseShiftBetweenCurrAndPrevPoint = std::remainder(lineLen, 1.0f / lineStyle.reciprocalStipplePatternLen) * lineStyle.reciprocalStipplePatternLen;
					linePoint.phaseShift = glm::fract(phaseShiftTotal + changeInPhaseShiftBetweenCurrAndPrevPoint);
					phaseShiftTotal = linePoint.phaseShift;

					if (lineStyle.isRoadStyleFlag)
					{
						PolylineConnectorHelperInfo connectorInfo;
						connectorInfo.worldSpaceCircleCenter = prevLinePoint.p;
						connectorInfo.normal = float32_t2(-line.y, line.x) / static_cast<float>(lineLen);
						connectorInfo.phaseShift = linePoint.phaseShift;

						connectorInfos.push_back(connectorInfo);
					}
				}
			}
			else if (section.type == ObjectType::QUAD_BEZIER)
			{
				// calculate phase shift at point P0 of each bezier
				const uint32_t quadBezierCnt = section.count;
				const bool isLastSection = sectionIdx == m_sections.size() - 1u;

				for (uint32_t i = 0u; i <= quadBezierCnt; i++)
				{
					// there is no need to calculate anything for the last bezier in the last section
					if (isLastSection && i == quadBezierCnt)
					{
						break;
					}

					const uint32_t currIdx = section.index + i;
					if (i == 0u)
					{
						QuadraticBezierInfo& firstInSectionQuadBezierInfo = m_quadBeziers[currIdx];
						firstInSectionQuadBezierInfo.phaseShift = phaseShiftTotal;
						continue;
					}

					const QuadraticBezierInfo& prevQuadBezierInfo = m_quadBeziers[currIdx - 1u];
					shapes::QuadraticBezier<double> quadraticBezier = shapes::QuadraticBezier<double>::construct(prevQuadBezierInfo.p[0], prevQuadBezierInfo.p[1], prevQuadBezierInfo.p[2]);
					shapes::Quadratic<double> quadratic = shapes::Quadratic<double>::constructFromBezier(quadraticBezier);
					shapes::Quadratic<double>::ArcLengthCalculator arcLenCalc = shapes::Quadratic<double>::ArcLengthCalculator::construct(quadratic);
					const double bezierLen = arcLenCalc.calcArcLen(1.0f);

					const double nextLineInSectionLocalPhaseShift = std::remainder(bezierLen, 1.0f / lineStyle.reciprocalStipplePatternLen) * lineStyle.reciprocalStipplePatternLen;
					phaseShiftTotal = glm::fract(phaseShiftTotal + nextLineInSectionLocalPhaseShift);

					if (i < quadBezierCnt)
					{
						QuadraticBezierInfo& quadBezierInfo = m_quadBeziers[currIdx];
						quadBezierInfo.phaseShift = phaseShiftTotal;
					}

					// TODO: bezier normal
					//if (lineStyle.isRoadStyleFlag)
					//{

					//}
				}
			}

			// generate miters
			if (lineStyle.isRoadStyleFlag)
			{
				for (uint32_t i = 1u; i < connectorInfos.size(); i++)
				{
					// TODO[Przemek]: this is tailored for specific case and not correct, every line has 2 normals at P0, and there should be other way to find the correct ones
					const float32_t2 prevLineNormal = connectorInfos[i - 1].normal;
					const float32_t2 nextLineNormal = connectorInfos[i].normal;

					// This basicly tells me if theta < alpha
					const float crossProductZ = nextLineNormal.x * prevLineNormal.y - prevLineNormal.x * nextLineNormal.y;

					if (std::abs(crossProductZ) < 0.000001f)
						continue;

					const float64_t2 intersectionDirection = glm::normalize(prevLineNormal + nextLineNormal);
					const float64_t cosAngleBetweenNormals = glm::dot(prevLineNormal, nextLineNormal);

					PolylineConnector res{};
					res.circleCenter = connectorInfos[i].worldSpaceCircleCenter;
					res.v = static_cast<float32_t2>(intersectionDirection * std::sqrt(2.0 / (1.0 + cosAngleBetweenNormals)));
					//res.cosAngleDifferenceHalf = std::cos(std::acos(cosAngleBetweenNormals) * 0.5f);
					res.cosAngleDifferenceHalf = static_cast<float32_t>(std::sqrt((1.0 + cosAngleBetweenNormals) * 0.5));
					res.phaseShift = connectorInfos[i].phaseShift;

					if (crossProductZ < 0.0f)
						res.v = -res.v;

					m_polylineConnector.push_back(res);
				}
			}
		}
	}

protected:
	// TODO[Przemek]: a vector of polyline connetor objects
	std::vector<PolylineConnector> m_polylineConnector;
	std::vector<SectionInfo> m_sections;
	std::vector<LinePointInfo> m_linePoints;
	std::vector<QuadraticBezierInfo> m_quadBeziers;

private:
	struct PolylineConnectorHelperInfo
	{
		float32_t2 worldSpaceCircleCenter;
		float32_t2 normal;
		float phaseShift;
	};

};
