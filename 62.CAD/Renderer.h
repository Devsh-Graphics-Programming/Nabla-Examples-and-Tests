#pragma once

#include <nabla.h>

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
		ret.phaseShift = phaseShift;

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

	const float64_t2& getLinePointAt(const uint32_t idx) const
	{
		return m_linePoints[idx];
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
		bool addNewSection = alwaysAddNewSection || !previousSectionIsLine;
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
		m_linePoints.insert(m_linePoints.end(), linePoints.begin(), linePoints.end());
	}

	void addEllipticalArcs(const core::SRange<curves::EllipticalArcInfo>& ellipses)
	{
		// TODO[Erfan] Approximate with quadratic beziers
	}

	// TODO[Przemek]: this input should be nbl::hlsl::QuadraticBezier instead cause `QuadraticBezierInfo` includes precomputed data I don't want user to see
	void addQuadBeziers(const core::SRange<QuadraticBezierInfo>& quadBeziers)
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
		m_quadBeziers.insert(m_quadBeziers.end(), quadBeziers.begin(), quadBeziers.end());
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

protected:
	// TODO[Przemek]: a vector of polyline connetor objects
	std::vector<SectionInfo> m_sections;
	// TODO[Przemek]: instead of float64_t2 for linePoints, store LinePointInfo
	std::vector<float64_t2> m_linePoints;
	std::vector<QuadraticBezierInfo> m_quadBeziers;
};
