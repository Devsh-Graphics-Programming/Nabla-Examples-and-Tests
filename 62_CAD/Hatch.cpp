
#include "Hatch.h"

#include <complex.h>
#include <tgmath.h>
#include <nbl/builtin/hlsl/shapes/util.hlsl>

// #define DEBUG_HATCH_VISUALLY

using namespace nbl;

bool Hatch::Segment::isStraightLineConstantMajor() const
{
	auto major = (uint32_t)SelectedMajorAxis;
	const double p0 = originalBezier->P0[major], 
		p1 = originalBezier->P1[major], 
		p2 = originalBezier->P2[major];
	//assert(p0 <= p1 && p1 <= p2); (PRECISION ISSUES ARISE ONCE MORE)
	return abs(p1 - p0) <= core::exp2(-24.0) && abs(p2 - p0) <= exp(-24);
}

std::array<double, 2> Hatch::Segment::intersect(const Segment& other) const
{
	auto major = (int)SelectedMajorAxis;
	auto minor = 1-major;

	std::array<double, 2> result = { core::nan<double>(), core::nan<double>() };
	int resultIdx = 0;

	// Use line intersections if one or both of the beziers are linear (a = 0)
	const bool selfLinear = isLineSegment(*originalBezier);
	const bool otherLinear = isLineSegment(*other.originalBezier);
	if (selfLinear && otherLinear)
	{
		// Line/line intersection
		//TODO: use cpp-compat hlsl builtin
		auto intersectionPoint =  nbl::hlsl::shapes::util::LineLineIntersection<float64_t>(
			originalBezier->P0, originalBezier->P2 - originalBezier->P0,
			other.originalBezier->P0, other.originalBezier->P2 - other.originalBezier->P0
		);
		const double x1 = originalBezier->P0.x, y1 = originalBezier->P0.y,
			x2 = originalBezier->P2.x, y2 = originalBezier->P2.y,
			x3 = other.originalBezier->P0.x, y3 = other.originalBezier->P0.y,
			x4 = other.originalBezier->P2.x, y4 = other.originalBezier->P2.y;

		// Return if point is on the lines
		if (std::min(x1, x2) <= intersectionPoint.x && y1 <= intersectionPoint.y && std::max(x1, x2) >= intersectionPoint.x && y2 >= intersectionPoint.y &&
			std::min(x3, x4) <= intersectionPoint.x && y3 <= intersectionPoint.y && std::max(x3, x4) >= intersectionPoint.x && y4 >= intersectionPoint.y)
		{
			// Gets t for "other" by using intersectOrtho
			auto otherT = intersectOrtho(*other.originalBezier, intersectionPoint.y, major);
			auto intersectionMajor = other.originalBezier->evaluate(otherT)[major];
			auto thisT = intersectOrtho(*originalBezier, intersectionMajor, major);

			if (otherT >= other.t_start && otherT <= other.t_end && thisT >= t_start && thisT <= t_end)
			{
				result[0] = otherT;
			}
		}
	}
	else if (selfLinear || otherLinear)
	{
		// Line/curve intersection
		const auto& line = selfLinear ? *originalBezier : *other.originalBezier;
		const auto& curve = selfLinear ? *other.originalBezier : *originalBezier;

		float64_t2  D = normalize(line.P2 - line.P0);
		float64_t2x2 rotation = { 
			{D.x, D.y}, 
			{-D.y, D.x} 
		};
		QuadraticBezier rotatedCurve = {
			mul(rotation, curve.P0 - line.P0),
			mul(rotation, curve.P1 - line.P0),
			mul(rotation, curve.P2 - line.P0)
		};

		auto intersectionCurveT = intersectOrtho(rotatedCurve, 0, (int)MajorAxis::MAJOR_Y /* Always in rotation to align with X Axis */);
		auto intersectionMajor = other.originalBezier->evaluate(intersectionCurveT)[major];
		auto intersectionLineT = intersectOrtho(line, intersectionMajor, major);

		auto thisT = selfLinear ? intersectionLineT : intersectionCurveT;
		auto otherT = selfLinear ? intersectionCurveT : intersectionLineT;

		if (otherT >= other.t_start && otherT <= other.t_end && thisT >= t_start && thisT <= t_end)
		{
			result[0] = otherT;
		}
	}
	else
	{
		auto thisBezier = *originalBezier;
		thisBezier.splitFromMinToMax(t_start, t_end); // to get correct P0, P1, P2 for intersection testing

		const auto p0 = thisBezier.P0;
		const auto p1 = thisBezier.P1;
		const auto p2 = thisBezier.P2;
		const bool sideP1 = nbl::hlsl::cross2D(p2 - p0, p1 - p0) >= 0.0;
		
		const auto& otherBezier = *other.originalBezier;
		const std::array<double, 4> intersections = bezierBezierIntersections(otherBezier, thisBezier);

		for (uint32_t i = 0; i < intersections.size(); i++)
		{
			auto t = intersections[i];
			if (core::isnan(t) || other.t_start >= t || t >= other.t_end)
				continue;

			auto intersection = otherBezier.evaluate(t);
			
			// Optimization istead of doing SDF to find other T and check against bounds:
			// If both P1 and the intersection point are on the same side of the P0 -> P2 line of thisBezier, it's a a valid intersection
			const bool sideIntersection = nbl::hlsl::cross2D(p2 - p0, intersection - p0) >= 0.0;
			if (sideP1 != sideIntersection)
				continue;

			const bool duplicateT = (resultIdx > 0 && t == result[0]) || (resultIdx > 1 && t == result[1]);
			if (!duplicateT)
			{
				if (resultIdx < 2)
				{
					result[resultIdx] = t;
					resultIdx++;
				}
				else
				{
					_NBL_DEBUG_BREAK_IF(true); // more intersections that expected
				}
			}
		}
	}

	return result;
}

Hatch::Hatch(std::span<CPolyline> lines, const MajorAxis majorAxis, nbl::system::logger_opt_smart_ptr logger, int32_t* debugStepPtr, const std::function<void(CPolyline, LineStyleInfo)>& debugOutput)
{
	// this threshsold is used to decide when to consider minor position to be 
	// the same and check tangents because intersection algorithms has rounding 
	// errors
	constexpr float64_t MinorPositionComparisonThreshhold = 1e-3;
	constexpr float64_t TangentComparisonThreshhold = 1e-7;

	std::vector<QuadraticBezier> beziers; // Referenced into by the segments
	std::stack<Segment> starts; // Next segments sorted by start points
	std::stack<double> ends; // Next end points
	std::priority_queue<double, std::vector<double>, std::greater<double> > intersections; // Next intersection points as major coordinate
	double maxMajor;

	int major = (int)majorAxis;
	int minor = 1-major; // Minor = Opposite of major (X)
	
#ifdef DEBUG_HATCH_VISUALLY
	int32_t debugStepDefault = 0u;
	int32_t& debugStep = (debugStepPtr) ? *debugStepPtr : debugStepDefault;

	auto drawDebugBezier = [&](QuadraticBezier bezier, float32_t4 color)
	{
		CPolyline outputPolyline;
		std::vector<shapes::QuadraticBezier<double>> beziers;
		shapes::QuadraticBezier<double> tmpBezier;
		tmpBezier.P0 = bezier.P0;
		tmpBezier.P1 = bezier.P1;
		tmpBezier.P2 = bezier.P2;
		beziers.push_back(tmpBezier);
		outputPolyline.addQuadBeziers(beziers);

		LineStyleInfo lineStyleInfo;
		lineStyleInfo.screenSpaceLineWidth = 4.0f;
		lineStyleInfo.worldSpaceLineWidth = 0.0f;
		lineStyleInfo.color = color;

		debugOutput(outputPolyline, lineStyleInfo);
	};

	auto drawDebugLine = [&](float64_t2 start, float64_t2 end, float32_t4 color)
	{
		CPolyline outputPolyline;
		std::vector<float64_t2> points;
		points.push_back(start);
		points.push_back(end);
		outputPolyline.addLinePoints(points);
		
		LineStyleInfo lineStyleInfo;
		lineStyleInfo.screenSpaceLineWidth = 2.0f;
		lineStyleInfo.worldSpaceLineWidth = 0.0f;
		lineStyleInfo.color = color;
		
		debugOutput(outputPolyline, lineStyleInfo);
	};
#endif

	{
		std::vector<Segment> segments;
		for (CPolyline& polyline : lines)
		{
			for (uint32_t secIdx = 0; secIdx < polyline.getSectionsCount(); secIdx ++)
			{
				auto addMonotonicBezier = [&](QuadraticBezier bezier)
				{
					auto outputBezier = bezier;
					if (outputBezier.P0[major] > outputBezier.P2[major])
					{
						outputBezier.P2 = bezier.P0;
						outputBezier.P0 = bezier.P2;
						assert(outputBezier.P0[major] <= outputBezier.P2[major]);
					}
					// fix in case of small precision issues when splitting into major monotonic segments
					if (outputBezier.P1.y < outputBezier.P0.y)
						outputBezier.P1.y = outputBezier.P0.y;

#ifdef DEBUG_HATCH_VISUALLY
					if (debugOutput)
					{
						uint32_t bezierIdx = beziers.size();
						float32_t4 colors[5] = {
							float32_t4(33,150,243, 255) / float32_t4(255.0),
							float32_t4(29,233,182, 255) / float32_t4(255.0),
							float32_t4(238,255,65, 255) / float32_t4(255.0),
							float32_t4(244,81,30, 255) / float32_t4(255.0),
							float32_t4(211,47,47, 255) / float32_t4(255.0)
						};
						//drawDebugBezier(bezier, colors[bezierIdx % 5]);
					}
#endif
					beziers.push_back(outputBezier);
				};

				auto section = polyline.getSectionInfoAt(secIdx);
				if (section.type == ObjectType::LINE)
					{
						for (uint32_t itemIdx = section.index; itemIdx < section.index + section.count; itemIdx++)
						{
							auto begin = polyline.getLinePointAt(itemIdx).p;
							auto end = polyline.getLinePointAt(itemIdx + 1).p;
							addMonotonicBezier(QuadraticBezier::construct(begin, (begin + end) * 0.5, end));
#ifdef DEBUG_HATCH_VISUALLY
							//if (debugOutput)
							//    drawDebugBezier(QuadraticBezier::construct(begin, (begin + end) * 0.5, end), float32_t4(0.0, 0.0, 1.0, 1.0));
#endif
					}
				}
				else if (section.type == ObjectType::QUAD_BEZIER)
				{
					for (uint32_t itemIdx = section.index; itemIdx < section.index + section.count; itemIdx ++)
					{
						auto bezierInfo = polyline.getQuadBezierInfoAt(itemIdx);
						auto unsplitBezier = bezierInfo.shape;
						
						// Beziers must be monotonically increasing along major
						// First step: Make sure the bezier is monotonic, split it if not
						std::array<QuadraticBezier, 2> monotonicSegments;
						auto isMonotonic = splitIntoMajorMonotonicSegments(unsplitBezier, monotonicSegments);

						if (isMonotonic)
						{
							// Already was monotonic
							addMonotonicBezier(unsplitBezier);
#ifdef DEBUG_HATCH_VISUALLY
							//if (debugOutput)
								//drawDebugBezier(unsplitBezier, float32_t4(0.8, 0.8, 0.8, 1.0));
#endif
						}
						else
						{
							addMonotonicBezier(monotonicSegments.data()[0]);
							addMonotonicBezier(monotonicSegments.data()[1]);
#ifdef DEBUG_HATCH_VISUALLY
							//if (debugOutput)
							//{ drawDebugBezier(monotonicSegments.data()[0], float32_t4(1.0, 0.0, 0.0, 1.0)); drawDebugBezier(monotonicSegments.data()[1], float32_t4(0.0, 1.0, 0.0, 1.0)); }
#endif
						}
					}
				}
			}
		}

		for (uint32_t bezierIdx = 0; bezierIdx < beziers.size(); bezierIdx++)
		{
			auto hatchBezier = &beziers[bezierIdx];
			Segment segment;
			segment.originalBezier = hatchBezier;
			segment.t_start = 0.0;
			segment.t_end = 1.0;
			segments.push_back(segment);
		}
		
		if (segments.empty())
		{
			logger.log("Empty Polylines with no segments were fed into the Hatch construction.", nbl::system::ILogger::ELL_WARNING);
			return;
		}

		std::sort(segments.begin(), segments.end(), [&](const Segment& a, const Segment& b) { return a.originalBezier->P0[major] > b.originalBezier->P0[major]; });
		for (Segment& segment : segments)
			starts.push(segment);

		std::sort(segments.begin(), segments.end(), [&](const Segment& a, const Segment& b) { return a.originalBezier->P2[major] > b.originalBezier->P2[major]; });
		for (Segment& segment : segments)
			ends.push(segment.originalBezier->P2[major]);
		maxMajor = segments.front().originalBezier->P2[major];
	}

#ifdef DEBUG_HATCH_VISUALLY
	int32_t step = 0;
#endif

	// Sweep line algorithm
	std::vector<Segment> activeCandidates; // Set of active candidates for neighbor search in sweep line

	// if we weren't spawning quads, we could just have unsorted `vector<Bezier*>`
	auto candidateComparator = [&](const Segment& lhs, const Segment& rhs)
	{
		// btw you probably want the beziers in Quadratic At^2+B+C form, not control points
		double _lhs = lhs.originalBezier->evaluate(lhs.t_start)[minor];
		double _rhs = rhs.originalBezier->evaluate(rhs.t_start)[minor];

		double lenLhs = glm::distance(lhs.originalBezier->P0, lhs.originalBezier->P2);
		double lenRhs = glm::distance(rhs.originalBezier->P0, rhs.originalBezier->P2);
		auto minLen = std::min(lenLhs, lenRhs);
#ifdef DEBUG_HATCH_VISUALLY
		if (debugOutput && step == debugStep)
		{
			//printf(std::format("comparison: lhs = ({}, {}), ({}, {}), ({}, {}) rhs = ({}, {}), ({}, {}), ({}, {})",
			//	lhs.originalBezier->P0.x, lhs.originalBezier->P0.y, 
			//	lhs.originalBezier->P1.x, lhs.originalBezier->P1.y, 
			//	lhs.originalBezier->P2.x, lhs.originalBezier->P2.y, 

			//	rhs.originalBezier->P0.x, rhs.originalBezier->P0.y, 
			//	rhs.originalBezier->P1.x, rhs.originalBezier->P1.y, 
			//	rhs.originalBezier->P2.x, rhs.originalBezier->P2.y
			//	).c_str());
			//drawDebugLine(float64_t2(_lhs, -1000.0), float64_t2(_lhs, 1000.0), float64_t4(0.1, 0.1, 1.0, 1.0));
			//drawDebugLine(float64_t2(_rhs, -1000.0), float64_t2(_rhs, 1000.0), float64_t4(0.1, 1.0, 1.0, 1.0));
			//printf(std::format("(comparing minor) _lhs: {} (len: {}) _rhs: {} (len: {}) minLen: {} diff: {} ",
			//	_lhs, lenLhs, _rhs, lenRhs, minLen, abs(_lhs - _rhs)).c_str());
		}
#endif

		// Threshhold here for intersection points, where the minor values for the curves are
		// very close but could be smaller, causing the curves to be in the wrong order
		if (abs(_lhs - _rhs) < MinorPositionComparisonThreshhold * minLen)
		{
			// this is how you want to order the derivatives dmin/dmaj=-INF dmin/dmaj = 0 dmin/dmaj=INF
			// also leverage the guarantee that `dmaj>=0` to ger numerically stable compare
			// also leverage the guarantee that `dmaj>=0` to ger numerically stable compare
			auto lhsQuadratic = QuadraticCurve::constructFromBezier(*lhs.originalBezier);
			auto rhsQuadratic = QuadraticCurve::constructFromBezier(*rhs.originalBezier);

			float64_t2 lTan = lhs.originalBezier->derivative(lhs.t_start);
			float64_t2 rTan = rhs.originalBezier->derivative(rhs.t_start);
			_lhs = lTan[minor] * rTan[major];
			_rhs = rTan[minor] * lTan[major];
#ifdef DEBUG_HATCH_VISUALLY
			if (false) //(debugOutput && step == debugStep)
			{
				printf(std::format("(comparing tangent) lTan: {}, {} rTan: {}, {} _lhs: {} _rhs: {} abs(_lhs - _rhs): {} abs(_lhs - 0.0): {} ",
					lTan.x, lTan.y, rTan.x, rTan.y, _lhs, _rhs, 
					abs(_lhs - _rhs), abs(_lhs - 0.0)).c_str());
			}
#endif
			// negative values mess with the comparison operator when using multiplication
			// they should be positive because of major monotonicity
			assert(lTan[major] >= 0.0);
			assert(rTan[major] >= 0.0);

			if (abs(_lhs - _rhs) < TangentComparisonThreshhold)
			{
				float64_t2 lAcc = 2.0 * lhsQuadratic.A;
				float64_t2 rAcc = 2.0 * rhsQuadratic.A;

				// In this branch, _lhs == _rhs == 0 (tangents are both 0)
				if (abs(_lhs - 0.0) < TangentComparisonThreshhold)
				{
					// TODO https://discord.com/channels/593902898015109131/723305695046533151/1169377896658383008
					bool lTanSign = lTan[minor] >= 0.0;
					bool rTanSign = rTan[minor] >= 0.0;
					// CASE A
					// 
					// If the signs of the horizontal tangents differ, we know thje negative 
					// one belongs to the left curve
					if (lTanSign != rTanSign)
					{
#ifdef DEBUG_HATCH_VISUALLY
						if (false) //(debugOutput && step == debugStep)
						{
							printf(std::format("(comparing sign) lTanSign: {} rTanSign: {} ",
								lTanSign ? "positive" : "negative", rTanSign ? "positive" : "negative").c_str());
							printf("\n");
						}
#endif
						// We want to return true if lhs < rhs (lhs is to the left of rhs)
						// For this to be false, rhs would need to be the left one (and therefore negative)
						return rTanSign;
					}

					// Otherwise (CASE B)
					// 
					// In this case tangents are both on the same side
					// so only the magnitude / abs of the d2Major / dMinor2 is important
					_lhs = -abs(lAcc[minor] * lTan[major] - lAcc[major] * lTan[minor]) * pow(rTan[minor], 3.0);
					_rhs = -abs(rAcc[minor] * rTan[major] - rAcc[major] * rTan[minor]) * pow(lTan[minor], 3.0);
#ifdef DEBUG_HATCH_VISUALLY
					if (debugOutput && step == debugStep)
					{
						printf(std::format("(comparing second derivatives) A1={},{} B1={},{} A2={},{} B2={},{} _lhs={} _rhs={}",
							lhsQuadratic.A.x, lhsQuadratic.A.y, lhsQuadratic.B.x, lhsQuadratic.B.y,
							rhsQuadratic.A.x, rhsQuadratic.A.y, rhsQuadratic.B.x, rhsQuadratic.B.y,
							_lhs, _rhs
						).c_str());
					}
#endif
				}
				else
				{
					_lhs = (lAcc[minor] * lTan[major] - lAcc[major] * lTan[minor]) * pow(rTan[major], 3.0);
					_rhs = (rAcc[minor] * rTan[major] - rAcc[major] * rTan[minor]) * pow(lTan[major], 3.0);
				}
			}
		}
#ifdef DEBUG_HATCH_VISUALLY
		if (debugOutput && step == debugStep)
		{
			printf("\n");
		}
#endif
		return _lhs < _rhs;
	};
	auto addToCandidateSet = [&](const Segment& entry)
	{
		if (entry.isStraightLineConstantMajor())
			return;
		// Look for intersections among active candidates
		// this is a little O(n^2) but only in the `n=candidates.size()`
		for (const auto& segment : activeCandidates)
		{
			// find intersections entry vs segment
			auto intersectionPoints = entry.intersect(segment);
#ifdef DEBUG_HATCH_VISUALLY
			if (debugOutput && step == debugStep)
			{
				for (uint32_t i = 0; i < intersectionPoints.size(); i++)
				{
					if (nbl::core::isnan(intersectionPoints[i]))
						continue;
					auto point = segment.originalBezier->evaluate(intersectionPoints[i]);
					auto min = point - 0.3;
					auto max = point + 0.3;
					drawDebugLine(float64_t2(min.x, min.y), float64_t2(max.x, min.y), float32_t4(0.0, 0.3, 0.0, 0.8));
					drawDebugLine(float64_t2(max.x, min.y), float64_t2(max.x, max.y), float32_t4(0.0, 0.3, 0.0, 0.8));
					drawDebugLine(float64_t2(min.x, max.y), float64_t2(max.x, max.y), float32_t4(0.0, 0.3, 0.0, 0.8));
					drawDebugLine(float64_t2(min.x, min.y), float64_t2(min.x, max.y), float32_t4(0.0, 0.3, 0.0, 0.8));
				}
			}
#endif

			for (uint32_t i = 0; i < intersectionPoints.size(); i++)
			{
				if (nbl::core::isnan(intersectionPoints[i]))
					continue;
				intersections.push(segment.originalBezier->evaluate(intersectionPoints[i])[major]);
			}
		}
		activeCandidates.push_back(entry);
	};

	double lastMajor = starts.top().originalBezier->evaluate(starts.top().t_start)[major];
	while (lastMajor!=maxMajor)
	{
#ifdef DEBUG_HATCH_VISUALLY
		if (debugOutput && step > debugStep)
			break;
		bool isCurrentDebugStep = step == debugStep;
#endif

		double newMajor;
		bool addStartSegmentToCandidates = false;

		if (ends.empty())
		{
			logger.log("Hatch Creation Failure: `ends` stack is empty in the main loop", nbl::system::ILogger::ELL_ERROR);
			_NBL_DEBUG_BREAK_IF(true); // This shouldn't happen, TODO: LOG
			break;
		}
		const double maxMajorEnds = ends.top();

		const Segment nextStartEvent = starts.empty() ? Segment() : starts.top();
		const double minMajorStart = nextStartEvent.originalBezier ? nextStartEvent.originalBezier->evaluate(nextStartEvent.t_start)[major] : 0.0;

		// We check which event, within start, end and intersection events have the smallest
		// major coordinate at this point
		auto intersectionVisit = [&]()
		{
			const double newMajor = intersections.top();
#ifdef DEBUG_HATCH_VISUALLY
			if (debugOutput && isCurrentDebugStep)
				drawDebugLine(float64_t2(-1000.0, newMajor), float64_t2(1000.0, newMajor), float32_t4(0.0, 0.0, 0.8, 1.0));
#endif
			intersections.pop(); // O(n)
			return newMajor;
		};

		// next start event is before next end event
		if (nextStartEvent.originalBezier && minMajorStart < maxMajorEnds)
		{
			// next start event is before next intersection event
			// (start event)
			if (intersections.empty() || minMajorStart < intersections.top()) // priority queue top() is O(1)
			{
				starts.pop();
				newMajor = minMajorStart;
				addStartSegmentToCandidates = true;
#ifdef DEBUG_HATCH_VISUALLY
				if (debugOutput && isCurrentDebugStep)
					drawDebugLine(float64_t2(-1000.0, newMajor), float64_t2(1000.0, newMajor), float32_t4(0.0, 0.8, 0.0, 1.0));
#endif
			}
			// (intersection event)
			else newMajor = intersectionVisit();
		}
		// next intersection event is before next end event
		// (intersection event)
		else if (!intersections.empty() && intersections.top() < maxMajorEnds)
			newMajor = intersectionVisit();
		else
		{
			// (end event)
			newMajor = maxMajorEnds;
			ends.pop();
#ifdef DEBUG_HATCH_VISUALLY
			if (debugOutput && isCurrentDebugStep)
				drawDebugLine(float64_t2(-1000.0, newMajor), float64_t2(1000.0, newMajor), float32_t4(0.0, 0.0, 0.8, 1.0));
#endif
			//std::cout << "End event at " << newMajor << "\n";
		}
		// spawn quads for the previous iterations if we advanced

		if (newMajor > lastMajor) 
		{
			const auto candidatesSize = std::distance(activeCandidates.begin(),activeCandidates.end());
			// Because n4ce works on loops, this must be `true` in almost every case, but can fail at times, because we skip adding beziers (lines) almost constant in major direction
			if (candidatesSize % 2u == 0u)
			{
#ifdef DEBUG_HATCH_VISUALLY
				if (debugOutput && isCurrentDebugStep)
					drawDebugLine(float64_t2(-1000.0, lastMajor), float64_t2(1000.0, lastMajor), float32_t4(0.1, 0.1, 0.0, 0.5));
#endif
				// trim
				if ((candidatesSize % 2u) != 0u)
				{
					logger.log("Hatch Creation Failure: candidatesSize is odd", nbl::system::ILogger::ELL_ERROR);
					_NBL_DEBUG_BREAK_IF(true); // input polyline/polygon 
				}
#ifdef DEBUG_HATCH_VISUALLY
				if (candidatesSize % 2u == 1u)
				{
					for (uint32_t i = 0u; i < candidatesSize; i++)
					{
						const Segment& item = activeCandidates[i];
						auto curveMinEnd = intersectOrtho(*item.originalBezier, newMajor, major);
						auto splitCurveMin = *item.originalBezier;
						splitCurveMin.splitCurveFromMinToMax(item.t_start, core::isnan(curveMinEnd) ? 1.0 : curveMinEnd);

						drawDebugBezier(splitCurveMin, (i == candidatesSize - 1) ? float32_t4(0.0, 0.0, 1.0, 1.0) : float32_t4(1.0, 0.0, 0.0, 1.0));
						if (i == candidatesSize - 1)
						{
							printf(std::format("problematic guy: ({}, {}), ({}, {}), ({}, {})",
								splitCurveMin.P0.x, splitCurveMin.P0.y,
								splitCurveMin.P1.x, splitCurveMin.P1.y,
								splitCurveMin.P2.x, splitCurveMin.P2.y
							).c_str());
						}
					}
				}
#endif
				for (auto i = 0u; i < (candidatesSize / 2) * 2;)
				{
					const Segment& left = activeCandidates[i++];
					const Segment& right = activeCandidates[i++];

					CurveHatchBox curveBox;

					// Due to precision, if the curve is right at the end, intersectOrtho may return nan
					auto curveMinEnd = intersectOrtho(*left.originalBezier, newMajor, major);
					auto curveMaxEnd = intersectOrtho(*right.originalBezier, newMajor, major);

					auto splitCurveMin = *left.originalBezier;
					splitCurveMin.splitFromMinToMax(left.t_start, core::isnan(curveMinEnd) ? 1.0 : curveMinEnd);
					auto splitCurveMax = *right.originalBezier;
					splitCurveMax.splitFromMinToMax(right.t_start, core::isnan(curveMaxEnd) ? 1.0 : curveMaxEnd);

					assert(splitCurveMin.evaluate(0.0)[major] <= splitCurveMin.evaluate(1.0)[major]);
					assert(splitCurveMax.evaluate(0.0)[major] <= splitCurveMax.evaluate(1.0)[major]);

					auto curveMinAabb = getBezierBoundingBoxMinor(splitCurveMin);
					auto curveMaxAabb = getBezierBoundingBoxMinor(splitCurveMax);
					curveBox.aabbMin = float64_t2(std::min(curveMinAabb.first.x, curveMaxAabb.first.x), lastMajor);
					curveBox.aabbMax = float64_t2(std::max(curveMinAabb.second.x, curveMaxAabb.second.x), newMajor);

#ifdef DEBUG_HATCH_VISUALLY
					if (isCurrentDebugStep)
					{
						drawDebugBezier(splitCurveMin, float64_t4(1.0, 0.0, 0.0, 1.0));
						drawDebugBezier(splitCurveMax, float64_t4(0.0, 1.0, 0.0, 1.0));

						printf(std::format("AABB min: {}, {} max: {}, {} curve min: ({}, {}), ({}, {}), ({}, {}) curve max ({}, {}), ({}, {}), ({}, {})\n",
							curveBox.aabbMin.x, curveBox.aabbMin.y, curveBox.aabbMax.x, curveBox.aabbMax.y,

							splitCurveMin.P0.x, splitCurveMin.P0.y,
							splitCurveMin.P1.x, splitCurveMin.P1.y,
							splitCurveMin.P2.x, splitCurveMin.P2.y,
							splitCurveMax.P0.x, splitCurveMax.P0.y,
							splitCurveMax.P1.x, splitCurveMax.P1.y,
							splitCurveMax.P2.x, splitCurveMax.P2.y
						).c_str());
					}
#endif

					// Transform curves into AABB UV space and turn them into quadratic coefficients
					// so we wont need to convert here
					auto transformCurves = [](Hatch::QuadraticBezier bezier, float64_t2 aabbMin, float64_t2 aabbMax, float32_t2* output) {
						auto rcpAabbExtents = float64_t2(1.0, 1.0) / (aabbMax - aabbMin);
						auto transformedBezier = QuadraticBezier::construct(
							(bezier.P0 - aabbMin) * rcpAabbExtents,
							(bezier.P1 - aabbMin) * rcpAabbExtents,
							(bezier.P2 - aabbMin) * rcpAabbExtents
						);
						auto quadratic = QuadraticCurve::constructFromBezier(transformedBezier);

						if (isLineSegment(transformedBezier))
							quadratic.A = float64_t2(0.0);

						output[0] = (quadratic.A);
						output[1] = (quadratic.B);
						output[2] = (quadratic.C);
						};
					transformCurves(splitCurveMin, curveBox.aabbMin, curveBox.aabbMax, &curveBox.curveMin[0]);
					transformCurves(splitCurveMax, curveBox.aabbMin, curveBox.aabbMax, &curveBox.curveMax[0]);

					hatchBoxes.push_back(curveBox);
				}
			}

			// advance and trim all of the beziers in the candidate set
			auto oit = activeCandidates.begin();
			for (auto iit = activeCandidates.begin(); iit != activeCandidates.end(); iit++)
			{
				const double evalAtMajor = iit->originalBezier->evaluate(iit->t_end)[major];

				auto origBez = iit->originalBezier;
				// if we scrolled past the end of the segment, remove it
				// (basically, we memcpy everything after something is different
				// and we skip on the memcpy for any items that are also different)
				// (this is supposedly a pattern with input/output operators)
				if (newMajor < evalAtMajor)
				{
					const double new_t_start = intersectOrtho(*iit->originalBezier, newMajor, major);

					// little optimization (don't memcpy anything before something was removed)
					if (oit != iit)
						*oit = *iit;
					oit->t_start = new_t_start;
					oit++;
				}
			}
			// trim
			const auto newSize = std::distance(activeCandidates.begin(), oit);
			activeCandidates.resize(newSize);
		}

		// If we had a start event, we need to add the candidate
		if (addStartSegmentToCandidates)
		{
			addToCandidateSet(nextStartEvent);
		}
		
		// We'll need to sort if we had a start event and added to the candidate set
		// or if we have advanced our candidate set
		if (addStartSegmentToCandidates || newMajor > lastMajor)
		{
			std::sort(activeCandidates.begin(), activeCandidates.end(), candidateComparator);
		}

		if (newMajor > lastMajor)
			lastMajor = newMajor;

#ifdef DEBUG_HATCH_VISUALLY
		step++;
#endif
	}
#ifdef DEBUG_HATCH_VISUALLY
	debugStep = debugStep - step;
#endif
}

// returns two possible values of t in the lhs curve where the curves intersect
std::array<double, 4> Hatch::bezierBezierIntersections(const QuadraticBezier& lhs, const QuadraticBezier& rhs)
{
	const auto quarticEquation = nbl::hlsl::shapes::getBezierBezierIntersectionEquation<float64_t>(lhs, rhs);
	
	using nbl::hlsl::math::equations::Quartic;
	using nbl::hlsl::math::equations::Cubic;
	using nbl::hlsl::math::equations::Quadratic;
	constexpr double QUARTIC_THRESHHOLD = 1e-10;
	
	std::array<double, 4> t = { core::nan<double>(), core::nan<double>(), core::nan<double>(), core::nan<double>() }; // only two candidates in range, ever
	
	const double quadCoeffMag = std::max(std::abs(quarticEquation.d), std::abs(quarticEquation.e));
	const double cubCoeffMag = std::max(std::abs(quarticEquation.c), quadCoeffMag);
	const double quartCoeffMag = std::max(std::abs(quarticEquation.b), cubCoeffMag);

	if (std::abs(quarticEquation.a) > quartCoeffMag * QUARTIC_THRESHHOLD)
	{
		auto res = quarticEquation.computeRoots();
		memcpy(&t[0], &res.x, sizeof(double) * 4);
	}
	else if (abs(quarticEquation.b) > quadCoeffMag * QUARTIC_THRESHHOLD)
	{
		auto res = Cubic<double>::construct(quarticEquation.b, quarticEquation.c, quarticEquation.d, quarticEquation.e).computeRoots();
		memcpy(&t[0], &res.x, sizeof(double) * 3);
	}
	else
	{
		auto res = Quadratic<double>::construct(quarticEquation.c, quarticEquation.d, quarticEquation.e).computeRoots();
		memcpy(&t[0], &res.x, sizeof(double) * 2);
	}
	
	// TODO: why did we do this?
	// if (t[0] == t[1] || core::isnan(t[0]) || core::isnan(t[1]))
	//	t[0] = (t[0] != 0.0) ? 0.0 : 1.0;
	
	return t;
}

double Hatch::intersectOrtho(const QuadraticBezier& bezier, double lineConstant, int component)
{
	// https://pomax.github.io/bezierinfo/#intersections
	double points[3];
	points[0] = bezier.P0[component];
	points[1] = bezier.P1[component];
	points[2] = bezier.P2[component];

	for (uint32_t i = 0; i < 3; i++)
		points[i] -= lineConstant;

	float64_t A = points[0] - 2.0 * points[1] + points[2];
	float64_t B = 2.0 * (points[1] - points[0]);
	float64_t C = points[0];

	float64_t2 roots = nbl::hlsl::math::equations::Quadratic<float64_t>::construct(A, B, C).computeRoots();
	if (roots.x >= 0.0 && roots.x <= 1.0) return roots.x;
	if (roots.y >= 0.0 && roots.y <= 1.0) return roots.y;
	return core::nan<double>();
}

bool Hatch::splitIntoMajorMonotonicSegments(const QuadraticBezier& bezier, std::array<Hatch::QuadraticBezier, 2>& out)
{
	auto quadratic = QuadraticCurve::constructFromBezier(bezier);

	// Getting derivatives for our quadratic bezier
	auto major = (uint32_t)SelectedMajorAxis;
	auto a = quadratic.A[major];
	auto b = quadratic.B[major];

	// Finding roots for the quadratic bezier derivatives (a straight line)
	auto t = -b / (2.0 * a);
	if (t <= 0.0 || t >= 1.0) return true;
	QuadraticBezier lower = bezier; lower.splitFromStart(t);
	QuadraticBezier upper = bezier; upper.splitToEnd(t);
	out = {lower, upper};
	return false;
}

// https://pomax.github.io/bezierinfo/#boundingbox
std::pair<float64_t2, float64_t2> Hatch::getBezierBoundingBoxMinor(const QuadraticBezier& bezier)
{
	auto minor = (uint32_t)SelectedMinorAxis;
	double A = bezier.P0[minor] - 2.0 * bezier.P1[minor] + bezier.P2[minor];
	double B = 2.0 * (bezier.P1[minor] - bezier.P0[minor]);

	const int searchTSize = 3;
	double searchT[searchTSize];
	searchT[0] = 0.0;
	searchT[1] = 1.0;
	searchT[2] = -B / (2 * A);

	float64_t2 min = float64_t2(std::numeric_limits<double>::infinity());
	float64_t2 max = float64_t2(-std::numeric_limits<double>::infinity());

	for (uint32_t i = 0; i < searchTSize; i++)
	{
		double t = searchT[i];
		if (t < 0.0 || t > 1.0 || core::isnan(t))
			continue;
		float64_t2 value = bezier.evaluate(t);
		min = float64_t2(std::min(min.x, value.x), std::min(min.y, value.y));
		max = float64_t2(std::max(max.x, value.x), std::max(max.y, value.y));
	}

	return std::pair<float64_t2, float64_t2>(min, max);
}

bool Hatch::isLineSegment(const QuadraticBezier& bezier)
{
	auto quadratic = QuadraticCurve::constructFromBezier(bezier);
	float64_t lenSqA = dot(quadratic.A, quadratic.A);
	return lenSqA < exp(-23.0f) * dot(quadratic.B, quadratic.B);
}

// TODO: the shape functions below should work with this instead of magic numbers
static constexpr float64_t FillPatternShapeExtent = 32.0;

void line(std::vector<CPolyline>& polylines, float64_t2 begin, float64_t2 end)
{
	std::vector<float64_t2> points = {
		begin, end
	};
	CPolyline polyline;
	polyline.addLinePoints(points);
	polylines.push_back(std::move(polyline));
}

void square(std::vector<CPolyline>& polylines, float64_t2 position, float64_t2 size = float64_t2(1, 1))
{
	std::array<float64_t2, 5u> points = {
		float64_t2(position.x, position.y),
		float64_t2(position.x, position.y + size.y),
		float64_t2(position.x + size.x, position.y + size.y),
		float64_t2(position.x + size.x, position.y),
		float64_t2(position.x, position.y)
	};
	CPolyline polyline;
	polyline.addLinePoints(points);
	polylines.push_back(std::move(polyline));
}

void checkered(std::vector<CPolyline>& polylines, const float64_t2& offset)
{
	CPolyline polyline;
	std::array<float64_t2, 5u> squarePointsCW = 
	{
		float64_t2(0.0, 1.0),
		float64_t2(0.5, 1.0),
		float64_t2(0.5, 0.5),
		float64_t2(0.0, 0.5),
		float64_t2(0.0, 1.0),
	};
	{
		std::vector<float64_t2> points;
		points.reserve(squarePointsCW.size());
		for (const auto& p : squarePointsCW) points.push_back(p * FillPatternShapeExtent + offset);
		polyline.addLinePoints(points);
	}
	{
		std::vector<float64_t2> points;
		points.reserve(squarePointsCW.size());
		for (const auto& p : squarePointsCW) points.push_back((p + float64_t2(0.5, -0.5)) * FillPatternShapeExtent + offset);
		polyline.addLinePoints(points);
	}
	polylines.push_back(std::move(polyline));
}

void diamonds(std::vector<CPolyline>& polylines, const float64_t2& offset)
{
	CPolyline polyline;
	float64_t innerSize = FillPatternShapeExtent / 2.0;
	float64_t outerSize = FillPatternShapeExtent;

	const std::array<float64_t2, 5u> diamondPointsCW = {
		float64_t2(0.0, 0.5),
		float64_t2(0.5, 0.0),
		float64_t2(0.0, -0.5),
		float64_t2(-0.5, 0.0),
		float64_t2(0.0, 0.5),
	};
	const std::array<float64_t2, 5u> diamondPointsCCW = {
		float64_t2(0.0, 0.5),
		float64_t2(-0.5, 0.0),
		float64_t2(0.0, -0.5),
		float64_t2(0.5, 0.0),
		float64_t2(0.0, 0.5),
	};

	float64_t2 origin = offset + float64_t2(FillPatternShapeExtent / 2.0, FillPatternShapeExtent / 2.0);

	// Outer
	{
		std::vector<float64_t2> points;
		points.reserve(diamondPointsCW.size());
		for (const auto& p : diamondPointsCW) points.push_back(p * outerSize + origin);
		polyline.addLinePoints(points);
	}
	// Inner
	{
		std::vector<float64_t2> points;
		points.reserve(diamondPointsCCW.size());
		for (const auto& p : diamondPointsCCW) points.push_back(p * innerSize + origin);
		polyline.addLinePoints(points);
	}
	polylines.push_back(std::move(polyline));
}

void crossHatch(std::vector<CPolyline>& polylines, const float64_t2& offset)
{
	CPolyline polyline;
	const std::array<float64_t2, 9u> outerPointsCW = {
			float64_t2(0.375, 0.0),
			float64_t2(0.0, 0.375),
			float64_t2(0.0, 0.625),
			float64_t2(0.375, 1.0),
			float64_t2(0.625, 1.0),
			float64_t2(1.0, 0.625),
			float64_t2(1.0, 0.375),
			float64_t2(0.625, 0.0),
			float64_t2(0.375, 0.0),
	};
	{
		std::vector<float64_t2> points;
		points.reserve(outerPointsCW.size());
		for (const auto& p : outerPointsCW) points.push_back(p * FillPatternShapeExtent + offset);
		polyline.addLinePoints(points);
	}
	
	const std::array<float64_t2, 5u> diamondPointsCCW = {
		float64_t2(0.0, 0.5),
		float64_t2(-0.5, 0.0),
		float64_t2(0.0, -0.5),
		float64_t2(0.5, 0.0),
		float64_t2(0.0, 0.5),
	};
	{
		float64_t2 origin = float64_t2(FillPatternShapeExtent/2.0, FillPatternShapeExtent/2.0) + offset;
		std::vector<float64_t2> points;
		points.reserve(diamondPointsCCW.size());
		for (const auto& p : diamondPointsCCW) points.push_back(p * 0.75 * FillPatternShapeExtent + origin);
		polyline.addLinePoints(points);
	}
	polylines.push_back(std::move(polyline));
}

void hatch(std::vector<CPolyline>& polylines, const float64_t2& offset)
{
	CPolyline polyline;

	float64_t2 basePt0 = float64_t2(FillPatternShapeExtent + 2.0, -2.0) + offset;
	float64_t2 basePt1 = float64_t2(-2.0, FillPatternShapeExtent  + 2.0) + offset;
	float64_t lineDiameter = 0.75;
	{
		float64_t2 radiusOffsetTL = float64_t2(+lineDiameter / 2.0, +lineDiameter / 2.0) * FillPatternShapeExtent / 8.0;
		float64_t2 radiusOffsetBL = float64_t2(-lineDiameter / 2.0, -lineDiameter / 2.0) * FillPatternShapeExtent / 8.0;
		std::vector<float64_t2> points = {
			basePt0 + radiusOffsetTL,
			basePt0 + radiusOffsetBL, // 0
			basePt1 + radiusOffsetBL, // 1
			basePt1 + radiusOffsetTL, // 2
			basePt0 + radiusOffsetTL
		};
		polyline.addLinePoints(points);
	}
	polylines.push_back(std::move(polyline));
}

void horizontal(std::vector<CPolyline>& polylines, const float64_t2& offset)
{
	CPolyline polyline;
	{
		std::array<float64_t2, 5u> points = {
			float64_t2(0.0, 3.0)/8.0 * FillPatternShapeExtent + offset ,
			float64_t2(0.0, 4.0)/8.0 * FillPatternShapeExtent + offset,
			float64_t2(8.0, 4.0)/8.0 * FillPatternShapeExtent + offset,
			float64_t2(8.0, 3.0)/8.0 * FillPatternShapeExtent + offset,
			float64_t2(0.0, 3.0)/8.0 * FillPatternShapeExtent + offset,
		};
		polyline.addLinePoints(points);
	}
	{
		std::array<float64_t2, 5u> points = {
			float64_t2(0.0, 7.0)/8.0 * FillPatternShapeExtent + offset,
			float64_t2(0.0, 8.0)/8.0 * FillPatternShapeExtent + offset,
			float64_t2(8.0, 8.0)/8.0 * FillPatternShapeExtent + offset,
			float64_t2(8.0, 7.0)/8.0 * FillPatternShapeExtent + offset,
			float64_t2(0.0, 7.0)/8.0 * FillPatternShapeExtent + offset,
		};
		polyline.addLinePoints(points);
	}
	polylines.push_back(std::move(polyline));
}

void vertical(std::vector<CPolyline>& polylines, const float64_t2& offset)
{
	CPolyline polyline;
	{
		std::array<float64_t2, 5u> points = {
			float64_t2(0.0, 0.0)/8.0 * FillPatternShapeExtent + offset,
			float64_t2(0.0, 8.0)/8.0 * FillPatternShapeExtent + offset,
			float64_t2(1.0, 8.0)/8.0 * FillPatternShapeExtent + offset,
			float64_t2(1.0, 0.0)/8.0 * FillPatternShapeExtent + offset,
			float64_t2(0.0, 0.0)/8.0 * FillPatternShapeExtent + offset,
		};
		polyline.addLinePoints(points);
	}
	{
		std::array<float64_t2, 5u> points = {
			float64_t2(4.0, 0.0)/8.0 * FillPatternShapeExtent + offset,
			float64_t2(4.0, 8.0)/8.0 * FillPatternShapeExtent + offset,
			float64_t2(5.0, 8.0)/8.0 * FillPatternShapeExtent + offset,
			float64_t2(5.0, 0.0)/8.0 * FillPatternShapeExtent + offset,
			float64_t2(4.0, 0.0)/8.0 * FillPatternShapeExtent + offset,
		};
		polyline.addLinePoints(points);
	}
	polylines.push_back(std::move(polyline));
}

void interwoven(std::vector<CPolyline>& polylines, const float64_t2& offset)
{
	CPolyline polyline;
	{
		std::array<float64_t2, 7u> points = {
			float64_t2(4.0, 0.0)/8.0 * FillPatternShapeExtent + offset,
			float64_t2(4.0, 1.0)/8.0 * FillPatternShapeExtent + offset, // 0
			float64_t2(7.0, 4.0)/8.0 * FillPatternShapeExtent + offset, // 1
			float64_t2(8.0, 4.0)/8.0 * FillPatternShapeExtent + offset, // 2
			float64_t2(8.0, 3.0)/8.0 * FillPatternShapeExtent + offset, // 3
			float64_t2(5.0, 0.0)/8.0 * FillPatternShapeExtent + offset, // 4
			float64_t2(4.0, 0.0)/8.0 * FillPatternShapeExtent + offset,
		};
		polyline.addLinePoints(points);
	}
	{
		std::array<float64_t2, 7u> points = {
			float64_t2(3.0, 4.0)/8.0 * FillPatternShapeExtent + offset,
			float64_t2(0.0, 7.0)/8.0 * FillPatternShapeExtent + offset, // 0
			float64_t2(0.0, 8.0)/8.0 * FillPatternShapeExtent + offset, // 1
			float64_t2(1.0, 8.0)/8.0 * FillPatternShapeExtent + offset, // 2
			float64_t2(4.0, 5.0)/8.0 * FillPatternShapeExtent + offset, // 3
			float64_t2(4.0, 4.0)/8.0 * FillPatternShapeExtent + offset, // 4
			float64_t2(3.0, 4.0)/8.0 * FillPatternShapeExtent + offset,
		};
		polyline.addLinePoints(points);
	}
	polylines.push_back(std::move(polyline));
}

void reverseHatch(std::vector<CPolyline>& polylines, const float64_t2& offset)
{
	CPolyline polyline;

	float64_t2 basePt0 = float64_t2(-2.0, -2.0) + offset;
	float64_t2 basePt1 = float64_t2(FillPatternShapeExtent + 2.0, FillPatternShapeExtent + 2.0) + offset;
	float64_t lineDiameter = 0.75;
	{
		float64_t2 radiusOffsetTL = float64_t2(-lineDiameter / 2.0, +lineDiameter / 2.0) * FillPatternShapeExtent / 8.0;
		float64_t2 radiusOffsetBL = float64_t2(+lineDiameter / 2.0, -lineDiameter / 2.0) * FillPatternShapeExtent / 8.0;
		std::vector<float64_t2> points = {
			basePt0 + radiusOffsetTL,
			basePt1 + radiusOffsetTL, // 0
			basePt1 + radiusOffsetBL, // 1
			basePt0 + radiusOffsetBL, // 2
			basePt0 + radiusOffsetTL
		};
		polyline.addLinePoints(points);
	}
	polylines.push_back(std::move(polyline));
}

void squares(std::vector<CPolyline>& polylines, const float64_t2& offset)
{
	CPolyline polyline;
	std::array<float64_t2, 5u> outerSquare = {
		float64_t2(1.0, 1.0)/8.0 * FillPatternShapeExtent + offset,
		float64_t2(1.0, 7.0)/8.0 * FillPatternShapeExtent + offset,
		float64_t2(7.0, 7.0)/8.0 * FillPatternShapeExtent + offset,
		float64_t2(7.0, 1.0)/8.0 * FillPatternShapeExtent + offset,
		float64_t2(1.0, 1.0)/8.0 * FillPatternShapeExtent + offset,
	};
	polyline.addLinePoints(outerSquare);
	std::array<float64_t2, 5u> innerSquare = {
		float64_t2(2.0, 2.0)/8.0 * FillPatternShapeExtent + offset,
		float64_t2(6.0, 2.0)/8.0 * FillPatternShapeExtent + offset,
		float64_t2(6.0, 6.0)/8.0 * FillPatternShapeExtent + offset,
		float64_t2(2.0, 6.0)/8.0 * FillPatternShapeExtent + offset,
		float64_t2(2.0, 2.0)/8.0 * FillPatternShapeExtent + offset,
	};
	polyline.addLinePoints(innerSquare);
	polylines.push_back(std::move(polyline));
}

void circle(std::vector<CPolyline>& polylines, const float64_t2& offset)
{
	CPolyline polyline;
	float64_t2 center = float64_t2(FillPatternShapeExtent / 2.0, FillPatternShapeExtent / 2.0) + offset;
	
	// outer
	{
		std::vector<shapes::QuadraticBezier<double>> quadBeziers;
		curves::EllipticalArcInfo myCurve;
		{
			myCurve.majorAxis = { FillPatternShapeExtent * 0.4375, 0.0 };
			myCurve.center = center;
			myCurve.angleBounds = {
				// starting from 2pi to 0.0 because our msdfs require filled shapes to be CW
				nbl::core::PI<double>() * 2.0,
				nbl::core::PI<double>() * 0.0
			};
			myCurve.eccentricity = 1.0; // circle
		}

		curves::Subdivision::AddBezierFunc addToBezier = [&](shapes::QuadraticBezier<double>&& info) -> void
			{
				quadBeziers.push_back(info);
			};

		curves::Subdivision::adaptive(myCurve, 0.1, addToBezier, 3u);
		polyline.addQuadBeziers(quadBeziers);
	}
	// inner
	{
		std::vector<shapes::QuadraticBezier<double>> quadBeziers;
		curves::EllipticalArcInfo myCurve;
		{
			myCurve.majorAxis = { FillPatternShapeExtent * 0.3125, 0.0 };
			myCurve.center = center;
			myCurve.angleBounds = {
				nbl::core::PI<double>() * 0.0,
				nbl::core::PI<double>() * 2.0
			};
			myCurve.eccentricity = 1.0; // circle
		}

		curves::Subdivision::AddBezierFunc addToBezier = [&](shapes::QuadraticBezier<double>&& info) -> void
			{
				quadBeziers.push_back(info);
			};

		curves::Subdivision::adaptive(myCurve, 0.1, addToBezier, 3u);
		polyline.addQuadBeziers(quadBeziers);
	}
	polylines.push_back(std::move(polyline));
}

void lightShaded(std::vector<CPolyline>& polylines, const float64_t2& offset)
{
	// Light shaded-2
	float64_t2 size = float64_t2(1.0, 1.0)/8.0 * FillPatternShapeExtent;

	square(polylines, float64_t2(0.0, 3.0)/8.0 * FillPatternShapeExtent + offset, size);
	square(polylines, float64_t2(0.0, 7.0)/8.0 * FillPatternShapeExtent + offset, size);
	square(polylines, float64_t2(2.0, 1.0)/8.0 * FillPatternShapeExtent + offset, size);
	square(polylines, float64_t2(2.0, 5.0)/8.0 * FillPatternShapeExtent + offset, size);
	square(polylines, float64_t2(4.0, 3.0)/8.0 * FillPatternShapeExtent + offset, size);
	square(polylines, float64_t2(4.0, 7.0)/8.0 * FillPatternShapeExtent + offset, size);
	square(polylines, float64_t2(6.0, 1.0)/8.0 * FillPatternShapeExtent + offset, size);
	square(polylines, float64_t2(6.0, 5.0)/8.0 * FillPatternShapeExtent + offset, size);
}

void shaded(std::vector<CPolyline>& polylines, const float64_t2& offset)
{
	for (uint32_t x = 0; x < 8; x++)
	{
		for (uint32_t y = 0; y < 8; y++)
		{
			if (x % 2 != y % 2)
				square(polylines, float64_t2((double)x, (double)y)/8.0 * FillPatternShapeExtent + offset);
		}
	}

}

core::smart_refctd_ptr<asset::ICPUImage> Hatch::generateHatchFillPatternMSDF(nbl::ext::TextRendering::TextRenderer* textRenderer, HatchFillPattern fillPattern, uint32_t2 msdfExtents)
{
	std::array<float64_t2, 9u> offsets = {};
	uint32_t idx = 0u;
	for (int32_t i = -1; i <= 1; ++i)
		for (int32_t j = -1; j <= 1; ++j)
			offsets[idx++] = float64_t2(FillPatternShapeExtent * (float64_t)i, FillPatternShapeExtent * (float64_t)j);

	std::vector<CPolyline> polylines;
	
	// float64_t2 offset = float64_t2(0.0, 0.0);
	for (const auto& offset : offsets)
	{
		switch (fillPattern)
		{
		case HatchFillPattern::CHECKERED:
			checkered(polylines, offset);
			break;
		case HatchFillPattern::DIAMONDS:
			diamonds(polylines, offset);
			break;
		case HatchFillPattern::CROSS_HATCH:
			crossHatch(polylines, offset);
			break;
		case HatchFillPattern::HATCH:
			hatch(polylines, offset);
			break;
		case HatchFillPattern::HORIZONTAL:
			horizontal(polylines, offset);
			break;
		case HatchFillPattern::VERTICAL:
			vertical(polylines, offset);
			break;
		case HatchFillPattern::INTERWOVEN:
			interwoven(polylines, offset);
			break;
		case HatchFillPattern::REVERSE_HATCH:
			reverseHatch(polylines, offset);
			break;
		case HatchFillPattern::SQUARES:
			squares(polylines, offset);
			break;
		case HatchFillPattern::CIRCLE:
			circle(polylines, offset);
			break;
		case HatchFillPattern::LIGHT_SHADED:
			lightShaded(polylines, offset);
			break;
		case HatchFillPattern::SHADED:
			shaded(polylines, offset);
			break;
		default:
			break;
		}
	}

	// Generate MSDFgen Shape
	msdfgen::Shape glyph;
	nbl::ext::TextRendering::GlyphShapeBuilder glyphShapeBuilder(glyph);
	for (uint32_t polylineIdx = 0; polylineIdx < polylines.size(); polylineIdx++)
	{
		auto& polyline = polylines[polylineIdx];
		for (uint32_t sectorIdx = 0; sectorIdx < polyline.getSectionsCount(); sectorIdx++)
		{
			auto& section = polyline.getSectionInfoAt(sectorIdx);
			if (section.type == ObjectType::LINE)
			{
				if (section.count == 0u) continue;

				glyphShapeBuilder.moveTo(polyline.getLinePointAt(section.index).p);
				for (uint32_t i = section.index + 1; i < section.index + section.count + 1; i++)
					glyphShapeBuilder.lineTo(polyline.getLinePointAt(i).p);
			}
			else if (section.type == ObjectType::QUAD_BEZIER)
			{
				if (section.count == 0u) continue;
				glyphShapeBuilder.moveTo(polyline.getQuadBezierInfoAt(section.index).shape.P0);
				for (uint32_t i = section.index; i < section.index + section.count; i++)
				{
					const auto& bez = polyline.getQuadBezierInfoAt(i).shape;
					glyphShapeBuilder.quadratic(bez.P1, bez.P2);
				}
			}
		}
	}
	glyphShapeBuilder.finish();
	glyph.normalize();

	float scaleX = (1.0 / float(FillPatternShapeExtent)) * float(msdfExtents.x);
	float scaleY = (1.0 / float(FillPatternShapeExtent)) * float(msdfExtents.y);

	auto bufferSize = msdfExtents.x * msdfExtents.y * sizeof(uint8_t) * 4;
	auto buffer = core::make_smart_refctd_ptr<ICPUBuffer>(bufferSize);
	size_t bufferOffset = 0ull;
	textRenderer->generateShapeMSDF(buffer.get(), &bufferOffset, glyph, MSDFPixelRange, msdfExtents, float32_t2(scaleX, scaleY), float32_t2(0, 0));
	assert(bufferOffset == bufferSize);

	ICPUImage::SCreationParams imgParams;
	{
		imgParams.flags = static_cast<ICPUImage::E_CREATE_FLAGS>(0u); // no flags
		imgParams.type = ICPUImage::ET_2D;
		imgParams.format = nbl::ext::TextRendering::TextRenderer::MSDFTextureFormat;
		imgParams.extent = { uint32_t(MSDFSize), uint32_t(MSDFSize), 1 };
		imgParams.mipLevels = 1u;
		imgParams.arrayLayers = 1u;
		imgParams.samples = ICPUImage::ESCF_1_BIT;
	}

	auto image = ICPUImage::create(std::move(imgParams));
	auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<IImage::SBufferCopy>>(1u);
	{
		auto& region = regions->front();
		region.bufferOffset = 0u;
		region.bufferRowLength = 0u;
		region.bufferImageHeight = 0u;
		region.imageSubresource.aspectMask = asset::IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
		region.imageSubresource.mipLevel = 0u;
		region.imageSubresource.baseArrayLayer = 0u;
		region.imageSubresource.layerCount = 1u;
		region.imageOffset = { 0u,0u,0u };
		region.imageExtent = { uint32_t(MSDFSize), uint32_t(MSDFSize), 1 };
	}
	image->setBufferAndRegions(std::move(buffer), std::move(regions));

	return image;
}
