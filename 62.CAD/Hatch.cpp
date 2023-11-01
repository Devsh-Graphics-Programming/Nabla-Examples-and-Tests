
#include "Hatch.h"

#define DEBUG_HATCH_VISUALLY

// Intended to mitigate issues with NaNs and precision by falling back to using simpler functions when the higher roots are small enough
std::array<double, 4> Hatch::solveQuarticRoots(double a, double b, double c, double d, double e, double t_start, double t_end)
{
	constexpr double QUARTIC_THRESHHOLD = 1e-10;

	std::array<double, 4> t = { -1.0, -1.0, -1.0, -1.0 }; // only two candidates in range, ever

	const double quadCoeffMag = std::max(std::abs(d), std::abs(e));
	const double cubCoeffMag = std::max(std::abs(c), quadCoeffMag);
	if (std::abs(a) > std::max(std::abs(b), cubCoeffMag) * QUARTIC_THRESHHOLD)
	{
		auto res = equations::Quartic<double>::construct(a, b, c, d, e).computeRoots();
		memcpy(&t[0], &res.x, sizeof(double) * 4);
	}
	else if (abs(b) > quadCoeffMag * QUARTIC_THRESHHOLD)
	{
		auto res = equations::Cubic<double>::construct(b, c, d, e).computeRoots();
		memcpy(&t[0], &res.x, sizeof(double) * 3);
	}
	else
	{
		auto res = equations::Quadratic<double>::construct(c, d, e).computeRoots();
		memcpy(&t[0], &res.x, sizeof(double) * 2);
	}

	// If either is NaN or both are equal
	// Same as: 
	// if (t[0] == t[1] || isnan(t[0]) || isnan(t[1]))
	if (!(t[0] != t[1]))
		t[0] = t[0] != t_start ? t_start : t_end;

	//// TODO: check that this clamp works with t[i] as NaN
	//for (auto i = 0; i < 2; i++)
	//	t[i] = nbl::core::clamp(t[i], t_start, t_end);
	//
	//// fix up a fuckup where both roots are NaN or were on the same side of the valid integral
	//
	//// TODO: do some Halley or Householder method steps on t
	////while ()
	////{
	////}

	// neither t still not in range, your beziers don't intersect
	return t;
}

Hatch::QuadraticBezier Hatch::splitCurveRange(const QuadraticBezier& bezier, double minT, double maxT)
{
	assert(maxT > minT);
	assert(0.0 <= minT && minT <= 1.0);
	assert(0.0 <= maxT && maxT <= 1.0);
    return splitCurveTakeUpper(splitCurveTakeLower(bezier, maxT), minT / maxT);
}

Hatch::QuadraticBezier Hatch::Segment::getSplitCurve() const
{
    return splitCurveRange(*originalBezier, t_start, t_end);
}

bool Hatch::Segment::isStraightLineConstantMajor() const
{
	auto major = (uint)SelectedMajorAxis;
	const double p0 = originalBezier->P0[major], 
		p1 = originalBezier->P1[major], 
		p2 = originalBezier->P2[major];
	//assert(p0 <= p1 && p1 <= p2); (PRECISION ISSUES ARISE ONCE MORE)
	return abs(p1 - p0) <= exp2(-24) && abs(p2 - p0) <= exp(-24);
}

// copied from curves.cpp
//TODO: move this to cpp-compat hlsl builtins
static float64_t2 LineLineIntersection(const float64_t2& p1, const float64_t2& v1, const float64_t2& p2, const float64_t2& v2)
{
	float64_t denominator = v1.y * v2.x - v1.x * v2.y;
	float64_t2 diff = p1 - p2;
	float64_t numerator = dot(float64_t2(v2.y, -v2.x), float64_t2(diff.x, diff.y));

	if (abs(denominator) < 1e-15 && abs(numerator) < 1e-15)
	{
		// are parallel and the same
		return (p1 + p2) / 2.0;
	}

	float64_t t = numerator / denominator;
	float64_t2 intersectionPoint = p1 + t * v1;
	return intersectionPoint;
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
		auto intersectionPoint = LineLineIntersection(
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
			{D.x, -D.y}, 
			{D.y, D.x} 
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
		auto p0 = originalBezier->P0;
		auto p1 = originalBezier->P1;
		auto p2 = originalBezier->P2;
		bool sideP1 = nbl::core::sign((p2.x - p0.x) * (p1.y - p0.y) - (p2.y - p0.y) * (p1.x - p0.x));

		auto otherBezier = *other.originalBezier;
		const std::array<double, 4> intersections = linePossibleIntersections(*originalBezier, otherBezier);

		for (uint32_t i = 0; i < intersections.size(); i++)
		{
			auto t = intersections[i];
			if (other.t_start >= t || t >= other.t_end)
				continue;

			auto intersection = other.originalBezier->evaluate(t);
			bool sideIntersection = nbl::core::sign<double>((p2.x - p0.x) * (intersection.y - p0.y) - (p2.y - p0.y) * (intersection.x - p0.x));

			// If both P1 and the intersection point are on the same side of the P0 -> P2 line
			// for the current line, we consider this as a valid intersection
			// (Otherwise, continue)
			if (sideP1 != sideIntersection)
				continue;

			result[resultIdx] = t;
			resultIdx++;
		}
	}

	return result;
}

Hatch::Hatch(core::SRange<CPolyline> lines, const MajorAxis majorAxis, int32_t& debugStep, std::function<void(CPolyline, CPULineStyle)> debugOutput /* tmp */)
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
    auto drawDebugBezier = [&](QuadraticBezier bezier, float32_t4 color)
    {
        CPolyline outputPolyline;
        std::vector<QuadraticBezierInfo> beziers;
        QuadraticBezierInfo bezierInfo;
        bezierInfo.p[0] = bezier.P0;
        bezierInfo.p[1] = bezier.P1;
        bezierInfo.p[2] = bezier.P2;
        beziers.push_back(bezierInfo);
        outputPolyline.addQuadBeziers(core::SRange<QuadraticBezierInfo>(beziers.data(), beziers.data() + beziers.size()));

        CPULineStyle cpuLineStyle;
        cpuLineStyle.screenSpaceLineWidth = 4.0f;
        cpuLineStyle.worldSpaceLineWidth = 0.0f;
        cpuLineStyle.color = color;

        debugOutput(outputPolyline, cpuLineStyle);
    };

    auto drawDebugLine = [&](float64_t2 start, float64_t2 end, float32_t4 color)
    {
        CPolyline outputPolyline;
        std::vector<float64_t2> points;
        points.push_back(start);
        points.push_back(end);
        outputPolyline.addLinePoints(core::SRange<float64_t2>(points.data(), points.data() + points.size()));
        
        CPULineStyle cpuLineStyle;
        cpuLineStyle.screenSpaceLineWidth = 2.0f;
        cpuLineStyle.worldSpaceLineWidth = 0.0f;
        cpuLineStyle.color = color;
        
        debugOutput(outputPolyline, cpuLineStyle);
    };
#endif

    {
        std::vector<Segment> segments;
        for (CPolyline& polyline : lines)
        {
            for (uint32_t secIdx = 0; secIdx < polyline.getSectionsCount(); secIdx ++)
            {
				auto addBezier = [&](QuadraticBezier bezier)
				{
					auto outputBezier = bezier;
					if (outputBezier.evaluate(0.0)[major] > outputBezier.evaluate(1.0)[major])
					{
						outputBezier.P2 = bezier.P0;
						outputBezier.P0 = bezier.P2;
						assert(outputBezier.evaluate(0.0)[major] <= outputBezier.evaluate(1.0)[major]);
					}

#ifdef DEBUG_HATCH_VISUALLY
					//if (debugOutput)
						//drawDebugBezier(outputBezier, float32_t4(0.0, 0.0, 0.0, 1.0));
#endif
					beziers.push_back(outputBezier);
				};

                auto section = polyline.getSectionInfoAt(secIdx);
                if (section.type == ObjectType::LINE)
                    {
						for (uint32_t itemIdx = section.index; itemIdx < section.index + section.count; itemIdx++)
						{
							auto begin = polyline.getLinePointAt(itemIdx);
							auto end = polyline.getLinePointAt(itemIdx + 1);
							addBezier(QuadraticBezier::construct(begin, (begin + end) * 0.5, end));
						}
					}
                else if (section.type == ObjectType::QUAD_BEZIER)
                {
                    for (uint32_t itemIdx = section.index; itemIdx < section.index + section.count; itemIdx ++)
                    {
                        auto bezierInfo = polyline.getQuadBezierInfoAt(itemIdx);
                        auto unsplitBezier = QuadraticBezier::construct(bezierInfo.p[0], bezierInfo.p[1], bezierInfo.p[2]);
                        
                        // Beziers must be monotonically increasing along major
                        // First step: Make sure the bezier is monotonic, split it if not
                        std::array<QuadraticBezier, 2> monotonicSegments;
                        auto isMonotonic = splitIntoMajorMonotonicSegments(unsplitBezier, monotonicSegments);

                        if (isMonotonic)
                        {
                            // Already was monotonic
                            addBezier(unsplitBezier);
#ifdef DEBUG_HATCH_VISUALLY
                            if (debugOutput)
                                drawDebugBezier(unsplitBezier, float32_t4(0.8, 0.8, 0.8, 0.8));
#endif
                        }
                        else
                        {
                            addBezier(monotonicSegments.data()[0]);
                            addBezier(monotonicSegments.data()[1]);
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
			drawDebugLine(float64_t2(_lhs, -1000.0), float64_t2(_lhs, 1000.0), float64_t4(0.1, 0.1, 1.0, 1.0));
			drawDebugLine(float64_t2(_rhs, -1000.0), float64_t2(_rhs, 1000.0), float64_t4(0.1, 1.0, 1.0, 1.0));
			printf(std::format("(comparing minor) _lhs: {} (len: {}) _rhs: {} (len: {}) minLen: {} diff: {} ",
				_lhs, lenLhs, _rhs, lenRhs, minLen, abs(_lhs - _rhs)).c_str());
		}
#endif

		// Threshhold here for intersection points, where the minor values for the curves are
		// very close but could be smaller, causing the curves to be in the wrong order
		if (abs(_lhs - _rhs) < MinorPositionComparisonThreshhold * minLen)
		{
			// this is how you want to order the derivatives dmin/dmaj=-INF dmin/dmaj = 0 dmin/dmaj=INF
			// also leverage the guarantee that `dmaj>=0` to ger numerically stable compare
			float64_t2 lTan = tangent(*lhs.originalBezier, lhs.t_start);
			float64_t2 rTan = tangent(*rhs.originalBezier, rhs.t_start);
			_lhs = lTan[minor] * rTan[major];
			_rhs = rTan[minor] * lTan[major];
#ifdef DEBUG_HATCH_VISUALLY
			if (debugOutput && step == debugStep)
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

			if (abs(_lhs - _rhs) < TangentComparisonThreshhold && abs(_lhs - 0.0) < TangentComparisonThreshhold)
			{ 
				// TODO https://discord.com/channels/593902898015109131/723305695046533151/1169377896658383008
				bool lTanSign = lTan[minor] >= 0.0;
				bool rTanSign = rTan[minor] >= 0.0;
				// If the signs differ, negative one is the left curve
				if (lTanSign != rTanSign)
				{
#ifdef DEBUG_HATCH_VISUALLY
					if (debugOutput && step == debugStep)
					{
						printf(std::format("(comparing sign) lTanSign: {} rTanSign: {} ",
							lTanSign ? "positive" : "negative", rTanSign ? "positive" : "negative").c_str());
					}
#endif
					// We want to return true if lhs < rhs (lhs is to the left of rhs)
					// For this to be false, rhs would need to be the left one (and therefore negative)
					return rTanSign;
				}

				// Otherwise (CASE B)
				// 
				// In this case tangents are both on the same side.
				// so only the magnitude / abs of the d2Major / dMinor2 is important
				auto lhsQuadratic = QuadraticEquation::constructFromBezier(*lhs.originalBezier);
				auto rhsQuadratic = QuadraticEquation::constructFromBezier(*rhs.originalBezier);

				_lhs = -abs(lhsQuadratic.A.x * lhsQuadratic.B.y - lhsQuadratic.A.y * lhsQuadratic.B.x) * (rhsQuadratic.B.x * rhsQuadratic.B.x * rhsQuadratic.B.x) ;
				_rhs = -abs(rhsQuadratic.A.x * rhsQuadratic.B.y - rhsQuadratic.A.y * rhsQuadratic.B.x) * (lhsQuadratic.B.x * lhsQuadratic.B.x * lhsQuadratic.B.x);
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
				//drawDebugBezier(entry.getSplitCurve(), float64_t4(1.0, 0.0, 0.0, 1.0));
				//drawDebugBezier(segment.getSplitCurve(), float64_t4(0.0, 1.0, 0.0, 1.0));

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
                //std::cout << "Start event at " << newMajor << "\n";
            }
            // (intersection event)
            else newMajor = intersectionVisit();
        }
        // next intersection event is before next end event
        // (intersection event)
        else if (!intersections.empty() && intersections.top() < maxMajorEnds)
            newMajor = intersectionVisit();
        // (end event)
        else
        {
            newMajor = maxMajorEnds;
            ends.pop();
#ifdef DEBUG_HATCH_VISUALLY
            if (debugOutput && isCurrentDebugStep)
                drawDebugLine(float64_t2(-1000.0, newMajor), float64_t2(1000.0, newMajor), float32_t4(0.0, 0.0, 0.8, 1.0));
#endif
            //std::cout << "End event at " << newMajor << "\n";
        }
        // spawn quads for the previous iterations if we advanced
        //std::cout << "New major: " << newMajor << " Last major: " << lastMajor << "\n";
        if (newMajor > lastMajor)
        {
#ifdef DEBUG_HATCH_VISUALLY
			if (debugOutput && isCurrentDebugStep)
				drawDebugLine(float64_t2(-1000.0, lastMajor), float64_t2(1000.0, lastMajor), float32_t4(0.1, 0.1, 0.0, 0.5));
#endif
            // trim
            const auto candidatesSize = std::distance(activeCandidates.begin(),activeCandidates.end());
			//std::cout << "Candidates size: " << candidatesSize << "\n";
            // because n4ce works on loops, this must be true
            assert((candidatesSize % 2u)==0u);
            for (auto i=0u; i< candidatesSize;)
            {
                const Segment& left = activeCandidates[i++];
                const Segment& right = activeCandidates[i++];

                CurveHatchBox curveBox;

				// Due to precision, if the curve is right at the end, intersectOrtho may return nan
				auto curveMinEnd = intersectOrtho(*left.originalBezier, newMajor, major);
				auto curveMaxEnd = intersectOrtho(*right.originalBezier, newMajor, major);

                auto splitCurveMin = splitCurveRange(*left.originalBezier, left.t_start, isnan(curveMinEnd) ? 1.0 : curveMinEnd);
				auto splitCurveMax = splitCurveRange(*right.originalBezier, right.t_start, isnan(curveMaxEnd) ? 1.0 : curveMaxEnd);
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
				}
#endif

                // Transform curves into AABB UV space and turn them into quadratic coefficients
                // so we wont need to convert here
				auto transformCurves = [](Hatch::QuadraticBezier bezier, float64_t2 aabbMin, float64_t2 aabbMax, uint32_t2* output) {
					auto rcpAabbExtents = float64_t2(1.0, 1.0) / (aabbMax - aabbMin);
					auto transformedBezier = QuadraticBezier::construct(
						(bezier.P0 - aabbMin) * rcpAabbExtents,
						(bezier.P1 - aabbMin) * rcpAabbExtents,
						(bezier.P2 - aabbMin) * rcpAabbExtents
					);
					auto quadratic = QuadraticEquation::constructFromBezier(transformedBezier);

					if (isLineSegment(transformedBezier))
						quadratic.A = float64_t2(0.0);

					// TODO: Use common.hlsl packCurveBoxUnorm & packCurveBoxSnorm after fixing nbl::hlsl::numeric_limits<uint32_t> on c++
					auto convertToUnorm = [](float64_t2 value) {
						return static_cast<uint32_t2>(value * float64_t2(static_cast<double>(std::numeric_limits<uint32_t>::max())));
					};
					auto convertToSnorm = [](float64_t2 value) {
						auto tmp = static_cast<int32_t2>(value * float64_t2(static_cast<double>(std::numeric_limits<int32_t>::max())));
						uint32_t2 out;
						std::memcpy(&out, &tmp, sizeof(tmp));
						return out;
					};
					
					// Values P0, P1 & P2 are in [0,1]
					//
					// A = P0 - 2 * P1 + P2; Range: [-2, 2]
					// B = 2 * (P1 - P0); Range: [-2, 2]
					// C = P0; Range: [0, 1]
					//
					// Convert A, B to [-1, 1], encode as Snorm
					// Convert C to [0, 1], encode as Unorm
					output[0] = convertToSnorm(quadratic.A / 2.0);
					output[1] = convertToSnorm(quadratic.B / 2.0);
					output[2] = convertToUnorm(quadratic.C);

					// B == 0.0 && A == 0.0 would mean this is a constant line in major direction, which
					// should've been ruled out at this point (isStraightLineCosntantMajor gets skipped)
					assert(quadratic.A.y != 0.0 || quadratic.B.y != 0.0);
                };
                transformCurves(splitCurveMin, curveBox.aabbMin, curveBox.aabbMax, &curveBox.curveMin[0]);
                transformCurves(splitCurveMax, curveBox.aabbMin, curveBox.aabbMax, &curveBox.curveMax[0]);

                hatchBoxes.push_back(curveBox);
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
			if (newMajor > lastMajor)
				lastMajor = newMajor;
		}

#ifdef DEBUG_HATCH_VISUALLY
		step++;
#endif
    }
#ifdef DEBUG_HATCH_VISUALLY
	debugStep = debugStep - step;
#endif
}


// returns two possible values of t in the second curve where the curves intersect
std::array<double, 4> Hatch::linePossibleIntersections(const QuadraticBezier& bezier, const QuadraticBezier& second)
{
	// Algorithm based on Computer Aided Geometric Design: 
	// https://scholarsarchive.byu.edu/cgi/viewcontent.cgi?article=1000&context=facpub#page99
	// Chapter 17.6 describes the implicitization of a curve, which transforms it into the following format:
	// Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
	// 
	// We then substitute x and y for the other curve's quadratic formulas.
	// This gives us a quartic equation:
	// At^4 + Bt^3 + Ct^2 + Dt + E = 0
	//
	// The roots for t then become our intersections points between both curves.
	//
	// A Desmos page including math for this as well as some of the graphs it generates is available here:
	// https://www.desmos.com/calculator/mjwqvnvyb8?lang=pt-BR

	double p0x = bezier.P0.x, p1x = bezier.P1.x, p2x = bezier.P2.x,
		p0y = bezier.P0.y, p1y = bezier.P1.y, p2y = bezier.P2.y;

	// Getting the values for the implicitization of the curve
	double t0 = (4 * p0y * p1y) - (4 * p0y * p2y) - (4 * (p1y * p1y)) + (4 * p1y * p2y) - ((p0y * p0y)) + (2 * p0y * p2y) - ((p2y * p2y));
	double t1 = -(4 * p0x * p1y) + (4 * p0x * p2y) - (4 * p1x * p0y) + (8 * p1x * p1y) - (4 * p1x * p2y) + (4 * p2x * p0y) - (4 * p2x * p1y) + (2 * p0x * p0y) - (2 * p0x * p2y) - (2 * p2x * p0y) + (2 * p2x * p2y);
	double t2 = (4 * p0x * p1x) - (4 * p0x * p2x) - (4 * (p1x * p1x)) + (4 * p1x * p2x) - ((p0x * p0x)) + (2 * p0x * p2x) - ((p2x * p2x));
	double t3 = (4 * p0x * (p1y * p1y)) - (4 * p0x * p1y * p2y) - (4 * p1x * p0y * p1y) + (8 * p1x * p0y * p2y) - (4 * p1x * p1y * p2y) - (4 * p2x * p0y * p1y) + (4 * p2x * (p1y * p1y)) - (2 * p0x * p0y * p2y) + (2 * p0x * (p2y * p2y)) + (2 * p2x * (p0y * p0y)) - (2 * p2x * p0y * p2y);
	double t4 = -(4 * p0x * p1x * p1y) - (4 * p0x * p1x * p2y) + (8 * p0x * p2x * p1y) + (4 * (p1x * p1x) * p0y) + (4 * (p1x * p1x) * p2y) - (4 * p1x * p2x * p0y) - (4 * p1x * p2x * p1y) + (2 * (p0x * p0x) * p2y) - (2 * p0x * p2x * p0y) - (2 * p0x * p2x * p2y) + (2 * (p2x * p2x) * p0y);
	double t5 = (4 * p0x * p1x * p1y * p2y) - (4 * (p1x * p1x) * p0y * p2y) + (4 * p1x * p2x * p0y * p1y) - ((p0x * p0x) * (p2y * p2y)) + (2 * p0x * p2x * p0y * p2y) - ((p2x * p2x) * (p0y * p0y)) - (4 * p0x * p2x * (p1y * p1y));

	// "Slam" the other curve onto it

	auto quadratic = QuadraticEquation::constructFromBezier(second);
	float64_t2 A = quadratic.A, B = quadratic.B, C = quadratic.C;

	// Getting the quartic params
	double a = ((A.x * A.x) * t0) + (A.x * A.y * t1) + (A.y * t2);
	double b = (2 * A.x * B.x * t0) + (A.x * B.y * t1) + (B.x * A.y * t1) + (2 * A.y * B.y * t2);
	double c = (2 * A.x * C.x * t0) + (A.x * C.y * t1) + (A.x * t3) + ((B.x * B.x) * t0) + (B.x * B.y * t1) + (C.x * A.y * t1) + (2 * A.y * C.y * t2) + (A.y * t4) + ((B.y * B.y) * t2);
	double d = (2 * B.x * C.x * t0) + (B.x * C.y * t1) + (B.x * t3) + (C.x * B.y * t1) + (2 * B.y * C.y * t2) + (B.y * t4);
	double e = ((C.x * C.x) * t0) + (C.x * C.y * t1) + (C.x * t3) + ((C.y * C.y) * t2) + (C.y * t4) + (t5);

	return Hatch::solveQuarticRoots(a, b, c, d, e, 0.0, 1.0);
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

	float64_t2 roots = nbl::hlsl::equations::Quadratic<float64_t>::construct(A, B, C).computeRoots();
	if (roots.x >= 0.0 && roots.x <= 1.0) return roots.x;
	if (roots.y >= 0.0 && roots.y <= 1.0) return roots.y;
	return core::nan<double>();
}

// https://pomax.github.io/bezierinfo/#pointvectors
float64_t2 Hatch::tangent(const QuadraticBezier& bezier, double t)
{
	auto derivativeOrder1First = 2.0 * (bezier.P1 - bezier.P0);
	auto derivativeOrder1Second = 2.0 * (bezier.P2 - bezier.P1);
	auto tangent = (1.0 - t) * derivativeOrder1First + t * derivativeOrder1Second;
	return glm::normalize(tangent);
}

Hatch::QuadraticBezier Hatch::splitCurveTakeLower(const QuadraticBezier& bezier, double t)
{
	QuadraticBezier outputCurve;
	outputCurve.P0 = bezier.P0;
	outputCurve.P1 = (1 - t) * bezier.P0 + t * bezier.P1;
	outputCurve.P2 = (1 - t) * ((1 - t) * bezier.P0 + t * bezier.P1) + t * ((1 - t) * bezier.P1 + t * bezier.P2);
	//assert(outputCurve.evaluate(0.0)[(int)SelectedMajorAxis] <= outputCurve.evaluate(1.0)[(int)SelectedMajorAxis]);

	return outputCurve;
}

Hatch::QuadraticBezier Hatch::splitCurveTakeUpper(const QuadraticBezier& bezier, double t)
{
	QuadraticBezier outputCurve;
	outputCurve.P0 = (1 - t) * ((1 - t) * bezier.P0 + t * bezier.P1) + t * ((1 - t) * bezier.P1 + t * bezier.P2);
	outputCurve.P1 = (1 - t) * bezier.P1 + t * bezier.P2;
	outputCurve.P2 = bezier.P2;
	//assert(outputCurve.evaluate(0.0)[(int)SelectedMajorAxis] <= outputCurve.evaluate(1.0)[(int)SelectedMajorAxis]);

	return outputCurve;
}

bool Hatch::splitIntoMajorMonotonicSegments(const QuadraticBezier& bezier, std::array<Hatch::QuadraticBezier, 2>& out)
{
	// Getting derivatives for our quadratic bezier
	auto major = (uint)SelectedMajorAxis;
	auto a = 2.0 * (bezier.P1[major] - bezier.P0[major]);
	auto b = 2.0 * (bezier.P2[major] - bezier.P1[major]);

	// Finding roots for the quadratic bezier derivatives (a straight line)
	auto rcp = 1.0 / (b - a);
	auto t = -a * rcp;
	if (isinf(rcp) || t <= 0.0 || t >= 1.0) return true;
	out = { splitCurveTakeLower(bezier, t), splitCurveTakeUpper(bezier, t) };
	return false;
}

// https://pomax.github.io/bezierinfo/#boundingbox
std::pair<float64_t2, float64_t2> Hatch::getBezierBoundingBoxMinor(const QuadraticBezier& bezier)
{
	auto minor = (uint)SelectedMinorAxis;
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
		if (t < 0.0 || t > 1.0 || isnan(t))
			continue;
		float64_t2 value = bezier.evaluate(t);
		min = float64_t2(std::min(min.x, value.x), std::min(min.y, value.y));
		max = float64_t2(std::max(max.x, value.x), std::max(max.y, value.y));
	}

	return std::pair<float64_t2, float64_t2>(min, max);
}

bool Hatch::isLineSegment(const QuadraticBezier& bezier)
{
	auto quadratic = QuadraticEquation::constructFromBezier(bezier);
	float64_t lenSqA = dot(quadratic.A, quadratic.A);
	return lenSqA < exp(-23.0f) * dot(quadratic.B, quadratic.B);
}
