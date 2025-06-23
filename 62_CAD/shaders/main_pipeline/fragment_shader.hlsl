#define FRAGMENT_SHADER_INPUT
#include "common.hlsl"
#include "dtm.hlsl"
#include <nbl/builtin/hlsl/shapes/beziers.hlsl>
#include <nbl/builtin/hlsl/shapes/line.hlsl>
#include <nbl/builtin/hlsl/algorithm.hlsl>
#include <nbl/builtin/hlsl/math/equations/quadratic.hlsl>
#include <nbl/builtin/hlsl/math/geometry.hlsl>
#include <nbl/builtin/hlsl/spirv_intrinsics/fragment_shader_pixel_interlock.hlsl>
#include <nbl/builtin/hlsl/text_rendering/msdf.hlsl>
//#include <nbl/builtin/hlsl/spirv_intrinsics/fragment_shader_barycentric.hlsl>

// sdf of Isosceles Trapezoid y-aligned by https://iquilezles.org/articles/distfunctions2d/
float sdTrapezoid(float2 p, float r1, float r2, float he)
{
    float2 k1 = float2(r2, he);
    float2 k2 = float2(r2 - r1, 2.0 * he);

    p.x = abs(p.x);
    float2 ca = float2(max(0.0, p.x - ((p.y < 0.0) ? r1 : r2)), abs(p.y) - he);
    float2 cb = p - k1 + k2 * clamp(dot(k1 - p, k2) / dot(k2,k2), 0.0, 1.0);

    float s = (cb.x < 0.0 && ca.y < 0.0) ? -1.0 : 1.0;

    return s * sqrt(min(dot(ca,ca), dot(cb,cb)));
}

// line segment sdf which returns the distance vector specialized for usage in hatch box line boundaries
float2 sdLineDstVec(float2 P, float2 A, float2 B)
{
    const float2 PA = P - A;
    const float2 BA = B - A;
    float h = clamp(dot(PA, BA) / dot(BA, BA), 0.0, 1.0);
    return PA - BA * h;
}

float miterSDF(float2 p, float thickness, float2 a, float2 b, float ra, float rb)
{
    float h = length(b - a) / 2.0;
    float2 d = normalize(b - a);
    float2x2 rot = float2x2(d.y, -d.x, d.x, d.y);
    p = mul(rot, p);
    p.y -= h - thickness;
    return sdTrapezoid(p, ra, rb, h);
}

// We need to specialize color calculation based on FragmentShaderInterlock feature availability for our transparency algorithm
// because there is no `if constexpr` in hlsl
// @params
// textureColor: color sampled from a texture
// useStyleColor: instead of writing and reading from colorStorage, use main object Idx to find the style color for the object.
template<bool FragmentShaderPixelInterlock>
float32_t4 calculateFinalColor(const uint2 fragCoord, const float localAlpha, const uint32_t currentMainObjectIdx, float3 textureColor, bool colorFromTexture);

template<>
float32_t4 calculateFinalColor<false>(const uint2 fragCoord, const float localAlpha, const uint32_t currentMainObjectIdx, float3 localTextureColor, bool colorFromTexture)
{
    uint32_t styleIdx = loadMainObject(currentMainObjectIdx).styleIdx;
    if (!colorFromTexture)
    {
        float32_t4 col = loadLineStyle(styleIdx).color;
        col.w *= localAlpha;
        return float4(col);
    }
    else
        return float4(localTextureColor, localAlpha);
}
template<>
float32_t4 calculateFinalColor<true>(const uint2 fragCoord, const float localAlpha, const uint32_t currentMainObjectIdx, float3 localTextureColor, bool colorFromTexture)
{
    float32_t4 color;
    nbl::hlsl::spirv::beginInvocationInterlockEXT();

    const uint32_t packedData = pseudoStencil[fragCoord];

    const uint32_t localQuantizedAlpha = (uint32_t)(localAlpha * 255.f);
    const uint32_t storedQuantizedAlpha = nbl::hlsl::glsl::bitfieldExtract<uint32_t>(packedData,0,AlphaBits);
    const uint32_t storedMainObjectIdx = nbl::hlsl::glsl::bitfieldExtract<uint32_t>(packedData,AlphaBits,MainObjectIdxBits);
    // if geomID has changed, we resolve the SDF alpha (draw using blend), else accumulate
    const bool differentMainObject = currentMainObjectIdx != storedMainObjectIdx; // meaning current pixel's main object is different than what is already stored
    const bool resolve = differentMainObject && storedMainObjectIdx != InvalidMainObjectIdx;
    uint32_t toResolveStyleIdx = InvalidStyleIdx;
    
    // load from colorStorage only if we want to resolve color from texture instead of style
    // sampling from colorStorage needs to happen in critical section because another fragment may also want to store into it at the same time + need to happen before store
    if (resolve)
    {
        toResolveStyleIdx = loadMainObject(storedMainObjectIdx).styleIdx;
        if (toResolveStyleIdx == InvalidStyleIdx) // if style idx to resolve is invalid, then it means we should resolve from color
            color = float32_t4(unpackR11G11B10_UNORM(colorStorage[fragCoord]), 1.0f);
    }
    
    // If current localAlpha is higher than what is already stored in pseudoStencil we will update the value in pseudoStencil or the color in colorStorage, this is equivalent to programmable blending MAX operation.
    // OR If previous pixel has a different ID than current's  (i.e. previous either empty/invalid or a differnet mainObject), we should update our alpha and color storages.
    if (differentMainObject || localQuantizedAlpha > storedQuantizedAlpha)
    {
        pseudoStencil[fragCoord] = nbl::hlsl::glsl::bitfieldInsert<uint32_t>(localQuantizedAlpha,currentMainObjectIdx,AlphaBits,MainObjectIdxBits);
        if (colorFromTexture) // writing color from texture
            colorStorage[fragCoord] = packR11G11B10_UNORM(localTextureColor);
    }
    
    nbl::hlsl::spirv::endInvocationInterlockEXT();

    if (!resolve)
        discard;

    // draw with previous geometry's style's color or stored in texture buffer :kek:
    // we don't need to load the style's color in critical section because we've already retrieved the style index from the stored main obj
    if (toResolveStyleIdx != InvalidStyleIdx) // if toResolveStyleIdx is valid then that means our resolved color should come from line style
    {
        color = loadLineStyle(toResolveStyleIdx).color;
        gammaUncorrect(color.rgb); // want to output to SRGB without gamma correction
    }
    
    color.a *= float(storedQuantizedAlpha) / 255.f;
    
    return color;
}

bool isLineValid(in nbl::hlsl::shapes::Line<float> l)
{
    bool isAnyLineComponentNaN = any(bool4(isnan(l.P0.x), isnan(l.P0.y), isnan(l.P1.x), isnan(l.P1.y)));
    if (isAnyLineComponentNaN)
        return false;
    return true;
}

[[vk::spvexecutionmode(spv::ExecutionModePixelInterlockOrderedEXT)]]
[shader("pixel")]
float4 fragMain(PSInput input) : SV_TARGET
{
    float localAlpha = 0.0f;
    float3 textureColor = float3(0, 0, 0); // color sampled from a texture

    ObjectType objType = input.getObjType();
    const uint32_t currentMainObjectIdx = input.getMainObjectIdx();
    const MainObject mainObj = loadMainObject(currentMainObjectIdx);
    
    if (pc.isDTMRendering)
    {
        DTMSettings dtmSettings = loadDTMSettings(mainObj.dtmSettingsIdx);

        float3 v[3];
        v[0] = input.getScreenSpaceVertexAttribs(0);
        v[1] = input.getScreenSpaceVertexAttribs(1);
        v[2] = input.getScreenSpaceVertexAttribs(2);

        const float3 baryCoord = dtm::calculateDTMTriangleBarycentrics(v[0].xy, v[1].xy, v[2].xy, input.position.xy);
        float height = baryCoord.x * v[0].z + baryCoord.y * v[1].z + baryCoord.z * v[2].z;
        float heightDeriv = fwidth(height);

        float4 dtmColor = float4(0.0f, 0.0f, 0.0f, 0.0f);
        
        if (dtmSettings.drawOutlineEnabled())                                                                                                    // TODO: do i need 'height' paramter here?
            dtmColor = dtm::blendUnder(dtmColor, dtm::calculateDTMOutlineColor(dtmSettings.outlineLineStyleIdx, v, input.position.xy));
        if (dtmSettings.drawContourEnabled())
        {
            for(uint32_t i = 0; i < dtmSettings.contourSettingsCount; ++i) // TODO: should reverse the order with blendUnder
                dtmColor = dtm::blendUnder(dtmColor, dtm::calculateDTMContourColor(dtmSettings.contourSettings[i], v, input.position.xy, height));
        }
        if (dtmSettings.drawHeightShadingEnabled())
            dtmColor = dtm::blendUnder(dtmColor, dtm::calculateDTMHeightColor(dtmSettings.heightShadingSettings, v, heightDeriv, input.position.xy, height));

        textureColor = dtmColor.rgb / dtmColor.a;
        localAlpha = dtmColor.a;

        // because final color is premultiplied by alpha
        textureColor = dtmColor.rgb / dtmColor.a;

        gammaUncorrect(textureColor); // want to output to SRGB without gamma correction
        return calculateFinalColor<DeviceConfigCaps::fragmentShaderPixelInterlock>(uint2(input.position.xy), localAlpha, currentMainObjectIdx, textureColor, true);
    }
    else
    {
        // figure out local alpha with sdf
        if (objType == ObjectType::LINE || objType == ObjectType::QUAD_BEZIER || objType == ObjectType::POLYLINE_CONNECTOR)
        {
            float distance = nbl::hlsl::numeric_limits<float>::max;
            if (objType == ObjectType::LINE)
            {
                const float2 start = input.getLineStart();
                const float2 end = input.getLineEnd();
                const uint32_t styleIdx = mainObj.styleIdx;
                const float thickness = input.getLineThickness();
                const float phaseShift = input.getCurrentPhaseShift();
                const float stretch = input.getPatternStretch();

                nbl::hlsl::shapes::Line<float> lineSegment = nbl::hlsl::shapes::Line<float>::construct(start, end);

                LineStyle style = loadLineStyle(styleIdx);

                if (!style.hasStipples() || stretch == InvalidStyleStretchValue)
                {
                    distance = ClippedSignedDistance< nbl::hlsl::shapes::Line<float> >::sdf(lineSegment, input.position.xy, thickness, style.isRoadStyleFlag);
                }
                else
                {
                    nbl::hlsl::shapes::Line<float>::ArcLengthCalculator arcLenCalc = nbl::hlsl::shapes::Line<float>::ArcLengthCalculator::construct(lineSegment);
                    LineStyleClipper clipper = LineStyleClipper::construct(loadLineStyle(styleIdx), lineSegment, arcLenCalc, phaseShift, stretch, globals.worldToScreenRatio);
                    distance = ClippedSignedDistance<nbl::hlsl::shapes::Line<float>, LineStyleClipper>::sdf(lineSegment, input.position.xy, thickness, style.isRoadStyleFlag, clipper);
                }
            }
            else if (objType == ObjectType::QUAD_BEZIER)
            {
                nbl::hlsl::shapes::Quadratic<float> quadratic = input.getQuadratic();
                nbl::hlsl::shapes::Quadratic<float>::ArcLengthCalculator arcLenCalc = input.getQuadraticArcLengthCalculator();

                const uint32_t styleIdx = mainObj.styleIdx;
                const float thickness = input.getLineThickness();
                const float phaseShift = input.getCurrentPhaseShift();
                const float stretch = input.getPatternStretch();

                LineStyle style = loadLineStyle(styleIdx);
                if (!style.hasStipples() || stretch == InvalidStyleStretchValue)
                {
                    distance = ClippedSignedDistance< nbl::hlsl::shapes::Quadratic<float> >::sdf(quadratic, input.position.xy, thickness, style.isRoadStyleFlag);
                }
                else
                {
                    BezierStyleClipper clipper = BezierStyleClipper::construct(loadLineStyle(styleIdx), quadratic, arcLenCalc, phaseShift, stretch, globals.worldToScreenRatio );
                    distance = ClippedSignedDistance<nbl::hlsl::shapes::Quadratic<float>, BezierStyleClipper>::sdf(quadratic, input.position.xy, thickness, style.isRoadStyleFlag, clipper);
                }
            }
            else if (objType == ObjectType::POLYLINE_CONNECTOR)
            {
                const float2 P = input.position.xy - input.getPolylineConnectorCircleCenter();
                distance = miterSDF(
                    P,
                    input.getLineThickness(),
                    input.getPolylineConnectorTrapezoidStart(),
                    input.getPolylineConnectorTrapezoidEnd(),
                    input.getPolylineConnectorTrapezoidLongBase(),
                    input.getPolylineConnectorTrapezoidShortBase());

            }
            localAlpha = 1.0f - smoothstep(-globals.antiAliasingFactor, globals.antiAliasingFactor, distance);
        }
        else if (objType == ObjectType::CURVE_BOX) 
        {
            const float minorBBoxUV = input.getMinorBBoxUV();
            const float majorBBoxUV = input.getMajorBBoxUV();

            nbl::hlsl::math::equations::Quadratic<float> curveMinMinor = input.getCurveMinMinor();
            nbl::hlsl::math::equations::Quadratic<float> curveMinMajor = input.getCurveMinMajor();
            nbl::hlsl::math::equations::Quadratic<float> curveMaxMinor = input.getCurveMaxMinor();
            nbl::hlsl::math::equations::Quadratic<float> curveMaxMajor = input.getCurveMaxMajor();

            //  TODO(Optimization): Can we ignore this majorBBoxUV clamp and rely on the t clamp that happens next? then we can pass `PrecomputedRootFinder`s instead of computing the values per pixel.
            nbl::hlsl::math::equations::Quadratic<float> minCurveEquation = nbl::hlsl::math::equations::Quadratic<float>::construct(curveMinMajor.a, curveMinMajor.b, curveMinMajor.c - clamp(majorBBoxUV, 0.0, 1.0));
            nbl::hlsl::math::equations::Quadratic<float> maxCurveEquation = nbl::hlsl::math::equations::Quadratic<float>::construct(curveMaxMajor.a, curveMaxMajor.b, curveMaxMajor.c - clamp(majorBBoxUV, 0.0, 1.0));

            const float minT = clamp(PrecomputedRootFinder<float>::construct(minCurveEquation).computeRoots(), 0.0, 1.0);
            const float minEv = curveMinMinor.evaluate(minT);

            const float maxT = clamp(PrecomputedRootFinder<float>::construct(maxCurveEquation).computeRoots(), 0.0, 1.0);
            const float maxEv = curveMaxMinor.evaluate(maxT);

            const bool insideMajor = majorBBoxUV >= 0.0 && majorBBoxUV <= 1.0;
            const bool insideMinor = minorBBoxUV >= minEv && minorBBoxUV <= maxEv;

            if (insideMinor && insideMajor)
            {
                localAlpha = 1.0;
            }
            else
            {
                // Find the true SDF of a hatch box boundary which is bounded by two curves, It requires knowing the distance from the current UV to the closest point on bounding curves and the limiting lines (in major direction)
                // We also keep track of distance vector (minor, major) to convert to screenspace distance for anti-aliasing with screenspace aaFactor
                const float InvalidT = nbl::hlsl::numeric_limits<float32_t>::max;
                const float MAX_DISTANCE_SQUARED = nbl::hlsl::numeric_limits<float32_t>::max;

                const float2 boxScreenSpaceSize = input.getCurveBoxScreenSpaceSize();


                float closestDistanceSquared = MAX_DISTANCE_SQUARED;
                const float2 pos = float2(minorBBoxUV, majorBBoxUV) * boxScreenSpaceSize;

                if (minorBBoxUV < minEv)
                {
                    // DO SDF of Min Curve
                    nbl::hlsl::shapes::Quadratic<float> minCurve = nbl::hlsl::shapes::Quadratic<float>::construct(
                        float2(curveMinMinor.a, curveMinMajor.a) * boxScreenSpaceSize,
                        float2(curveMinMinor.b, curveMinMajor.b) * boxScreenSpaceSize,
                        float2(curveMinMinor.c, curveMinMajor.c) * boxScreenSpaceSize);

                    nbl::hlsl::shapes::Quadratic<float>::Candidates candidates = minCurve.getClosestCandidates(pos);
                    [[unroll(nbl::hlsl::shapes::Quadratic<float>::MaxCandidates)]]
                    for (uint32_t i = 0; i < nbl::hlsl::shapes::Quadratic<float>::MaxCandidates; i++)
                    {
                        candidates[i] = clamp(candidates[i], 0.0, 1.0);
                        const float2 distVector = minCurve.evaluate(candidates[i]) - pos;
                        const float candidateDistanceSquared = dot(distVector, distVector);
                        if (candidateDistanceSquared < closestDistanceSquared)
                            closestDistanceSquared = candidateDistanceSquared;
                    }
                }
                else if (minorBBoxUV > maxEv)
                {
                    // Do SDF of Max Curve
                    nbl::hlsl::shapes::Quadratic<float> maxCurve = nbl::hlsl::shapes::Quadratic<float>::construct(
                        float2(curveMaxMinor.a, curveMaxMajor.a) * boxScreenSpaceSize,
                        float2(curveMaxMinor.b, curveMaxMajor.b) * boxScreenSpaceSize,
                        float2(curveMaxMinor.c, curveMaxMajor.c) * boxScreenSpaceSize);
                    nbl::hlsl::shapes::Quadratic<float>::Candidates candidates = maxCurve.getClosestCandidates(pos);
                    [[unroll(nbl::hlsl::shapes::Quadratic<float>::MaxCandidates)]]
                    for (uint32_t i = 0; i < nbl::hlsl::shapes::Quadratic<float>::MaxCandidates; i++)
                    {
                        candidates[i] = clamp(candidates[i], 0.0, 1.0);
                        const float2 distVector = maxCurve.evaluate(candidates[i]) - pos;
                        const float candidateDistanceSquared = dot(distVector, distVector);
                        if (candidateDistanceSquared < closestDistanceSquared)
                            closestDistanceSquared = candidateDistanceSquared;
                    }
                }

                if (!insideMajor)
                {
                    const bool minLessThanMax = minEv < maxEv;
                    float2 majorDistVector = float2(MAX_DISTANCE_SQUARED, MAX_DISTANCE_SQUARED);
                    if (majorBBoxUV > 1.0)
                    {
                        const float2 minCurveEnd = float2(minEv, 1.0) * boxScreenSpaceSize;
                        if (minLessThanMax)
                            majorDistVector = sdLineDstVec(pos, minCurveEnd, float2(maxEv, 1.0) * boxScreenSpaceSize);
                        else
                            majorDistVector = pos - minCurveEnd;
                    }
                    else
                    {
                        const float2 minCurveStart = float2(minEv, 0.0) * boxScreenSpaceSize;
                        if (minLessThanMax)
                            majorDistVector = sdLineDstVec(pos, minCurveStart, float2(maxEv, 0.0) * boxScreenSpaceSize);
                        else
                            majorDistVector = pos - minCurveStart;
                    }

                    const float majorDistSq = dot(majorDistVector, majorDistVector);
                    if (majorDistSq < closestDistanceSquared)
                        closestDistanceSquared = majorDistSq;
                }

                const float dist = sqrt(closestDistanceSquared);
                localAlpha = 1.0f - smoothstep(0.0, globals.antiAliasingFactor, dist);
            }

            LineStyle style = loadLineStyle(mainObj.styleIdx);
            uint32_t textureId = asuint(style.screenSpaceLineWidth);
            if (textureId != InvalidTextureIndex)
            {
                // For Hatch fiils we sample the first mip as we don't fill the others, because they are constant in screenspace and render as expected
                // If later on we decided that we can have different sizes here, we should do computations similar to FONT_GLYPH
                float3 msdfSample = msdfTextures.SampleLevel(msdfSampler, float3(frac(input.position.xy / HatchFillMSDFSceenSpaceSize), float(textureId)), 0.0).xyz;
                float msdf = nbl::hlsl::text::msdfDistance(msdfSample, MSDFPixelRange * HatchFillMSDFSceenSpaceSize / MSDFSize);
                localAlpha *= 1.0f - smoothstep(-globals.antiAliasingFactor / 2.0f, globals.antiAliasingFactor / 2.0f, msdf);
            }
        }
        else if (objType == ObjectType::FONT_GLYPH) 
        {
            const float2 uv = input.getFontGlyphUV();
            const uint32_t textureId = input.getFontGlyphTextureId();

            if (textureId != InvalidTextureIndex)
            {
                float mipLevel = msdfTextures.CalculateLevelOfDetail(msdfSampler, uv);
                float3 msdfSample = msdfTextures.SampleLevel(msdfSampler, float3(uv, float(textureId)), mipLevel);
                float msdf = nbl::hlsl::text::msdfDistance(msdfSample, input.getFontGlyphPxRange());
                /*
                    explaining "*= exp2(max(mipLevel,0.0))"
                    Each mip level has constant MSDFPixelRange
                    Which essentially makes the msdfSamples here (Harware Sampled) have different scales per mip
                    As we go up 1 mip level, the msdf distance should be multiplied by 2.0
                    While this makes total sense for NEAREST mip sampling when mipLevel is an integer and only one mip is being sampled.
                    It's a bit complex when it comes to trilinear filtering (LINEAR mip sampling), but it works in practice!
                
                    Alternatively you can think of it as doing this instead:
                    localAlpha = smoothstep(+globals.antiAliasingFactor / exp2(max(mipLevel,0.0)), 0.0, msdf);
                    Which is reducing the aa feathering as we go up the mip levels. 
                    to avoid aa feathering of the MAX_MSDF_DISTANCE_VALUE to be less than aa factor and eventually color it and cause greyed out area around the main glyph
                */
                msdf *= exp2(max(mipLevel,0.0));
            
                LineStyle style = loadLineStyle(mainObj.styleIdx);
                const float screenPxRange = input.getFontGlyphPxRange() / MSDFPixelRangeHalf;
                const float bolden = style.worldSpaceLineWidth * screenPxRange; // worldSpaceLineWidth is actually boldenInPixels, aliased TextStyle with LineStyle
                localAlpha = 1.0f - smoothstep(-globals.antiAliasingFactor / 2.0f + bolden, globals.antiAliasingFactor / 2.0f + bolden, msdf);
            }
        }
        else if (objType == ObjectType::STATIC_IMAGE) 
        {
            const float2 uv = input.getImageUV();
            const uint32_t textureId = input.getImageTextureId();

            if (textureId != InvalidTextureIndex)
            {
                float4 colorSample = textures[NonUniformResourceIndex(textureId)].Sample(textureSampler, float2(uv.x, uv.y));
                textureColor = colorSample.rgb;
                localAlpha = colorSample.a;
            }
        }
        else if (objType == ObjectType::GRID_DTM)
        {
            // NOTE: create and read from a texture as a last step, you can generate the height values procedurally from a function while you're working on the sdf stuff.
            
            // Query dtm settings
            // use texture Gather to get 4 corners: https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-to-gather
            // DONE (but needs to be fixed): A. the outlines can be stippled, use phaseshift of the line such that they started from the grid's origin worldspace coordinate
            // DONE: B. the contours are computed for triangles, use the same function as for dtms, choose between the two triangles based on local UV coords in current cell
                // DONE: Make it so we can choose which diagonal to use to construct the triangle, it's either u=v or u=1-v
            // DONE: C. Height shading same as contours (split into two triangles)

            // DONE (but needs to be tested after i implement texture height maps) Heights can have invalid values (let's say NaN) if a cell corner has NaN value then no triangle (for contour and shading) and no outline should include that corner. (see DTM image in discord with gaps)
            
            // TODO: we need to emulate dilation and do sdf of neighbouring cells as well. because contours, outlines and shading can bleed into other cells for AA.
            // [NOTE] Do dilation as last step, when everything else works fine

            DTMSettings dtmSettings = loadDTMSettings(mainObj.dtmSettingsIdx);

            if (!dtmSettings.drawContourEnabled() && !dtmSettings.drawOutlineEnabled() && !dtmSettings.drawHeightShadingEnabled())
                discard;

            float2 pos = input.getGridDTMScreenSpacePosition();
            float2 uv = input.getImageUV();
            const uint32_t textureId = input.getGridDTMHeightTextureID();

            float2 topLeft = input.getGridDTMScreenSpaceTopLeft();
            float2 gridExtents = input.getGridDTMScreenSpaceGridExtents();
            const float cellWidth = input.getGridDTMScreenSpaceCellWidth();

            float2 gridSpacePos = uv * gridExtents;
            float2 cellCoords;
            {
                float2 gridSpacePosDivGridCellWidth = gridSpacePos / cellWidth;
                cellCoords.x = int32_t(gridSpacePosDivGridCellWidth.x);
                cellCoords.y = int32_t(gridSpacePosDivGridCellWidth.y);
            }

            float2 gridSpaceCellTopLeftCoords = cellCoords * cellWidth;

            // grid consists of square cells and cells are divided into two triangles:
            // depending on mode it is
            // either:        or:
            // v2a-------v1   v0-------v2b
            // |  A     / |   | \     B  |
            // |     /    |   |    \     |
            // |  /  B    |   |   A   \  |
            // v0-------v2b   v2a-------v1
            // 

            const bool gridOnly = textureId == InvalidTextureIndex && dtmSettings.drawOutlineEnabled();
            if (gridOnly)
            {
                nbl::hlsl::shapes::Line<float> outlineLineSegments[2];
                
                const float halfCellWidth = cellWidth * 0.5f;
                const float2 horizontalBounds = float2(topLeft.y, topLeft.y + gridExtents.y);
                const float2 verticalBounds = float2(topLeft.x, topLeft.x + gridExtents.x);
                float2 nearestLineRemainingCoords = int2((gridSpacePos + halfCellWidth) / cellWidth) * cellWidth + topLeft;
                // shift lines outside of the grid to a bound
                nearestLineRemainingCoords.x = clamp(nearestLineRemainingCoords.x, verticalBounds.x, verticalBounds.y);
                nearestLineRemainingCoords.y = clamp(nearestLineRemainingCoords.y, horizontalBounds.x, horizontalBounds.y);

                // find the nearest horizontal line
                outlineLineSegments[0].P0 = float32_t2(verticalBounds.x, nearestLineRemainingCoords.y);
                outlineLineSegments[0].P1 = float32_t2(verticalBounds.y, nearestLineRemainingCoords.y);
                // find the nearest vertical line
                outlineLineSegments[1].P0 = float32_t2(nearestLineRemainingCoords.x, horizontalBounds.x);
                outlineLineSegments[1].P1 = float32_t2(nearestLineRemainingCoords.x, horizontalBounds.y);
                
                float4 dtmColor = dtm::calculateGridDTMOutlineColor(dtmSettings.outlineLineStyleIdx, outlineLineSegments, input.position.xy, 0.0f);
                textureColor = dtmColor.rgb;
                localAlpha = dtmColor.a;
            }
            else
            {
            
                // calculate insideCellCoord and figure out the 4 cells we're gonna do sdf with
                float2 insideCellCoord = gridSpacePos - float2(cellWidth, cellWidth) * cellCoords; // TODO: use fmod instead?
                // 0.2, 0.1 --> 0, 0 ---> [0, 0], [-1, -1], [-1, 0], [0, -1]
                float offsetX = round(insideCellCoord.x) - 1.0f;
                float offsetY = round(insideCellCoord.y) - 1.0f;
                
                // for each of those cells (some might be out of bounds, so we skip)
                    // gather 
                    // then figure out their triangles (A and B) and fill array of max 8 triangles (dtm::GridDTMTriangle)

                // Contours:
                // Is Contours Enabled?
                    // for each contour settings (in reverse)
                        // float sdf = max;
                        // for each triangle
                            // sdf = min(sdf, sdfOfContourSettings[i]);
                        // based on sdf, the contour line style + smoothstep: we compute color and alpha
                        // blendUnder

                // Outlines:
                // Is Outlines Enabled?
                    // float sdf = max;
                    // for each triangle
                        // sdf = min(sdf, sdfOfOutlineSetting);
                    // based on sdf, the outline line style + smoothstep: we compute color and alpha
                    // blendUnder
                
                // Height Shading:
                    // We just do sdf with current triangle (if valid)
            }
            
#if 0
            // calculate screen space coordinates of vertices of the current tiranlge within the grid
            dtm::GridDTMTriangle currentTriangle;
            dtm::GridDTMCell neighbouringCells[8];
            if (dtmSettings.drawContourEnabled() || dtmSettings.drawHeightShadingEnabled())
            {
                if (textureId == InvalidTextureIndex)
                    discard;

                // heightData.heihts.x - bottom left texel
                // heightData.heihts.y - bottom right texel
                // heightData.heihts.z - top right texel
                // heightData.heihts.w - top left texel
                dtm::GridDTMHeightMapData heightData = dtm::retrieveGridDTMCellDataFromHeightMap(gridExtents, cellCoords, cellWidth, texturesU32[NonUniformResourceIndex(textureId)]);
                if (heightData.cellDiagonal == E_CELL_DIAGONAL::INVALID)
                    discard;

                const bool diagonalFromTopLeftToBottomRight = heightData.cellDiagonal == E_CELL_DIAGONAL::TOP_LEFT_TO_BOTTOM_RIGHT;

                float2 insideCellCoord = gridSpacePos - float2(cellWidth, cellWidth) * cellCoords; // TODO: use fmod instead?
                // my ASCII art above explains which triangle is A and which is B
                const bool triangleA = diagonalFromTopLeftToBottomRight ?
                    insideCellCoord.x < insideCellCoord.y :
                    insideCellCoord.x < cellWidth - insideCellCoord.y;

                if (diagonalFromTopLeftToBottomRight)
                {
                    currentTriangle.vertices[0] = float3(gridSpaceCellTopLeftCoords.x, gridSpaceCellTopLeftCoords.y, heightData.heights.w);
                    currentTriangle.vertices[1] = float3(gridSpaceCellTopLeftCoords.x + cellWidth, gridSpaceCellTopLeftCoords.y + cellWidth, heightData.heights.y);
                    currentTriangle.vertices[2] = triangleA ? float3(gridSpaceCellTopLeftCoords.x, gridSpaceCellTopLeftCoords.y + cellWidth, heightData.heights.x) : float3(gridSpaceCellTopLeftCoords.x + cellWidth, gridSpaceCellTopLeftCoords.y, heightData.heights.z);

                    // TODO: use cell space instead https://github.com/Devsh-Graphics-Programming/Nabla-Examples-and-Tests/pull/186#discussion_r2133699055
                    //currentTriangle.vertices[0] = float3(0.0f, 0.0f, heightData.heights.w);
                    //currentTriangle.vertices[1] = float3(cellWidth, cellWidth, heightData.heights.y);
                    //currentTriangle.vertices[2] = triangleA ? float3(0.0f, cellWidth, heightData.heights.x) : float3(cellWidth, 0.0f, heightData.heights.z);
                }
                else
                {
                    currentTriangle.vertices[0] = float3(gridSpaceCellTopLeftCoords.x, gridSpaceCellTopLeftCoords.y + cellWidth, heightData.heights.x);
                    currentTriangle.vertices[1] = float3(gridSpaceCellTopLeftCoords.x + cellWidth, gridSpaceCellTopLeftCoords.y, heightData.heights.z);
                    currentTriangle.vertices[2] = triangleA ? float3(gridSpaceCellTopLeftCoords.x, gridSpaceCellTopLeftCoords.y, heightData.heights.w) : float3(gridSpaceCellTopLeftCoords.x + cellWidth, gridSpaceCellTopLeftCoords.y + cellWidth, heightData.heights.y);

                    // TODO: use cell space instead https://github.com/Devsh-Graphics-Programming/Nabla-Examples-and-Tests/pull/186#discussion_r2133699055
                    //currentTriangle.vertices[0] = float3(0.0f, 0.0f + cellWidth, heightData.heights.x);
                    //currentTriangle.vertices[1] = float3(0.0f + cellWidth, 0.0f, heightData.heights.z);
                    //currentTriangle.vertices[2] = triangleA ? float3(0.0f, 0.0f, heightData.heights.w) : float3(cellWidth, cellWidth, heightData.heights.y);
                }

                bool isTriangleInvalid = isnan(currentTriangle.vertices[0].z) || isnan(currentTriangle.vertices[1].z) || isnan(currentTriangle.vertices[2].z);
                bool isCellPartiallyInvalid = isnan(heightData.heights.x) || isnan(heightData.heights.y) || isnan(heightData.heights.z) || isnan(heightData.heights.w);

                if (isTriangleInvalid)
                    discard;

                // move from grid space to screen space
                [unroll]
                for (int i = 0; i < 3; ++i)
                    currentTriangle.vertices[i].xy += topLeft;

                const float2 neighbouringCellsCellOffsets[8] = {
                    float2(-1.0f, -1.0f),
                    float2(0.0f, -1.0f),
                    float2(1.0f, -1.0f),
                    float2(-1.0f, 0.0f),
                    float2(-1.0f, 0.0f),
                    float2(-1.0f, 1.0f),
                    float2(0.0f, 1.0f),
                    float2(1.0f, 1.0f)
                };

                // construct triangles of neighbouring cells
                for (int i = 0; i < 8; ++i)
                {
                    float2 neighbouringCellCoords = cellCoords + neighbouringCellsCellOffsets[i];
                    neighbouringCells[i] = dtm::calculateCellTriangles(topLeft, gridExtents, neighbouringCellCoords, cellWidth, texturesU32[NonUniformResourceIndex(textureId)]);
                }
            }

            const float3 baryCoord = dtm::calculateDTMTriangleBarycentrics(currentTriangle.vertices[0].xy, currentTriangle.vertices[1].xy, currentTriangle.vertices[2].xy, input.position.xy);
            float height = baryCoord.x * currentTriangle.vertices[0].z + baryCoord.y * currentTriangle.vertices[1].z + baryCoord.z * currentTriangle.vertices[2].z;
            float heightDeriv = fwidth(height);

            const bool outOfBoundsUV = uv.x < 0.0f || uv.y < 0.0f || uv.x > 1.0f || uv.y > 1.0f;
            float4 dtmColor = float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (dtmSettings.drawContourEnabled() && !outOfBoundsUV)
            {
                for (int i = dtmSettings.contourSettingsCount-1u; i >= 0; --i) 
                    dtmColor = dtm::blendUnder(dtmColor, dtm::calculateDTMContourColor(dtmSettings.contourSettings[i], currentTriangle.vertices, input.position.xy, height));

                // draw shit form neighbouring cells
                for (int i = 0; i < 8; ++i)
                {
                    for (int j = dtmSettings.contourSettingsCount - 1u; j >= 0; --j)
                    {
                        dtmColor = dtm::blendUnder(dtmColor, dtm::calculateDTMContourColor(dtmSettings.contourSettings[i], neighbouringCells[i].triangleA.vertices, input.position.xy, height));
                        dtmColor = dtm::blendUnder(dtmColor, dtm::calculateDTMContourColor(dtmSettings.contourSettings[i], neighbouringCells[i].triangleB.vertices, input.position.xy, height));
                    }
                }
            }
            if (dtmSettings.drawOutlineEnabled())
                dtmColor = dtm::blendUnder(dtmColor, dtm::calculateGridDTMOutlineColor(dtmSettings.outlineLineStyleIdx, outlineLineSegments, input.position.xy, 0.0f));
            if (dtmSettings.drawHeightShadingEnabled() && !outOfBoundsUV)
                dtmColor = dtm::blendUnder(dtmColor, dtm::calculateDTMHeightColor(dtmSettings.heightShadingSettings, currentTriangle.vertices, heightDeriv, input.position.xy, height));

            textureColor = dtmColor.rgb / dtmColor.a;
            localAlpha = dtmColor.a;

            // because final color is premultiplied by alpha
            textureColor = dtmColor.rgb / dtmColor.a;

            // test out of bounds draw
            /*if (outOfBoundsUV)
                textureColor = float3(0.0f, 1.0f, 0.0f);
            else
                textureColor = float3(0.0f, 0.0f, 1.0f);

            localAlpha = 0.5f;*/
#endif
        }
        else if (objType == ObjectType::STREAMED_IMAGE) 
        {
            const float2 uv = input.getImageUV();
            const uint32_t textureId = input.getImageTextureId();

            if (textureId != InvalidTextureIndex)
            {
                float4 colorSample = textures[NonUniformResourceIndex(textureId)].Sample(textureSampler, float2(uv.x, uv.y));
                textureColor = colorSample.rgb;
                localAlpha = colorSample.a;
            }
        }

        
        if (localAlpha <= 0)
            discard;
        
        uint2 fragCoord = uint2(input.position.xy);
        const bool colorFromTexture = objType == ObjectType::STREAMED_IMAGE || objType == ObjectType::STATIC_IMAGE || objType == ObjectType::GRID_DTM;

        return calculateFinalColor<DeviceConfigCaps::fragmentShaderPixelInterlock>(fragCoord, localAlpha, currentMainObjectIdx, textureColor, colorFromTexture);
    }
}
