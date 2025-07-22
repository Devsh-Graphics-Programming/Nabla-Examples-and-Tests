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
// Trapezoid centered around origin (0,0), the top edge has length r2, the bottom edge has length r1, the height of the trapezoid is he*2.0
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

/*
                    XXXXXXX b XXXXXX              Long Base (len = rb)
                   X                X            
                  X                 X            
                 X                   X           
                X    XXXXXXXXXXX      X          
               X XXXX     |     XXXX  X          
              XXX         |         XXXX         
            XX            |            XX        
           XX             |             XX       
          XX              |              XX      
         XX               T Trapz Center XX      (2) p.y = 0 after p.y = p.y - halfHeight + radius
        XX                |               XX     
       X X                C Circle Center  X     (1) p = (0,0) at circle center
      X  X                |                X     
     X    X               |               X X    
    X     X               |               X  X   
   X       X              |              X   X   
  X         XX            |            XX     X  
 X            XXX         |         XXX        X 
X                XXXX     |     XXXX           X 
XXXXXXXXXXXXXXXXXXXXXXXXX a XXXXXXXXXXXXXXXXXXXXX Short Base (len = ra)
*/
// p is in circle's space (the circle centered at line intersection and radius = thickness)
// a and b are points at each trapezoid base (short and long base)
// TODO[Optimization] we can probably send less info, since we only use length of b-a and the normalize vector
float miterSDF(float2 p, float thickness, float2 a, float2 b, float ra, float rb)
{
    float halfHeight = length(b - a) / 2.0;
    float2 d = normalize(b - a);
    float2x2 rot = float2x2(d.y, -d.x, d.x, d.y);
    p = mul(rot, p); // rotate(change of basis) such that the point is now in the space where trapezoid is y-axis aligned, see (1) above 
    p.y = p.y - halfHeight + thickness; // see (2) above
    return sdTrapezoid(p, ra, rb, halfHeight);
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
    float worldToScreenRatio = input.getCurrentWorldToScreenRatio();
    
    if (pc.isDTMRendering)
    {
        DTMSettings dtmSettings = loadDTMSettings(mainObj.dtmSettingsIdx);

        float3 triangleVertices[3];
        triangleVertices[0] = input.getScreenSpaceVertexAttribs(0);
        triangleVertices[1] = input.getScreenSpaceVertexAttribs(1);
        triangleVertices[2] = input.getScreenSpaceVertexAttribs(2);

        const float3 baryCoord = dtm::calculateDTMTriangleBarycentrics(triangleVertices[0].xy, triangleVertices[1].xy, triangleVertices[2].xy, input.position.xy);

        float height = baryCoord.x * triangleVertices[0].z + baryCoord.y * triangleVertices[1].z + baryCoord.z * triangleVertices[2].z;
        float heightDeriv = fwidth(height);

        float4 dtmColor = float4(0.0f, 0.0f, 0.0f, 0.0f);
        
        if (dtmSettings.drawOutlineEnabled())                                                                                                    // TODO: do i need 'height' paramter here?
            dtmColor = dtm::blendUnder(dtmColor, dtm::calculateDTMOutlineColor(dtmSettings.outlineLineStyleIdx, worldToScreenRatio, triangleVertices, input.position.xy));
        if (dtmSettings.drawContourEnabled())
        {
            for(uint32_t i = 0; i < dtmSettings.contourSettingsCount; ++i) // TODO: should reverse the order with blendUnder
            {
                LineStyle contourStyle = loadLineStyle(dtmSettings.contourSettings[i].contourLineStyleIdx);
                float sdf = dtm::calculateDTMContourSDF(dtmSettings.contourSettings[i], contourStyle, worldToScreenRatio, triangleVertices, input.position.xy, height);
                float4 contourColor = contourStyle.color;
                contourColor.a *= 1.0f - smoothstep(-globals.antiAliasingFactor, globals.antiAliasingFactor, sdf);
                dtmColor = dtm::blendUnder(dtmColor, contourColor);
            }
        }
        if (dtmSettings.drawHeightShadingEnabled())
            dtmColor = dtm::blendUnder(dtmColor, dtm::calculateDTMHeightColor(dtmSettings.heightShadingSettings, triangleVertices, heightDeriv, input.position.xy, height));

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
                    LineStyleClipper clipper = LineStyleClipper::construct(loadLineStyle(styleIdx), lineSegment, arcLenCalc, phaseShift, stretch, worldToScreenRatio);
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
                    BezierStyleClipper clipper = BezierStyleClipper::construct(loadLineStyle(styleIdx), quadratic, arcLenCalc, phaseShift, stretch, worldToScreenRatio );
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
            DTMSettings dtmSettings = loadDTMSettings(mainObj.dtmSettingsIdx);

            if (!dtmSettings.drawContourEnabled() && !dtmSettings.drawOutlineEnabled() && !dtmSettings.drawHeightShadingEnabled())
                discard;

            float2 uv = input.getImageUV();
            const uint32_t textureId = input.getGridDTMHeightTextureID();

            float2 gridExtents = input.getGridDTMScreenSpaceGridExtents();
            const float cellWidth = input.getGridDTMScreenSpaceCellWidth();
            // TODO: I think we can get it from the height map size if texture is valid?!, better if it comes directly from CPU side, vertex shader or something, division + round to integer is error-prone for large integer values
            float2 gridDimensions = round(gridExtents / cellWidth); // texturesU32[NonUniformResourceIndex(textureId)].GetDimensions()? 

            float2 gridSpacePos = uv * gridExtents;
            float2 gridSpacePosDivGridCellWidth = gridSpacePos / cellWidth;
            float2 currentCellCoord;
            {
                currentCellCoord.x = floor(gridSpacePosDivGridCellWidth.x);
                currentCellCoord.y = floor(gridSpacePosDivGridCellWidth.y);
            }

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
                const float2 horizontalBounds = float2(0.0f, gridExtents.y);
                const float2 verticalBounds = float2(0.0f, gridExtents.x);
                float2 nearestLineRemainingCoords = int2((gridSpacePos + halfCellWidth) / cellWidth) * cellWidth;
                // shift lines outside of the grid to a bound
                nearestLineRemainingCoords.x = clamp(nearestLineRemainingCoords.x, verticalBounds.x, verticalBounds.y);
                nearestLineRemainingCoords.y = clamp(nearestLineRemainingCoords.y, horizontalBounds.x, horizontalBounds.y);

                // find the nearest horizontal line
                outlineLineSegments[0].P0 = float32_t2(verticalBounds.x, nearestLineRemainingCoords.y);
                outlineLineSegments[0].P1 = float32_t2(verticalBounds.y, nearestLineRemainingCoords.y);
                // find the nearest vertical line
                outlineLineSegments[1].P0 = float32_t2(nearestLineRemainingCoords.x, horizontalBounds.x);
                outlineLineSegments[1].P1 = float32_t2(nearestLineRemainingCoords.x, horizontalBounds.y);
                
                LineStyle outlineStyle = loadLineStyle(dtmSettings.outlineLineStyleIdx);
                float sdf = dtm::calculateLineSDF(outlineStyle, worldToScreenRatio, outlineLineSegments[0], gridSpacePos, 0.0f);
                sdf = min(sdf, dtm::calculateLineSDF(outlineStyle, worldToScreenRatio, outlineLineSegments[1], gridSpacePos, 0.0f));

                float4 dtmColor = outlineStyle.color;
                dtmColor.a *= 1.0f - smoothstep(-globals.antiAliasingFactor, globals.antiAliasingFactor, sdf);
                
                textureColor = dtmColor.rgb;
                localAlpha = dtmColor.a;
            }
            else
            {
                // calculate localUV and figure out the 4 cells we're gonna do sdf with
                float2 localUV = gridSpacePosDivGridCellWidth - currentCellCoord; // TODO: use fmod instead?
                int2 roundedLocalUV = round(localUV);
                float2 offset = roundedLocalUV * 2.0f - 1.0f;

                // Triangles
                const uint32_t MaxTrianglesToDoSDFWith = 8u;
                dtm::GridDTMTriangle triangles[MaxTrianglesToDoSDFWith];
                float interpolatedHeights[MaxTrianglesToDoSDFWith]; // these are height based on barycentric interpolation of current pixel with all the triangles above
                uint32_t triangleCount = 0u;
                
                // We can do sdf for up to 4 maximum lines for the outlines, 2 belong to the current cell and the other 2 belong to the opposite neighbouring cell
                /* Example:
                          |                  
                          |     opposite cell
                          |                  
                    ------+------            
                          |                  
        current cell      |                  
                          |                  
                          
                   `+` is the current corner and we draw the 4 lines leading up to it.
                */
                
                // curr cell horizontal, curr cell vertical, opposite cell horizontal, opposite cell vertical 
                bool4 linesValidity = bool4(false, false, false, false);

                [unroll]
                for (int i = 0; i < 2; ++i)
                {
                    for (int j = 0; j < 2; ++j)
                    {
                        float2 cellCoord = currentCellCoord + float2(i, j) * offset;
                        const bool isCellWithinRange = 
                            cellCoord.x >= 0.0f && cellCoord.y >= 0.0f && 
                            cellCoord.x < gridDimensions.x && cellCoord.y < gridDimensions.y;
                        if (isCellWithinRange)
                        {
                            dtm::GridDTMHeightMapData heightData = dtm::retrieveGridDTMCellDataFromHeightMap(gridDimensions, cellCoord, texturesU32[NonUniformResourceIndex(textureId)]);
                            dtm::GridDTMCell gridCellFormed = dtm::calculateCellTriangles(heightData, cellCoord, cellWidth);
                            if (gridCellFormed.validA)
                                triangles[triangleCount++] = gridCellFormed.triangleA;
                            if (gridCellFormed.validB)
                                triangles[triangleCount++] = gridCellFormed.triangleB;

                            // we just need to check and set lines validity
                            // Formulas to get current cell's horizontal and vertical lines validity
                            // All this to avoid extra texel fetch to check validity and use the Gather result instead :D
                            // TODO: Only 0,0 and 1,1 is enough to check if cells are valid, but other checks required in case current cell is invalid (out of bounds) but it's line is valid
                            if (i == 0 && j == 0)
                            {
                                // current cell's line validity
                                linesValidity[0] = !isInvalidGridDtmHeightValue(heightData.heights[2 - (roundedLocalUV.y * 2)]) && !isInvalidGridDtmHeightValue(heightData.heights[3 - (roundedLocalUV.y * 2)]);
                                linesValidity[1] = !isInvalidGridDtmHeightValue(heightData.heights[roundedLocalUV.x ^ 0]) && !isInvalidGridDtmHeightValue(heightData.heights[roundedLocalUV.x ^ 3]);
                            }
                            if (i == 1 && j == 0)
                            {
                                linesValidity[1] = !isInvalidGridDtmHeightValue(heightData.heights[roundedLocalUV.x ^ 1]) && !isInvalidGridDtmHeightValue(heightData.heights[roundedLocalUV.x ^ 2]);
                                linesValidity[2] = !isInvalidGridDtmHeightValue(heightData.heights[2 - (roundedLocalUV.y * 2)]) && !isInvalidGridDtmHeightValue(heightData.heights[3 - (roundedLocalUV.y * 2)]);;
                            }
                            if (i == 0 && j == 1)
                            {
                                linesValidity[0] = !isInvalidGridDtmHeightValue(heightData.heights[roundedLocalUV.y * 2]) && !isInvalidGridDtmHeightValue(heightData.heights[roundedLocalUV.y * 2 + 1]);
                                linesValidity[3] = !isInvalidGridDtmHeightValue(heightData.heights[roundedLocalUV.x ^ 0]) && !isInvalidGridDtmHeightValue(heightData.heights[roundedLocalUV.x ^ 3]);
                            }
                            if (i == 1 && j == 1)
                            {
                                linesValidity[2] = !isInvalidGridDtmHeightValue(heightData.heights[roundedLocalUV.y * 2]) && !isInvalidGridDtmHeightValue(heightData.heights[roundedLocalUV.y * 2 + 1]);
                                linesValidity[3] = !isInvalidGridDtmHeightValue(heightData.heights[roundedLocalUV.x ^ 1]) && !isInvalidGridDtmHeightValue(heightData.heights[roundedLocalUV.x ^ 2]);
                            }
                        }
                    }
                }
                
                const uint32_t InvalidTriangleIndex = nbl::hlsl::numeric_limits<uint32_t>::max;
                uint32_t currentTriangleIndex = InvalidTriangleIndex;
                // For height shading, merge this loop with the previous one, because baryCoord all positive means point inside triangle and we can use that to figure out the triangle we want to do height shading for.
                for (int t = 0; t < triangleCount; ++t)
                {
                    dtm::GridDTMTriangle tri = triangles[t];
                    const float3 baryCoord = dtm::calculateDTMTriangleBarycentrics(tri.vertices[0].xy, tri.vertices[1].xy, tri.vertices[2].xy, gridSpacePos);
                    interpolatedHeights[t] = baryCoord.x * tri.vertices[0].z + baryCoord.y * tri.vertices[1].z + baryCoord.z * tri.vertices[2].z;

                    if (currentTriangleIndex == InvalidTriangleIndex)
                    {
                        const float minValue = 0.0f - nbl::hlsl::numeric_limits<float>::epsilon;
                        const float maxValue = 1.0f + nbl::hlsl::numeric_limits<float>::epsilon;
                        if (all(baryCoord >= minValue) && all(baryCoord <= maxValue))
                            currentTriangleIndex = t;
                    }
                }

                float4 dtmColor = float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (dtmSettings.drawContourEnabled())
                {
                    for (int i = dtmSettings.contourSettingsCount-1u; i >= 0; --i) 
                    {
                        LineStyle contourStyle = loadLineStyle(dtmSettings.contourSettings[i].contourLineStyleIdx);
                        float sdf = nbl::hlsl::numeric_limits<float>::max;
                        for (int t = 0; t < triangleCount; ++t)
                        {
                            const dtm::GridDTMTriangle tri = triangles[t];
                            const float currentInterpolatedHeight = interpolatedHeights[t];
                            sdf = min(sdf, dtm::calculateDTMContourSDF(dtmSettings.contourSettings[i], contourStyle, worldToScreenRatio, tri.vertices, gridSpacePos, currentInterpolatedHeight));
                        }
                        
                        float4 contourColor = contourStyle.color; contourColor.a = 0.5f;
                        contourColor.a *= 1.0f - smoothstep(-globals.antiAliasingFactor, globals.antiAliasingFactor, sdf);
                        dtmColor = dtm::blendUnder(dtmColor, contourColor);
                    }
                }

                if (dtmSettings.drawOutlineEnabled())
                {
                    float sdf = nbl::hlsl::numeric_limits<float>::max;
                    LineStyle outlineStyle = loadLineStyle(dtmSettings.outlineLineStyleIdx);
                    nbl::hlsl::shapes::Line<float> lineSegment;

                    // Doing SDF of outlines as if cooridnate system is centered around the nearest corner of the cell
                    float2 localCellSpaceOrigin = (currentCellCoord + float2(roundedLocalUV)) * cellWidth; // in local cell space, origin
                    float2 localGridTopLeftCorner = -localCellSpaceOrigin; // top left in local cell space: topLeft is (0, 0) implicitly
                    float2 localFragPos = gridSpacePos - localCellSpaceOrigin; // we compute the current fragment pos, in local cell space
                    
                    float phaseShift = 0.0f;
                    const bool hasStipples = outlineStyle.hasStipples();
                    const float rcpPattenLenScreenSpace = outlineStyle.reciprocalStipplePatternLen * worldToScreenRatio;
                    // Drawing the lines that form a plus sign around the current corner:
                    if (linesValidity[0])
                    {
                        // this cells horizontal line
                        lineSegment.P0 = float2((offset.x > 0) ? -offset.x * cellWidth : 0.0f, 0.0f);
                        lineSegment.P1 = float2((offset.x < 0) ? -offset.x * cellWidth : 0.0f, 0.0f);
                        phaseShift = fract((lineSegment.P0.x - localGridTopLeftCorner.x) * rcpPattenLenScreenSpace);
                        sdf = min(sdf, dtm::calculateLineSDF(outlineStyle, worldToScreenRatio, lineSegment, localFragPos, phaseShift));
                    }
                    if (linesValidity[1])
                    {
                        // this cells vertical line
                        lineSegment.P0 = float2(0.0f, (offset.y > 0) ? -offset.y * cellWidth : 0.0f);
                        lineSegment.P1 = float2(0.0f, (offset.y < 0) ? -offset.y * cellWidth : 0.0f);
                        phaseShift = fract((lineSegment.P0.y - localGridTopLeftCorner.y) * rcpPattenLenScreenSpace);
                        sdf = min(sdf, dtm::calculateLineSDF(outlineStyle, worldToScreenRatio, lineSegment, localFragPos, phaseShift));
                    }
                    if (linesValidity[2])
                    {
                        // opposite cell horizontal line
                        lineSegment.P0 = float2((offset.x < 0) ? offset.x * cellWidth : 0.0f, 0.0f);
                        lineSegment.P1 = float2((offset.x > 0) ? offset.x * cellWidth : 0.0f, 0.0f);
                        phaseShift = fract((lineSegment.P0.x - localGridTopLeftCorner.x) * rcpPattenLenScreenSpace);
                        sdf = min(sdf, dtm::calculateLineSDF(outlineStyle, worldToScreenRatio, lineSegment, localFragPos, phaseShift));
                    }
                    if (linesValidity[3])
                    {
                        // opposite cell vertical line
                        lineSegment.P0 = float2(0.0f, (offset.y < 0) ? offset.y * cellWidth : 0.0f);
                        lineSegment.P1 = float2(0.0f, (offset.y > 0) ? offset.y * cellWidth : 0.0f);
                        phaseShift = fract((lineSegment.P0.y - localGridTopLeftCorner.y) * rcpPattenLenScreenSpace);
                        sdf = min(sdf, dtm::calculateLineSDF(outlineStyle, worldToScreenRatio, lineSegment, localFragPos, phaseShift));
                    }

                    float4 outlineColor = outlineStyle.color;
                    outlineColor.a *= 1.0f - smoothstep(-globals.antiAliasingFactor, globals.antiAliasingFactor, sdf);
                    dtmColor = dtm::blendUnder(dtmColor, outlineColor);
                }
                
                if (dtmSettings.drawHeightShadingEnabled())
                {
                    if (currentTriangleIndex != InvalidTriangleIndex)
                    {
                        dtm::GridDTMTriangle currentTriangle = triangles[currentTriangleIndex];
                        float heightDeriv = fwidth(interpolatedHeights[currentTriangleIndex]);
                        dtmColor = dtm::blendUnder(dtmColor, dtm::calculateDTMHeightColor(dtmSettings.heightShadingSettings, currentTriangle.vertices, heightDeriv, gridSpacePos, interpolatedHeights[currentTriangleIndex]));
                    }
                    else
                    {
                        // TODO[Future]: Average color of nearby valid triangles (dtm height function should return color + polygon sdf) 
                    }

                }
                
                textureColor = dtmColor.rgb / dtmColor.a;
                localAlpha = dtmColor.a;
            }

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
