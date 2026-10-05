//// Copyright (C) 2026 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _SOLID_ANGLE_VIS_EXAMPLE_DRAWING_HLSL_INCLUDED_
#define _SOLID_ANGLE_VIS_EXAMPLE_DRAWING_HLSL_INCLUDED_

#include "common.hlsl"
#include "silhouette.hlsl"
#include <nbl/builtin/hlsl/shapes/obb.hlsl>

using namespace nbl::hlsl;

// ============================================================================
// SphereDrawer: all visualization primitives for the solid angle visualizer.
// All methods are static and read VisContext for ndc/spherePos/aaWidth.
// ============================================================================
struct SphereDrawer
{
   // ========================================================================
   // Coordinate helpers
   // ========================================================================

   // Project sphere point to circle-space (doesn't change Z)
   static float32_t3 sphereToCircle(float32_t3 spherePoint)
   {
      if (spherePoint.z >= 0.0f)
      {
         return float32_t3(spherePoint.xy * CIRCLE_RADIUS, spherePoint.z);
      }
      else
      {
         float32_t r2       = (1.0f - spherePoint.z) / (1.0f + spherePoint.z);
         float32_t uv2Plus1 = r2 + 1.0f;
         return float32_t3((spherePoint.xy * uv2Plus1 / 2.0f) * CIRCLE_RADIUS, spherePoint.z);
      }
   }

   // ========================================================================
   // Primitives
   // ========================================================================

   // Great circle arc between two points on the sphere
   static float32_t drawGreatCircleArc(float32_t3 points[2], float32_t width = 0.01f)
   {
      float32_t3 v0  = normalize(points[0]);
      float32_t3 v1  = normalize(points[1]);
      float32_t3 ndc = normalize(VisContext::spherePos());

      float32_t3 arcNormal = normalize(cross(v0, v1));
      float32_t  dist      = abs(dot(ndc, arcNormal));

      float32_t dotMid = dot(v0, v1);
      bool      onArc  = (dot(ndc, v0) >= dotMid) && (dot(ndc, v1) >= dotMid);

      if (!onArc)
         return 0.0f;

      float32_t avgDepth   = (length(points[0]) + length(points[1])) * 0.5f;
      float32_t depthScale = 3.0f / avgDepth;

      width                   = min(width * depthScale, 0.02f);
      const float32_t aaWidth = VisContext::aaWidth();
      float32_t       alpha   = 1.0f - smoothstep(width - aaWidth, width + aaWidth, dist);

      return alpha;
   }

   // 2D cross marker
   static float32_t drawCross2D(float32_t2 fragPos, float32_t2 center, float32_t size, float32_t thickness)
   {
      float32_t2 ndc = abs(fragPos - center);

      bool inHorizontal = (ndc.x <= size && ndc.y <= thickness);
      bool inVertical   = (ndc.y <= size && ndc.x <= thickness);

      return (inHorizontal || inVertical) ? 1.0f : 0.0f;
   }

   // Dot (circle) with optional inner hollow for hidden corners
   static float32_t4 drawDot(float32_t3 cornerNDCPos, float32_t dotSize, float32_t innerDotSize, float32_t3 dotColor)
   {
      float32_t4       color   = float32_t4(0, 0, 0, 0);
      const float32_t  aaWidth = VisContext::aaWidth();
      const float32_t2 ndc     = VisContext::ndc();
      const float32_t  dist    = length(ndc - cornerNDCPos.xy);

      float32_t outerAlpha = 1.0f - smoothstep(dotSize - aaWidth, dotSize + aaWidth, dist);

      if (outerAlpha <= 0.0f)
         return color;

      color += float32_t4(dotColor * outerAlpha, outerAlpha);

      if (cornerNDCPos.z < 0.0f && innerDotSize > 0.0)
      {
         float32_t innerAlpha = 1.0f - smoothstep(innerDotSize - aaWidth, innerDotSize + aaWidth, dist);
         innerAlpha *= outerAlpha;
         color -= float32_t4(hlsl::promote<float32_t3>(innerAlpha), 0.0f);
      }

      return color;
   }

   // Line segment in NDC space
   static float32_t lineSegment(float32_t2 ndc, float32_t2 a, float32_t2 b, float32_t thickness)
   {
      float32_t2 pa   = ndc - a;
      float32_t2 ba   = b - a;
      float32_t  h    = saturate(dot(pa, ba) / dot(ba, ba));
      float32_t  dist = length(pa - ba * h);
      return smoothstep(thickness, thickness * 0.5, dist);
   }

   // Draw half of a great circle (visible half of a lune boundary)
   static float32_t4 drawGreatCircleHalf(float32_t3 normal, float32_t3 axis3, float32_t3 color, float32_t thickness)
   {
      // Point is on great circle if dot(point, normal) ~= 0
      // Only draw the half where dot(point, axis3) > 0 (toward silhouette)
      const float32_t3 spherePos = VisContext::spherePos();
      const float32_t  aaWidth   = VisContext::aaWidth();

      float32_t dist     = abs(dot(spherePos, normal));
      float32_t sideFade = smoothstep(-0.1f, 0.1f, dot(spherePos, axis3));
      float32_t alpha    = (1.0f - smoothstep(thickness - aaWidth, thickness + aaWidth, dist)) * sideFade;
      return float32_t4(color * alpha, alpha);
   }

   // Unit-circle ring
   static float32_t4 drawRing(float32_t2 ndc)
   {
      const float32_t aaWidth        = VisContext::aaWidth();
      float32_t       ringWidth      = 0.003f;
      float32_t       positionLength = length(ndc);

      float32_t ringDistance = abs(positionLength - CIRCLE_RADIUS);
      float32_t ringAlpha    = 1.0f - smoothstep(ringWidth - aaWidth, ringWidth + aaWidth, ringDistance);
      return ringAlpha * float32_t4(0, 0, 0, 1);
   }

   // ========================================================================
   // Composite drawing helpers
   // ========================================================================

   // Silhouette edge with color from LUT
   static float32_t4 drawEdge(uint32_t originalEdgeIdx, float32_t3 pts[2], float32_t width = 0.003f)
   {
      float32_t alpha = drawGreatCircleArc(pts, width);
      return float32_t4(colorLUT[originalEdgeIdx] * alpha, alpha);
   }

   static float32_t4 drawCorner(float32_t3 cornerPos, float32_t dotSize, float32_t innerDotSize, float32_t3 dotColor)
   {
      float32_t3 cornerCirclePos = sphereToCircle(cornerPos);
      return drawDot(cornerCirclePos, dotSize, innerDotSize, dotColor);
   }

   // All 8 cube corners as colored dots
   static float32_t4 drawCorners(float32_t3x4 modelMatrix, float32_t dotSize)
   {
      float32_t4 color        = float32_t4(0, 0, 0, 0);
      float32_t  innerDotSize = dotSize * 0.5f;

      shapes::OBBView<float32_t> view = shapes::OBBView<float32_t>::create(modelMatrix);

      for (uint32_t i = 0; i < 8; i++)
      {
         color += drawCorner(normalize(view.getVertex(i)), dotSize, innerDotSize, colorLUT[i]);
      }

      return color;
   }

   static float32_t4 drawClippedSilhouetteVertices(float32_t3 vertices[MAX_SILHOUETTE_VERTICES], uint32_t count)
   {
      const float32_t  dotSize  = 0.03f;
      const float32_t2 ndc      = VisContext::ndc();
      const float32_t  rcpDenom = rcp(float32_t(max(1u, count - 1)));

      float32_t4 color = 0;

      for (uint32_t i = 0; i < count; i++)
      {
         const float32_t3 cornerCirclePos = sphereToCircle(normalize(vertices[i]));
         const float32_t  dist            = length(ndc - cornerCirclePos.xy);
         const float32_t  alpha           = 1.0f - smoothstep(dotSize * 0.8f, dotSize, dist);
         if (alpha > 0.0f)
         {
            const float32_t  t           = float32_t(i) * rcpDenom;
            const float32_t3 vertexColor = lerp(float32_t3(1, 0, 0), float32_t3(0, 1, 1), t);
            color += float32_t4(vertexColor * alpha, alpha);
         }
      }

      return color;
   }

   // Non-silhouette cube edges (drawn as faint lines)
   static float32_t4 drawHiddenEdges(float32_t3x4 modelMatrix, uint32_t silEdgeMask)
   {
      float32_t4 color           = 0;
      float32_t3 hiddenEdgeColor = float32_t3(0.1, 0.1, 0.1);

      shapes::OBBView<float32_t> view = shapes::OBBView<float32_t>::create(modelMatrix);

      // Enumerate all 12 cube edges: for each of 3 axes, 4 edges parallel to that axis.
      // compact (0..3) is the 2-bit corner index with the axis bit stripped out.
      // Reconstruct the full corner by re-inserting the axis bit as 0.
      NBL_UNROLL
      for (uint32_t axis = 0; axis < 3; axis++)
      {
         NBL_UNROLL
         for (uint32_t compact = 0; compact < 4; compact++)
         {
            uint32_t edgeIdx = axis * 4 + compact;
            if (silEdgeMask & (1u << edgeIdx))
               continue;

            // Re-insert the axis bit (as 0) to recover the low corner index
            uint32_t below  = compact & ((1u << axis) - 1u);
            uint32_t above  = compact >> axis;
            uint32_t corner = (above << (axis + 1u)) | below;

            float32_t3 v0 = normalize(view.getVertex(corner));
            float32_t3 v1 = normalize(view.getVertex(corner | (1u << axis)));

            bool neg0 = v0.z < 0.0f;
            bool neg1 = v1.z < 0.0f;

            // fully behind camera
            if (neg0 && neg1)
               continue;

            float32_t3 p0 = v0;
            float32_t3 p1 = v1;

            // clip if one vertex is behind camera
            if (neg0 ^ neg1)
            {
               float32_t  t    = v0.z / (v0.z - v1.z);
               float32_t3 clip = normalize(lerp(v0, v1, t));

               p0 = neg0 ? clip : v0;
               p1 = neg1 ? clip : v1;
            }

            float32_t3 pts[2] = {p0, p1};
            float32_t  c      = drawGreatCircleArc(pts, 0.003f);
            color += float32_t4(hiddenEdgeColor * c, c);
         }
      }

      return color;
   }

   // Best caliper edge highlighted in gold
   static float32_t4 visualizeBestCaliperEdge(float32_t3 vertices[MAX_SILHOUETTE_VERTICES], uint32_t count, uint32_t bestEdgeIdx)
   {
      float32_t4 result = float32_t4(0, 0, 0, 0);

      if (bestEdgeIdx >= count)
         return result;

      float32_t3 v0 = vertices[bestEdgeIdx];
      float32_t3 v1 = vertices[(bestEdgeIdx + 1) % count];

      float32_t3 pts[2]         = {v0, v1};
      float32_t3 highlightColor = float32_t3(1.0f, 0.8f, 0.0f);
      float32_t  alpha          = drawGreatCircleArc(pts, 0.008f);
      result += float32_t4(highlightColor * alpha, alpha);

      return result;
   }

   // ========================================================================
   // Sample visualization (sphere dot + parameter-space square overlay)
   // ========================================================================

   static float32_t4 visualizeSample(float32_t3 sampleDir, float32_t2 xi, uint32_t colorIndex, float32_t2 screenUV)
   {
      float32_t4 accumColor  = 0;
      float32_t3 sampleColor = colorLUT[colorIndex].rgb;

      // 3D dot on the sphere
      float32_t dist3D  = distance(sampleDir, normalize(VisContext::spherePos()));
      float32_t alpha3D = 1.0f - smoothstep(0.0f, 0.02f, dist3D);
      if (alpha3D > 0.0f)
         accumColor += float32_t4(sampleColor * alpha3D, alpha3D);

      // Parameter-space square (PSS) overlay
      static const float32_t2 pssSize     = float32_t2(0.2, 0.2);
      static const float32_t2 pssPos      = float32_t2(0.01, 0.01);
      bool                    isInsidePSS = all(and(screenUV >= pssPos, screenUV <= (pssPos + pssSize)));

      if (isInsidePSS)
      {
         // Cross marker at the sample's xi position
         float32_t2 xiPixelPos = pssPos + xi * pssSize;
         float32_t  alpha2D    = drawCross2D(screenUV, xiPixelPos, 0.005f, 0.001f);
         if (alpha2D > 0.0f)
            accumColor += float32_t4(sampleColor * alpha2D, alpha2D);

         // Faint border outline
         float32_t2 edgeDist    = min(screenUV - pssPos, (pssPos + pssSize) - screenUV);
         float32_t  borderDist  = min(edgeDist.x, edgeDist.y);
         float32_t  borderAlpha = 1.0f - smoothstep(0.001f, 0.003f, borderDist);
         if (borderAlpha > 0.0f)
            accumColor += float32_t4(0.3f, 0.3f, 0.3f, 1.0f) * borderAlpha;
      }

      return accumColor;
   }

   // ========================================================================
   // 3D ray arrow visualization
   // ========================================================================

   // Project 3D point to NDC space
   static float32_t2 projectToNDC(float32_t3 worldPos, float32_t4x4 viewProj, float32_t aspect)
   {
      float32_t4 clipPos = mul(viewProj, float32_t4(worldPos, 1.0));
      clipPos /= clipPos.w;
      clipPos.x *= aspect;
      return clipPos.xy;
   }

   struct ArrowResult
   {
      float32_t4 color;
      float32_t  depth;
   };

   // Visualize a ray as an arrow from origin in NDC space.
   // Returns color (rgb), intensity (a), and depth.
   static ArrowResult visualizeRayAsArrow(float32_t3 rayOrigin, float32_t4 directionAndPdf, float32_t arrowLength,
      float32_t2 ndcPos, float32_t aspect, float32_t4x4 viewProjMatrix)
   {
      ArrowResult result;
      result.color = float32_t4(0, 0, 0, 0);
      result.depth = 0.0; // Far plane in reversed-Z

      float32_t3 rayDir = normalize(directionAndPdf.xyz);
      float32_t  pdf    = directionAndPdf.w;

      // Define the 3D line segment
      float32_t3 worldStart = rayOrigin;
      float32_t3 worldEnd   = rayOrigin + rayDir * arrowLength;

      float32_t4 clipStart = mul(viewProjMatrix, float32_t4(worldStart, 1.0));
      float32_t4 clipEnd   = mul(viewProjMatrix, float32_t4(worldEnd, 1.0));

      // Clip against near plane (w = 0 plane in clip space)
      // If both points are behind camera, reject
      if (clipStart.w <= 0.001 && clipEnd.w <= 0.001)
         return result;

      // If line crosses the near plane, clip it
      float32_t t0 = 0.0;
      float32_t t1 = 1.0;

      if (clipStart.w <= 0.001)
      {
         float32_t t = (0.001 - clipStart.w) / (clipEnd.w - clipStart.w);
         t0          = saturate(t);
         clipStart   = lerp(clipStart, clipEnd, t0);
         worldStart  = lerp(worldStart, worldEnd, t0);
      }

      if (clipEnd.w <= 0.001)
      {
         float32_t t = (0.001 - clipStart.w) / (clipEnd.w - clipStart.w);
         t1          = saturate(t);
         clipEnd     = lerp(clipStart, clipEnd, t1);
         worldEnd    = lerp(worldStart, worldEnd, t1);
      }

      // Now check if the clipped segment is valid
      if (t0 >= t1)
         return result;

      // Perspective divide to NDC
      float32_t2 ndcStart = clipStart.xy / clipStart.w;
      float32_t2 ndcEnd   = clipEnd.xy / clipEnd.w;

      // Apply aspect ratio correction
      ndcStart.x *= aspect;
      ndcEnd.x *= aspect;

      // Calculate arrow direction in NDC
      float32_t2 arrowVec       = ndcEnd - ndcStart;
      float32_t  arrowNDCLength = length(arrowVec);

      // Skip if arrow is too small on screen
      if (arrowNDCLength < 0.005)
         return result;

      // Calculate perpendicular distance to line segment in NDC space
      float32_t2 toPixel = ndcPos - ndcStart;
      float32_t  t_ndc   = saturate(dot(toPixel, arrowVec) / dot(arrowVec, arrowVec));

      // Draw line shaft
      float32_t lineThickness = 0.002;
      float32_t lineIntensity = lineSegment(ndcPos, ndcStart, ndcEnd, lineThickness);

      // Calculate perspective-correct depth
      if (lineIntensity > 0.0)
      {
         float32_t4 clipPos  = lerp(clipStart, clipEnd, t_ndc);
         float32_t  depthNDC = clipPos.z / clipPos.w;
         result.depth        = 1.0f - depthNDC;

         if (result.depth < 0.0 || result.depth > 1.0)
            lineIntensity = 0.0;
      }

      // Modulate by PDF
      float32_t  pdfIntensity = saturate(pdf * 0.5);
      float32_t3 finalColor   = float32_t3(pdfIntensity, pdfIntensity, pdfIntensity);

      result.color = float32_t4(finalColor, lineIntensity);
      return result;
   }
};

#endif // _SOLID_ANGLE_VIS_EXAMPLE_DRAWING_HLSL_INCLUDED_
