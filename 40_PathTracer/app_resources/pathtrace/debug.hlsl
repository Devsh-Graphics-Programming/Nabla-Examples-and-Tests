#include "common.hlsl"

// Light-tree NEE debug visualiser.
//
// Goal: convince yourself the stochastic light-cut tree behaves. A movable probe
// (the "shading point", driven by the ImGuizmo in the Debug Probe panel) is
// tested against the tree on the CPU every time it moves; the renderer ships the
// results to us through the host-coherent SDebugProbe buffer so this shader does
// ZERO tree descent per pixel (GPU descent here spills private memory and device-losts).
// We only READ pre-chewed telemetry and draw it.
//
// What you see, composited over a flat schematic of the scene:
//   - emitter geometry           : tinted by its own NEE pdf (heat ramp)
//   - leaf AABBs                 : wireframe, heat-ramped by that leaf's pdf
//   - internal cluster AABBs     : faint wireframe, tinted by tree depth
//   - the u=0.5 descent path     : the root->leaf chain the sampler would walk,
//                                  drawn as a bright green wireframe on top
//   - screen border              : green when sum(pdf)==1 (sampler is a valid
//                                  probability distribution), red when it isn't

[[vk::push_constant]] SDebugPushConstants pc;

// sentinel raygen writes when the primary ray hits no geometry
static const uint32_t NoGeometryHit = NonEmitterCustomIndex;

struct[raypayload] DebugPayload
{
   uint32_t   instanceCustomIndex : read(caller) : write(closesthit, miss); // per-instance base (no longer the emitter idx), or NoGeometryHit on miss
   float      tHit : read(caller) : write(closesthit, miss);
   float32_t3 hitNormal : read(caller) : write(closesthit, miss);
};

// pdf -> colour. log10 maps [1e-5,1] to [0,1]; ramp is blue (cold/unlikely) ->
// green -> red (hot/likely). Distinct enough to read relative pdf at a glance.
// Distance (world units) from the probe -> colour. near = red, mid = green,
// far = blue. kProbeMaxDist is the scene scale; the ramp saturates beyond it.
static const float kProbeMaxDist = 6.0f; // ~scene width in metres; tune per scene
float32_t3         distHeat(float dist)
{
   const float      t        = clamp(dist * (1.0f / kProbeMaxDist), 0.0f, 1.0f);
   const float32_t3 cClose   = float32_t3(1.0f, 0.15f, 0.0f);
   const float32_t3 cMid     = float32_t3(0.0f, 1.0f, 0.2f);
   const float32_t3 cDistant = float32_t3(0.0f, 0.3f, 1.0f);
   return (t < 0.5f) ? lerp(cClose, cMid, t * 2.0f) : lerp(cMid, cDistant, (t - 0.5f) * 2.0f);
}

// EXACT mirror of the descent's distSq (stochastic_lightcut_tree.hlsl
// lightcutTreeChildWeight, mode 0): squared distance from the shading point to
// the box CENTROID, floored only against divide-by-zero by halfDiagSq*1e-6.
// This is the value that goes into the cluster selection weight (power*orient/distSq).
// Colouring boxes by sqrt(this) shows what distance the descent actually "sees"
// to each cluster, which is the centroid distance, NOT the nearest-surface
// distance the box wireframe sits at.
float descentDistSq(float32_t3 bboxMin, float32_t3 bboxMax, float32_t3 x)
{
   const float32_t3 ext            = bboxMax - bboxMin;
   const float      halfDiagSq     = 0.25f * dot(ext, ext);
   const float32_t3 center         = 0.5f * (bboxMin + bboxMax);
   const float32_t3 dToCentroid    = center - x;
   const float      centroidDistSq = dot(dToCentroid, dToCentroid);
   return max(centroidDistSq, halfDiagSq * 1e-6f);
}

// Fold the probe NORMAL into a distance colour. `p` is the shaded world point.
// dir = probe->p; points below the probe's tangent plane (n.dir <= 0) are NEE-
// unreachable -> rendered dark (culled), matching the tree's orientFactor cull.
// In front, the hue (distance) is kept but dimmed by a cosine falloff so the
// probe's facing hemisphere is obvious.
float32_t3 applyProbeOrientation(float32_t3 distColor, float32_t3 p, float32_t3 probePoint, float32_t3 probeNormal)
{
   const float32_t3 toP = p - probePoint;
   const float      len = length(toP);
   const float      ndl = (len > 1e-6f) ? dot(probeNormal, toP / len) : 1.0f;
   if (ndl <= 0.0f)
      return float32_t3(0.02f, 0.02f, 0.025f); // below horizon: NEE-culled
   return distColor * (0.25f + 0.75f * ndl);
}

// Wire colour for a box edge: its importance heat normally, but a white CORE when
// the box is on the u=0.5 descent path (prox is the edge proximity; the heat band
// is wider than the white core, so a path box reads as "hot cluster + white line").
float32_t3 wireColor(float32_t3 heat, float prox, bool onPath)
{ return (onPath && prox < 0.022f) ? float32_t3(1.0f, 1.0f, 1.0f) : heat; }

// Viridis-ish depth color: depth 0 (root) -> purple, deeper -> green. For internal
// cluster wireframes so siblings at the same level read as a coherent shell.
float32_t3 depthColor(uint32_t depth, uint32_t maxDepth)
{
   const float      t     = float(depth) / float(max(maxDepth, 1u));
   const float32_t3 cRoot = float32_t3(0.40f, 0.0f, 0.60f);
   const float32_t3 cMid  = float32_t3(0.10f, 0.55f, 0.55f);
   const float32_t3 cLeaf = float32_t3(0.20f, 0.85f, 0.25f);
   return (t < 0.5f) ? lerp(cRoot, cMid, t * 2.0f) : lerp(cMid, cLeaf, (t - 0.5f) * 2.0f);
}

// Vivid per-emitter color so adjacent lights read as distinct outlines. PCG-style
// hash of emitterID, then map to a saturated HSV-like primary so the value channel
// stays bright (no dark-on-dark surface).
float32_t3 emitterColor(uint32_t emitterID)
{
   uint32_t h = emitterID * 0x9E3779B9u + 0xCAFEBABEu;
   h ^= h >> 16u;
   h *= 0x7FEB352Du;
   h ^= h >> 15u;
   h *= 0x846CA68Bu;
   h ^= h >> 16u;
   const float hueDeg = float(h & 0xFFFFFFu) * (360.0f / 16777216.0f);
   const float c      = 1.0f; // saturation * value -> max chroma
   const float hp     = hueDeg * (1.0f / 60.0f); // [0,6)
   const float x      = c * (1.0f - abs(fmod(hp, 2.0f) - 1.0f));
   float32_t3  rgb;
   if (hp < 1.0f)
      rgb = float32_t3(c, x, 0.0f);
   else if (hp < 2.0f)
      rgb = float32_t3(x, c, 0.0f);
   else if (hp < 3.0f)
      rgb = float32_t3(0.0f, c, x);
   else if (hp < 4.0f)
      rgb = float32_t3(0.0f, x, c);
   else if (hp < 5.0f)
      rgb = float32_t3(x, 0.0f, c);
   else
      rgb = float32_t3(c, 0.0f, x);
   return rgb;
}

// Is `node` an ancestor (or equal) of the deterministic-descent leaf? Walk the
// leaf's parent chain upward. This reconstructs the full traversal path from a
// single heap index, so the host only has to ship that one number.
bool onDescentPath(uint32_t node, uint32_t descentLeaf)
{
   if (descentLeaf == ~0u)
      return false;
   uint32_t h = descentLeaf;
   NBL_HLSL_LOOP
   for (uint32_t i = 0u; i < 33u; ++i)
   {
      if (h == node)
         return true;
      if (h == 0u)
         break;
      h = (h - 1u) / 4u;
   }
   return false;
}

// Closest ray-AABB entry distance in [tMin,tMax]; -1 if missed. Zero-component
// dirs are nudged so a ray parallel to (and coplanar with) a slab can't make 0*inf.
float rayAabbEnterT(float32_t3 origin, float32_t3 dir, float tMin, float tMax, float32_t3 bMin, float32_t3 bMax)
{
   const float      kTiny   = 1e-30f;
   const float32_t3 safeDir = float32_t3(abs(dir.x) < kTiny ? (dir.x < 0.0f ? -kTiny : kTiny) : dir.x, abs(dir.y) < kTiny ? (dir.y < 0.0f ? -kTiny : kTiny) : dir.y, abs(dir.z) < kTiny ? (dir.z < 0.0f ? -kTiny : kTiny) : dir.z);
   const float32_t3 invD    = float32_t3(1.0f, 1.0f, 1.0f) / safeDir;
   const float32_t3 t0      = (bMin - origin) * invD;
   const float32_t3 t1      = (bMax - origin) * invD;
   const float32_t3 tNear   = min(t0, t1);
   const float32_t3 tFar    = max(t0, t1);
   const float      tEnter  = max(max(tNear.x, tNear.y), max(tNear.z, tMin));
   const float      tExit   = min(min(tFar.x, tFar.y), min(tFar.z, tMax));
   return (tEnter <= tExit) ? tEnter : -1.0f;
}

// Wireframe edge proximity for a point `p` on a box surface: the normalized
// distance to the nearest box EDGE along the entry face. The entry-face axis is
// pinned to a boundary (~0), so the relevant value is the SECOND smallest of the
// three per-axis normalized boundary distances (== 0 on an edge, large on a face
// interior). Compare against a threshold to draw a wire of that thickness; a
// smaller threshold inside it draws a thinner core. Edges only => transparent
// interiors so nested clusters don't occlude.
float aabbEdgeProximity(float32_t3 p, float32_t3 bMin, float32_t3 bMax)
{
   const float32_t3 ext = max(bMax - bMin, float32_t3(1e-6f, 1e-6f, 1e-6f));
   const float32_t3 d   = min(abs(p - bMin), abs(p - bMax)) / ext;
   const float      mn  = min(d.x, min(d.y, d.z));
   const float      mx  = max(d.x, max(d.y, d.z));
   return d.x + d.y + d.z - mn - mx; // middle value = 2nd smallest
}

[shader("raygeneration")] void raygen()
{
   const uint16_t3 launchID = uint16_t3(spirv::LaunchIdKHR);

   SPixelSamplingInfo samplingInfo = advanceSampleCount(launchID, 1, uint16_t(pc.sensorDynamics.keepAccumulating), 0xffffffffu);

   const float32_t2 pixelSizeNDC = promote<float32_t2>(2.f) / float32_t2(spirv::LaunchSizeKHR.xy);
   const float32_t2 NDC          = float32_t2(launchID.xy) * pixelSizeNDC - promote<float32_t2>(1.f);

   [[vk::ext_storage_class(spv::StorageClassRayPayloadKHR)]]
   DebugPayload payload;

   // Deterministic centre-of-pixel ray: the box wireframe is overwritten (not
   // accumulated), so any sub-pixel jitter would make it shimmer frame to frame.
   const float16_t2 randVec = float16_t2(0.5h, 0.5h);
   SPrimaryRay      primary = genPrimaryRay(pc.sensorDynamics, pixelSizeNDC, NDC, randVec);
   spirv::traceRayKHR(gTLASes[0], spv::RayFlagsMaskNone, 0xff, 0u, 0u, 0u, primary.ray.origin, primary.tMin, primary.ray.direction.getDirection(), pc.sensorDynamics.tMax, payload);

   const float32_t3 rayO = primary.ray.origin;
   const float32_t3 rayD = primary.ray.direction.getDirection();
   const float      tHit = payload.tHit;

   const bool     haveTree  = gScene.init.pLightTreeLeaves != 0 && gScene.init.lightTreeNumLeavesPadded > 0u;
   const uint32_t firstLeaf = gScene.init.lightTreeFirstLeafIndex;

   // Probe telemetry (CPU-computed; see SDebugProbe). No descent on the GPU.
   float32_t3 probePoint  = float32_t3(0.0f, 0.0f, 0.0f);
   float32_t3 probeNormal = float32_t3(0.0f, 1.0f, 0.0f);
   float      pdfSum      = 0.0f;
   uint32_t   descentLeaf = ~0u;
   if (gScene.init.pDebugProbe != 0)
   {
      // 32 B SDebugProbe in two uint4 taps: lo = probePoint.xyz | pdfSum, hi = probeNormal.xyz | descentLeaf.
      const uint32_t4 lo = vk::RawBufferLoad<uint32_t4>(gScene.init.pDebugProbe + 0ull,  16u);
      const uint32_t4 hi = vk::RawBufferLoad<uint32_t4>(gScene.init.pDebugProbe + 16ull, 16u);
      probePoint  = asfloat(lo.xyz);
      pdfSum      = asfloat(lo.w);
      probeNormal = asfloat(hi.xyz);
      descentLeaf = hi.w;
   }

   // --- base fill: mesh surfaces shaded by distance + probe orientation ---
   // A geometry hit (emitter or not) has tHit < tMax; a sky miss keeps tHit==tMax.
   const bool       hitGeometry = tHit < pc.sensorDynamics.tMax;
   const float32_t3 hitPos      = rayO + rayD * tHit;
   float32_t3       col         = hitGeometry ? applyProbeOrientation(distHeat(distance(hitPos, probePoint)), hitPos, probePoint, probeNormal) : float32_t3(0.04f, 0.04f, 0.05f); // sky / miss

   // --- wireframe overlay: leaves (per-emitter hue) + internal clusters (depth color) ---
   // No quantization anymore; wide-node child bboxes are bit-exact equal to the precise leaf
   // bboxes for leaf-child slots, so there's no halo to draw. One pass per layer.
   if (haveTree)
   {
      const uint64_t pNodes = gScene.init.pLightTreeNodes;

      // Pass 1: precise leaf AABBs (the per-light wires). One bright color per emitterID.
      float      bestT_p   = tHit;
      bool       gotEdge_p = false;
      float32_t3 edgeCol_p = col;
      NBL_HLSL_LOOP
      for (uint32_t l = 0u; l < gScene.init.lightTreeNumLeavesPadded; ++l)
      {
         nbl::this_example::LightTreeLeaf leaf;
         nbl::this_example::BDALightTreeLeafAccessor::create(gScene.init.pLightTreeLeaves).template get<nbl::this_example::LightTreeLeaf, uint32_t>(l, leaf);
         if (leaf.emitterID >= NoGeometryHit)
            continue;
         const uint32_t leafHeap = firstLeaf + l;
         const bool     onPath   = onDescentPath(leafHeap, descentLeaf);
         const float    t        = rayAabbEnterT(rayO, rayD, primary.tMin, bestT_p, leaf.bboxMin, leaf.bboxMax);
         if (t >= 0.0f && t < bestT_p)
         {
            const float prox  = aabbEdgeProximity(rayO + rayD * t, leaf.bboxMin, leaf.bboxMax);
            const float thick = onPath ? 0.07f : 0.03f;
            if (prox < thick)
            {
               bestT_p   = t;
               gotEdge_p = true;
               edgeCol_p = wireColor(emitterColor(leaf.emitterID), prox, onPath);
            }
         }
      }

      // Pass 2: internal-cluster wireframes (every non-leaf child of every wide-node). Colored
      // by tree depth so siblings at the same level form a coherent shell, clearly distinct
      // from the per-emitter leaf hues. Each wide-node is decoded via the library unpack
      // (lightcutTreeUnpackWideNode), same contract as the renderer's BDA accessors.
      float          bestT_c   = tHit;
      bool           gotEdge_c = false;
      float32_t3     edgeCol_c = col;
      const uint32_t kMaxDepth = 10u;
      NBL_HLSL_LOOP
      for (uint32_t W = 0u; pNodes != 0 && W < firstLeaf; ++W)
      {
         const uint64_t addr = pNodes + uint64_t(W) * 32ull;
         // Load + library-unpack the 32 B wide-node. a.w carries the leaf mask, so the all-leaf
         // early-out skips the second tap + the unpack.
         const uint32_t4 a        = vk::RawBufferLoad<uint32_t4>(addr + 0ull, 16u);
         const uint32_t  leafMask = (a.w >> 24u) & 0xFu;
         if (leafMask == 0xFu)
            continue; // every child is a leaf, already drawn by Pass 1
         const uint32_t4 b = vk::RawBufferLoad<uint32_t4>(addr + 16ull, 16u);
         nbl::hlsl::sampling::LightcutTreePackedWideNode packed;
         packed.origin      = asfloat(a.xyz);
         packed.powExpMask  = a.w;
         packed.childPacked = b;
         const nbl::hlsl::sampling::LightcutTreeWideNode<float32_t> node = nbl::hlsl::sampling::lightcutTreeUnpackWideNode<float32_t>(packed);

         uint32_t depth = 0u;
         {
            uint32_t h = W;
            NBL_HLSL_LOOP
            for (uint32_t i = 0u; i < 16u; ++i)
            {
               if (h == 0u)
                  break;
               h = (h - 1u) / 4u;
               ++depth;
            }
         }

         NBL_UNROLL
         for (uint32_t slot = 0u; slot < 4u; ++slot)
         {
            if (((leafMask >> slot) & 1u) != 0u)
               continue; // leaf, handled by Pass 1
            if (!(node.children[slot].power > 0.f))
               continue; // padding cluster
            const float32_t3 bMin = node.children[slot].bboxMin;
            const float32_t3 bMax = node.children[slot].bboxMax;

            const uint32_t childHeap = 4u * W + 1u + slot;
            const bool     onPath    = onDescentPath(childHeap, descentLeaf);
            const float    t         = rayAabbEnterT(rayO, rayD, primary.tMin, bestT_c, bMin, bMax);
            if (t >= 0.0f && t < bestT_c)
            {
               const float prox  = aabbEdgeProximity(rayO + rayD * t, bMin, bMax);
               const float thick = onPath ? 0.05f : 0.025f;
               if (prox < thick)
               {
                  bestT_c   = t;
                  gotEdge_c = true;
                  edgeCol_c = wireColor(depthColor(depth + 1u, kMaxDepth), prox, onPath);
               }
            }
         }
      }

      // Cluster wires furthest back, leaf wires on top. White descent-path core (wireColor)
      // still wins inside its proximity band.
      if (gotEdge_c)
         col = edgeCol_c;
      if (gotEdge_p)
         col = edgeCol_p;
   }

   // --- correctness border: sum(pdf) over all leaves must be ~1 ---
   if (haveTree)
   {
      const uint32_t2 dim      = uint32_t2(spirv::LaunchSizeKHR.xy);
      const uint32_t2 px       = uint32_t2(launchID.xy);
      const uint32_t  margin   = 8u;
      const bool      inBorder = px.x < margin || px.y < margin || px.x + margin >= dim.x || px.y + margin >= dim.y;
      if (inBorder)
      {
         const float g = clamp(1.0f - abs(pdfSum - 1.0f) * 10.0f, 0.0f, 1.0f);
         col           = lerp(float32_t3(1.0f, 0.0f, 0.0f), float32_t3(0.0f, 1.0f, 0.0f), g);
      }
   }

   gRWMCCascades[launchID] = uint32_t2(payload.instanceCustomIndex, 0u);
   Accumulator<ImageAccessor_gAlbedo> albedoAcc;
   albedoAcc.accumulate(launchID.xy, launchID.z, col, 1.0f, false);
   Accumulator<ImageAccessor_gNormal> normalAcc;
   normalAcc.accumulate(launchID.xy, launchID.z, float32_t3(correctSNorm10WhenStoringToUnorm(payload.hitNormal)), 1.0f, false);
}

   [shader("closesthit")] void closestHit(inout DebugPayload payload, in BuiltInTriangleIntersectionAttributes attribs)
{
   payload.instanceCustomIndex = uint32_t(spirv::InstanceCustomIndexKHR);
   payload.tHit                = spirv::RayTmaxKHR;

   float32_t3 hitNormal = reconstructGeometricNormal();
   if (dot(spirv::WorldRayDirectionKHR, hitNormal) > 0.f)
      hitNormal = -hitNormal;
   payload.hitNormal = hitNormal;
}

[shader("miss")] void miss(inout DebugPayload payload)
{
   payload.instanceCustomIndex = NoGeometryHit;
   payload.tHit                = spirv::RayTmaxKHR;
   payload.hitNormal           = -hlsl::normalize(spirv::WorldRayDirectionKHR);
}
