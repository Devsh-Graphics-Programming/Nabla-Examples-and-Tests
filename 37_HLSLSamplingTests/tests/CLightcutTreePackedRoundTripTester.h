#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_C_LIGHTCUT_TREE_PACKED_ROUNDTRIP_TESTER_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_C_LIGHTCUT_TREE_PACKED_ROUNDTRIP_TESTER_INCLUDED_

#include <nbl/builtin/hlsl/sampling/stochastic_lightcut_tree.hlsl>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

class CLightcutTreePackedRoundTripTester
{
   public:
   CLightcutTreePackedRoundTripTester(system::ILogger* logger) : m_logger(logger) {}

   bool run()
   {
      bool pass = true;
      m_logger->log("LightcutTree packed round-trip tests:", system::ILogger::ELL_INFO);

      // Deterministic edge cases: cube, thin anisotropic slabs (shared-exponent stress), large
      // coordinates (fp32-origin reason), tiny extent.
      pass &= testNode("Cube",       float3(0, 0, 0),              float3(1, 1, 1),          nullptr);
      pass &= testNode("ThinSlabX",  float3(-5, 2, 9),             float3(0.01f, 4, 4),      nullptr);
      pass &= testNode("ThinSlabY",  float3(100, 100, 0),         float3(8, 0.02f, 8),      nullptr);
      pass &= testNode("BigCoords",  float3(70000, -90000, 50000), float3(1000, 1000, 1000), nullptr);
      pass &= testNode("TinyExtent", float3(0, 0, 0),              float3(1e-3f, 1e-3f, 1e-3f), nullptr);

      // Random nodes with anisotropic extents + random children strictly inside the parent.
      // Silent per-node (only failures log); one summary line for the whole batch.
      std::mt19937                          rng(0x1347u);
      std::uniform_real_distribution<float> origDist(-1000.f, 1000.f);
      std::uniform_real_distribution<float> extDist(0.01f, 200.f);
      bool                                  randPass = true;
      for (uint32_t i = 0; i < 256u; ++i)
      {
         const float3 origin(origDist(rng), origDist(rng), origDist(rng));
         const float3 ext(extDist(rng), extDist(rng), extDist(rng));
         randPass &= testNode("Random", origin, ext, &rng, /*quiet=*/true);
      }
      if (randPass)
         m_logger->log("  [Random x256] PASSED", system::ILogger::ELL_PERFORMANCE);
      pass &= randPass;

      pass &= testLeaf();
      return pass;
   }

   private:
   using float3 = nbl::hlsl::float32_t3;

   // One axis of a child box inside [origin, origin+ext]: returns [lo,hi] absolute coords.
   static void axisBox(float origin, float ext, bool rand, std::mt19937* rng, uint32_t s, float& lo, float& hi)
   {
      if (rand)
      {
         std::uniform_real_distribution<float> u01(0.f, 1.f);
         const float a = u01(*rng), b = u01(*rng);
         lo = origin + std::min(a, b) * ext;
         hi = origin + std::max(a, b) * ext;
      }
      else
      {
         const float f0 = (s & 1u) ? 0.5f : 0.0f;
         const float f1 = (s == 3u) ? 1.0f : ((s & 1u) ? 1.0f : 0.5f);
         lo = origin + f0 * ext;
         hi = origin + f1 * ext;
      }
   }

   bool testNode(const char* name, const float3 origin, const float3 ext, std::mt19937* rng, bool quiet = false) const
   {
      float3   childMin[4];
      float3   childMax[4];
      float    childPower[4];
      uint32_t leafMask = 0u;
      for (uint32_t s = 0; s < 4u; ++s)
      {
         float lox, hix, loy, hiy, loz, hiz;
         axisBox(origin.x, ext.x, rng != nullptr, rng, s, lox, hix);
         axisBox(origin.y, ext.y, rng != nullptr, rng, s, loy, hiy);
         axisBox(origin.z, ext.z, rng != nullptr, rng, s, loz, hiz);
         childMin[s]   = float3(lox, loy, loz);
         childMax[s]   = float3(hix, hiy, hiz);
         childPower[s] = (s == 2u) ? 0.0f : (0.1f + float(s)); // child 2 is zero-power padding
         if (childPower[s] > 0.f)
            leafMask |= (1u << s);
      }

      const float parentPower = 4.2f;

      // ----- pack (mirrors what the CPU builders do, via the library helpers) -----
      const float    maxExt          = std::max({ext.x, ext.y, ext.z});
      const uint32_t expS            = nbl::hlsl::sampling::lightcutTreePickBiasedExp(maxExt);
      const float    scale           = nbl::hlsl::sampling::lightcutTreeBiasedExpToScale(expS);
      const float    parentPowerSafe = parentPower > 0.f ? parentPower : 1.f;

      nbl::hlsl::sampling::LightcutTreePackedWideNode packed = {};
      packed.origin     = origin;
      packed.powExpMask = nbl::hlsl::sampling::lightcutTreePackPowExpMask(parentPower, expS, leafMask);
      uint32_t cp[4];
      for (uint32_t s = 0; s < 4u; ++s)
         cp[s] = nbl::hlsl::sampling::lightcutTreePackChild(childMin[s] - origin, childMax[s] - origin, scale, childPower[s], parentPowerSafe);
      packed.childPacked = nbl::hlsl::uint32_t4(cp[0], cp[1], cp[2], cp[3]);

      // ----- unpack -----
      const nbl::hlsl::sampling::LightcutTreeWideNode<float> dec = nbl::hlsl::sampling::lightcutTreeUnpackWideNode<float>(packed);

      bool pass = true;
      if (dec.childLeafMask != (leafMask & 0xFu))
      {
         m_logger->log("PackRoundTrip[%s] childLeafMask: expected %u got %u", system::ILogger::ELL_ERROR, name, leafMask & 0xFu, dec.childLeafMask);
         pass = false;
      }

      const float originMag = std::max({std::abs(origin.x), std::abs(origin.y), std::abs(origin.z)});
      const float eps       = std::max(1e-3f, 1e-4f * (maxExt + originMag));
      for (uint32_t s = 0; s < 4u; ++s)
      {
         pass &= containsAxis(name, s, 'x', dec.children[s].bboxMin.x, dec.children[s].bboxMax.x, childMin[s].x, childMax[s].x, eps);
         pass &= containsAxis(name, s, 'y', dec.children[s].bboxMin.y, dec.children[s].bboxMax.y, childMin[s].y, childMax[s].y, eps);
         pass &= containsAxis(name, s, 'z', dec.children[s].bboxMin.z, dec.children[s].bboxMax.z, childMin[s].z, childMax[s].z, eps);

         // Reachability: a positive-power child must decode to positive power (floor-to-1 rule);
         // zero-power padding must stay zero.
         const bool truePos = childPower[s] > 0.f;
         const bool decPos  = dec.children[s].power > 0.f;
         if (truePos != decPos)
         {
            m_logger->log("PackRoundTrip[%s] child %u: power reachability mismatch (true %f -> decoded %f)", system::ILogger::ELL_ERROR, name, s, childPower[s], dec.children[s].power);
            pass = false;
         }
      }

      if (pass && !quiet)
         m_logger->log("  [%s] PASSED", system::ILogger::ELL_PERFORMANCE, name);
      return pass;
   }

   bool containsAxis(const char* name, uint32_t s, char axis, float decMin, float decMax, float trueMin, float trueMax, float eps) const
   {
      bool pass = true;
      if (decMin > trueMin + eps)
      {
         m_logger->log("PackRoundTrip[%s] child %u axis %c: decoded min %f does not contain true min %f", system::ILogger::ELL_ERROR, name, s, axis, decMin, trueMin);
         pass = false;
      }
      if (decMax < trueMax - eps)
      {
         m_logger->log("PackRoundTrip[%s] child %u axis %c: decoded max %f does not contain true max %f", system::ILogger::ELL_ERROR, name, s, axis, decMax, trueMax);
         pass = false;
      }
      return pass;
   }

   bool testLeaf() const
   {
      bool                                  pass = true;
      std::mt19937                          rng(0xFEA0u);
      std::uniform_real_distribution<float> d(-50000.f, 50000.f);
      for (uint32_t i = 0; i < 64u; ++i)
      {
         const float3   mn(d(rng), d(rng), d(rng));
         const float3   mx(mn.x + std::abs(d(rng)) * 1e-2f, mn.y + std::abs(d(rng)) * 1e-2f, mn.z + std::abs(d(rng)) * 1e-2f);
         const uint32_t eid = (i == 0u) ? nbl::hlsl::sampling::LightcutTreePackedNoEmitter : i;

         nbl::hlsl::sampling::LightcutTreePackedLeaf pl = {};
         pl.bboxMin   = mn;
         pl.bboxMax   = mx;
         pl.emitterID = eid;

         const nbl::hlsl::sampling::LightcutTreeLeaf<float> dec = nbl::hlsl::sampling::lightcutTreeUnpackLeaf<float>(pl);
         if (dec.bboxMin.x != mn.x || dec.bboxMin.y != mn.y || dec.bboxMin.z != mn.z ||
             dec.bboxMax.x != mx.x || dec.bboxMax.y != mx.y || dec.bboxMax.z != mx.z)
         {
            m_logger->log("PackRoundTrip[Leaf %u] bbox not exact", system::ILogger::ELL_ERROR, i);
            pass = false;
         }
         const uint32_t expectedEid = (eid == nbl::hlsl::sampling::LightcutTreePackedNoEmitter) ? ~0u : eid;
         if (dec.emitterID != expectedEid)
         {
            m_logger->log("PackRoundTrip[Leaf %u] emitterID expected %u got %u", system::ILogger::ELL_ERROR, i, expectedEid, dec.emitterID);
            pass = false;
         }
      }
      if (pass)
         m_logger->log("  [Leaf] PASSED", system::ILogger::ELL_PERFORMANCE);
      return pass;
   }

   system::ILogger* m_logger;
};

#endif
