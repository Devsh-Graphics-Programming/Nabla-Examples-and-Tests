#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_C_STOCHASTIC_LIGHTCUT_TREE_GPU_TESTER_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_C_STOCHASTIC_LIGHTCUT_TREE_GPU_TESTER_INCLUDED_

#include "nbl/examples/examples.hpp"
#include "app_resources/common/stochastic_lightcut_tree.hlsl"
#include "nbl/examples/Tester/ITester.h"
#include "SamplerTestHelpers.h"

template<typename Executor, uint32_t Mode>
class CStochasticLightcutTreeGPUTester final : public ITester<LightcutTreeInputValues, LightcutTreeTestResults, Executor>
{
	using base_t = ITester<LightcutTreeInputValues, LightcutTreeTestResults, Executor>;
	using R      = LightcutTreeTestResults;

	using typename base_t::TestType;
	using base_t::getRandomEngine;
	using base_t::verifyTestValue;
	using base_t::printTestFail;

	// Weight modes (mirror NBL_LIGHTCUT_TREE_WEIGHT_MODE): 0 = power*orient/dist^2, 1 = power,
	// 2 = uniform, 3 = power*orient. The geometric scenarios are instantiated once per mode and
	// their analytic expectations differ by mode (see analyticChildWeight + the per-scenario blocks).
	//
	// Discrimination matrix (what each mode is uniquely pinned by, beyond the universal
	// CPU==GPU / jacobian==1 / pdf>0 invariants that hold for ALL modes):
	//   mode 0: distFalloff gives 4/5,1/5 (inverse-square); no other mode does.
	//   mode 3: belowPlane culls (orientation), AND distFalloff gives 1/2,1/2 -> uniquely identified.
	//   modes 1 & 2: every scenario here has equal live-leaf power, so power and uniform coincide
	//                (1/2,1/2 on distFalloff, no cull on belowPlane). They are validated for
	//                consistency only, NOT discriminated from each other -- a node with unequal
	//                live-leaf powers would be needed to separate them, which no current scenario has.
	static constexpr bool kIsSingleLeaf  = std::is_same_v<Executor, LightcutTreeSingleLeafExecutor>;
	static constexpr bool kIsBelowPlane  = std::is_same_v<Executor, LightcutTreeBelowPlaneExecutor<Mode>>;
	static constexpr bool kIsDistFalloff = std::is_same_v<Executor, LightcutTreeDistanceFalloffExecutor<Mode>>;
	static constexpr bool kIsDepth2      = std::is_same_v<Executor, LightcutTreeDepth2Executor<Mode>>;
	static constexpr const char* kLeafHeapName        = kIsSingleLeaf ? "LightcutTree(single)::generatedLeafHeap"  : "LightcutTree::generatedLeafHeap";
	static constexpr const char* kEmitterIdName       = kIsSingleLeaf ? "LightcutTree(single)::generatedEmitterID" : "LightcutTree::generatedEmitterID";
	static constexpr const char* kForwardPdfName      = kIsSingleLeaf ? "LightcutTree(single)::forwardPdf"         : "LightcutTree::forwardPdf";
	static constexpr const char* kBackwardPdfName     = kIsSingleLeaf ? "LightcutTree(single)::backwardPdf"        : "LightcutTree::backwardPdf";
	static constexpr const char* kForwardWeightName   = kIsSingleLeaf ? "LightcutTree(single)::forwardWeight"      : "LightcutTree::forwardWeight";
	static constexpr const char* kBackwardWeightName  = kIsSingleLeaf ? "LightcutTree(single)::backwardWeight"     : "LightcutTree::backwardWeight";
	static constexpr const char* kJacobianName        = kIsSingleLeaf ? "LightcutTree(single)::jacobianProduct"    : "LightcutTree::jacobianProduct";
	static constexpr const char* kPdfConsistencyName  = kIsSingleLeaf ? "LightcutTree(single)::pdf consistency"    : "LightcutTree::pdf consistency";
	static constexpr const char* kBboxMinXName        = kIsSingleLeaf ? "LightcutTree(single)::leafBbox.min.x"     : "LightcutTree::leafBbox.min.x";
	static constexpr const char* kBboxMinYName        = kIsSingleLeaf ? "LightcutTree(single)::leafBbox.min.y"     : "LightcutTree::leafBbox.min.y";
	static constexpr const char* kBboxMinZName        = kIsSingleLeaf ? "LightcutTree(single)::leafBbox.min.z"     : "LightcutTree::leafBbox.min.z";
	static constexpr const char* kBboxMaxXName        = kIsSingleLeaf ? "LightcutTree(single)::leafBbox.max.x"     : "LightcutTree::leafBbox.max.x";
	static constexpr const char* kBboxMaxYName        = kIsSingleLeaf ? "LightcutTree(single)::leafBbox.max.y"     : "LightcutTree::leafBbox.max.y";
	static constexpr const char* kBboxMaxZName        = kIsSingleLeaf ? "LightcutTree(single)::leafBbox.max.z"     : "LightcutTree::leafBbox.max.z";

	// Independent C++ re-implementation of lightcutTreeChildWeight (mode 0), NOT a call into the
	// library: a library bug would make the CPU and GPU executors agree on the same wrong value and pass
	// the expected/actual comparison, so this analytic reference is the only thing that catches it. To
	// stay independent it evaluates the receiver-cosine bound via explicit angles (cos(max(phi-alpha,0)),
	// phi = acos(cosPhi), alpha = asin(sinAlpha)) instead of the library's algebraic expansion
	// cosPhi*cosAlpha + sinPhi*sinAlpha, so an algebra slip there still trips.
	static float analyticChildWeight(const nbl::hlsl::float32_t3& bMin, const nbl::hlsl::float32_t3& bMax,
	                                 float power,
	                                 const nbl::hlsl::float32_t3& x, const nbl::hlsl::float32_t3& n)
	{
		if (!(power > 0.0f)) return 0.0f;

		// Match lightcutTreeChildWeight's mode dispatch. Modes 1/2 short-circuit before any
		// geometry; modes 0/3 share the orientation cone bound and mode 0 adds the distance term.
		if constexpr (Mode == 2u) return 1.0f;
		if constexpr (Mode == 1u) return power;

		const nbl::hlsl::float32_t3 ext{bMax.x - bMin.x, bMax.y - bMin.y, bMax.z - bMin.z};
		const float halfDiagSq = 0.25f * (ext.x*ext.x + ext.y*ext.y + ext.z*ext.z);

		// Importance distance to the NEAREST point of the AABB, floored at the cluster's
		// half-diagonal^2. Matches the library mode-0 distance term. Computed per-axis as
		// max(bMin - x, x - bMax, 0) (independent of how the library writes it).
		const nbl::hlsl::float32_t3 c{0.5f*(bMin.x + bMax.x), 0.5f*(bMin.y + bMax.y), 0.5f*(bMin.z + bMax.z)};
		const nbl::hlsl::float32_t3 dc{c.x - x.x, c.y - x.y, c.z - x.z};
		const float centroidDistSq = dc.x*dc.x + dc.y*dc.y + dc.z*dc.z;
		const nbl::hlsl::float32_t3 dNear{
			std::max(std::max(bMin.x - x.x, x.x - bMax.x), 0.0f),
			std::max(std::max(bMin.y - x.y, x.y - bMax.y), 0.0f),
			std::max(std::max(bMin.z - x.z, x.z - bMax.z), 0.0f)};
		const float minDistSq = dNear.x*dNear.x + dNear.y*dNear.y + dNear.z*dNear.z;
		const float distSq    = std::max(minDistSq, halfDiagSq);

		// Receiver-side cosine UPPER BOUND over the whole bbox: widen the centroid cosine by the bbox
		// angular radius alpha (sin alpha = halfDiag / distToCentroid) and take cos(max(phi - alpha, 0)).
		// orientFactor == 0 doubles as the below-horizon cull.
		const float distRefSq = std::max(centroidDistSq, halfDiagSq);
		const float invLen    = 1.0f / std::sqrt(distRefSq);
		const float cosPhi    = std::clamp((n.x*dc.x + n.y*dc.y + n.z*dc.z) * invLen, -1.0f, 1.0f);
		const float sinAlpha  = std::min(std::sqrt(halfDiagSq) * invLen, 1.0f);
		const float phi       = std::acos(cosPhi);
		const float alpha     = std::asin(sinAlpha);
		const float orientFactor = std::max(std::cos(std::max(phi - alpha, 0.0f)), 0.0f);
		if (!(orientFactor > 0.0f)) return 0.0f;

		if constexpr (Mode == 3u) return power * orientFactor; // orientation only, NO distance
		return power * orientFactor / distSq;                  // mode 0: + inverse-square distance
	}

public:
	CStochasticLightcutTreeGPUTester(const uint32_t testBatchCount) : base_t(testBatchCount, WORKGROUP_SIZE) {}

private:
	LightcutTreeInputValues generateInputTestValues() override
	{
		std::uniform_real_distribution<float> uDist(0.0f, 1.0f);
		LightcutTreeInputValues input;
		input.u = uDist(getRandomEngine());
		return input;
	}

	LightcutTreeTestResults determineExpectedResults(const LightcutTreeInputValues& input) override
	{
		LightcutTreeTestResults expected;
		Executor                executor;
		executor(input, expected);
		return expected;
	}

	bool verifyTestResults(const LightcutTreeTestResults& expected, const LightcutTreeTestResults& actual, const size_t iteration, const uint32_t seed, TestType testType) override
	{
		bool pass = true;

		// BelowPlane scenario: every child sits below the tangent plane. The orientation
		// cone bound (modes 0 and 3) kills them all, so generate() must return the ~0u
		// sentinel with zero pdfs; the generic consistency / "PDFs > 0" / jacobian == 1
		// checks below don't apply and would flag legitimate failure output as a bug.
		// Modes 1 (power) and 2 (uniform) ignore orientation, so the leaves stay alive and
		// the scenario degrades to an ordinary pick -- fall through to the generic suite there.
		if constexpr (kIsBelowPlane && (Mode == 0u || Mode == 3u))
		{
			if (actual.generatedLeafHeap != 0xFFFFFFFFu)
			{
				pass = false;
				printTestFail("LightcutTree(belowPlane)::generatedLeafHeap == ~0u",
					float(0xFFFFFFFFu), float(actual.generatedLeafHeap), iteration, seed, testType, 0.0, 0.0);
			}
			pass &= verifyTestValue("LightcutTree(belowPlane)::forwardPdf",  0.0f, actual.forwardPdf,  iteration, seed, testType, 0.0, 0.0);
			pass &= verifyTestValue("LightcutTree(belowPlane)::backwardPdf", 0.0f, actual.backwardPdf, iteration, seed, testType, 0.0, 0.0);
			return pass;
		}

		if (expected.generatedLeafHeap != actual.generatedLeafHeap)
		{
			pass = false;
			printTestFail(kLeafHeapName, float(expected.generatedLeafHeap), float(actual.generatedLeafHeap), iteration, seed, testType, 0.0, 0.0);
		}
		if (expected.generatedEmitterID != actual.generatedEmitterID)
		{
			pass = false;
			printTestFail(kEmitterIdName, float(expected.generatedEmitterID), float(actual.generatedEmitterID), iteration, seed, testType, 0.0, 0.0);
		}

		VERIFY_FIELDS(pass, expected, actual, iteration, seed, testType,
			FieldCheck{kForwardPdfName,     &R::forwardPdf,     1e-5, 1e-6},
			FieldCheck{kBackwardPdfName,    &R::backwardPdf,    1e-5, 1e-6},
			FieldCheck{kForwardWeightName,  &R::forwardWeight,  1e-5, 1e-6},
			FieldCheck{kBackwardWeightName, &R::backwardWeight, 1e-5, 1e-6},
			FieldCheck{kBboxMinXName,       &R::leafBboxMinX,   1e-5, 1e-6},
			FieldCheck{kBboxMinYName,       &R::leafBboxMinY,   1e-5, 1e-6},
			FieldCheck{kBboxMinZName,       &R::leafBboxMinZ,   1e-5, 1e-6},
			FieldCheck{kBboxMaxXName,       &R::leafBboxMaxX,   1e-5, 1e-6},
			FieldCheck{kBboxMaxYName,       &R::leafBboxMaxY,   1e-5, 1e-6},
			FieldCheck{kBboxMaxZName,       &R::leafBboxMaxZ,   1e-5, 1e-6});
		VERIFY_PDFS_POSITIVE(pass, actual, iteration, seed, testType,
			PdfCheck{kForwardPdfName,  &R::forwardPdf},
			PdfCheck{kBackwardPdfName, &R::backwardPdf});

		// Jacobian == 1 IS the fwd/bwd pdf consistency check ((1/fwd)*bwd == 1 when they match for the
		// picked leaf); the direct pdf-consistency line is kept for a clearer failure message.
		pass &= verifyTestValue(kJacobianName,       1.0f,              actual.jacobianProduct, iteration, seed, testType, 1e-4, 1e-4);
		pass &= verifyTestValue(kPdfConsistencyName, actual.forwardPdf, actual.backwardPdf,     iteration, seed, testType, 1e-5, 1e-6);

		// Depth-2 analytic backward-pdf cross-check: CPU and GPU both run the same library backwardPdf,
		// so a bug inside lightcutTreeChildWeight or the heap-walk would be invisible to expected==actual.
		// Rebuild the synthetic tree in C++ and multiply per-level conditional weights root->leaf via the
		// hand-rolled analyticChildWeight().
		if constexpr (kIsDepth2)
		{
			const nbl::hlsl::float32_t3 groupC[4] = {
				{ 2.0f, 1.5f,  2.0f}, {-2.0f, 1.5f,  2.0f},
				{ 2.0f, 2.5f, -2.0f}, {-2.0f, 3.5f, -2.0f}
			};
			const nbl::hlsl::float32_t3 leafOff[4] = {
				{ 0.10f, 0.0f,  0.10f}, {-0.10f, 0.0f,  0.10f},
				{ 0.10f, 0.0f, -0.10f}, {-0.10f, 0.0f, -0.10f}
			};
			constexpr float kHalfExt = 0.02f;
			const nbl::hlsl::float32_t3 x{0.0f, 0.0f, 0.0f};
			const nbl::hlsl::float32_t3 n{0.0f, 1.0f, 0.0f};

			// Rebuild root + leaf-parent wide-nodes.
			struct Child { nbl::hlsl::float32_t3 bMin, bMax; float power; };
			Child root[4];
			for (uint32_t g = 0u; g < 4u; ++g)
			{
				nbl::hlsl::float32_t3 mn{1e30f, 1e30f, 1e30f}, mx{-1e30f, -1e30f, -1e30f};
				for (uint32_t s = 0u; s < 4u; ++s)
				{
					const auto c = nbl::hlsl::float32_t3{groupC[g].x + leafOff[s].x, groupC[g].y + leafOff[s].y, groupC[g].z + leafOff[s].z};
					mn = nbl::hlsl::float32_t3{std::min(mn.x, c.x - kHalfExt), std::min(mn.y, c.y - kHalfExt), std::min(mn.z, c.z - kHalfExt)};
					mx = nbl::hlsl::float32_t3{std::max(mx.x, c.x + kHalfExt), std::max(mx.y, c.y + kHalfExt), std::max(mx.z, c.z + kHalfExt)};
				}
				root[g].bMin = mn; root[g].bMax = mx; root[g].power = 4.0f;
			}

			// heap layout: root=0, wide-nodes 1..4 (=leaf parents), leaves 5..20
			const uint32_t leafHeap = actual.generatedLeafHeap;
			if (leafHeap >= 5u && leafHeap <= 20u)
			{
				const uint32_t leafArr   = leafHeap - 5u;
				const uint32_t parentSlot = leafArr / 4u;          // 0..3 -- which root child
				const uint32_t leafSlot   = leafArr % 4u;          // 0..3 -- which child of that parent

				// Root-level pdf: weight(parentSlot) / sum(all 4 children)
				float rw[4], rwSum = 0.0f;
				for (uint32_t g = 0; g < 4u; ++g)
				{
					rw[g] = analyticChildWeight(root[g].bMin, root[g].bMax, root[g].power, x, n);
					rwSum += rw[g];
				}
				const float pRoot = rw[parentSlot] / rwSum;

				// Leaf-level pdf: among the 4 leaves of parentSlot, weight(leafSlot) / sum
				float lw[4], lwSum = 0.0f;
				for (uint32_t s = 0u; s < 4u; ++s)
				{
					const auto c    = nbl::hlsl::float32_t3{groupC[parentSlot].x + leafOff[s].x, groupC[parentSlot].y + leafOff[s].y, groupC[parentSlot].z + leafOff[s].z};
					const auto lMin = nbl::hlsl::float32_t3{c.x - kHalfExt, c.y - kHalfExt, c.z - kHalfExt};
					const auto lMax = nbl::hlsl::float32_t3{c.x + kHalfExt, c.y + kHalfExt, c.z + kHalfExt};
					lw[s] = analyticChildWeight(lMin, lMax, 1.0f, x, n);
					lwSum += lw[s];
				}
				const float pLeaf = lw[leafSlot] / lwSum;

				const float analyticPdf = pRoot * pLeaf;
				pass &= verifyTestValue("LightcutTree(depth2)::backwardPdf vs analytic",
					analyticPdf, actual.backwardPdf, iteration, seed, testType, 1e-4, 1e-5);
				pass &= verifyTestValue("LightcutTree(depth2)::forwardPdf vs analytic",
					analyticPdf, actual.forwardPdf,  iteration, seed, testType, 1e-4, 1e-5);
			}
		}

		// DistanceFalloff: independent analytic pdf check against an external formula, not the sampler's
		// own evaluation.
		if constexpr (kIsDistFalloff)
		{
			// Two equal-power, fully-facing point leaves at distance 1 and 2. Only mode 0's
			// inverse-square term discriminates them (4/5, 1/5); modes 1 (power), 2 (uniform)
			// and 3 (orientation only, both orientFactor == 1) all collapse to a 1/2, 1/2 split.
			constexpr float kClosePdf = (Mode == 0u) ? 0.8f : 0.5f;
			constexpr float kFarPdf   = (Mode == 0u) ? 0.2f : 0.5f;
			const char* closeName = (Mode == 0u) ? "LightcutTree(distFalloff)::close pdf == 4/5" : "LightcutTree(distFalloff)::close pdf == 1/2";
			const char* farName   = (Mode == 0u) ? "LightcutTree(distFalloff)::far pdf == 1/5"   : "LightcutTree(distFalloff)::far pdf == 1/2";
			if (actual.generatedLeafHeap == 1u)
				pass &= verifyTestValue(closeName, kClosePdf, actual.forwardPdf, iteration, seed, testType, 1e-3, 1e-3);
			else if (actual.generatedLeafHeap == 2u)
				pass &= verifyTestValue(farName,   kFarPdf,   actual.forwardPdf, iteration, seed, testType, 1e-3, 1e-3);
			else
			{
				// Padding (heap 3/4) has zero power; should be unreachable. Flag
				// as failure rather than silently passing.
				pass = false;
				printTestFail("LightcutTree(distFalloff)::picked padding leaf",
					float(1u), float(actual.generatedLeafHeap), iteration, seed, testType, 0.0, 0.0);
			}
		}

		return pass;
	}
};

// multi/single are mode-agnostic consistency checks, pinned to the library default weight mode.
using CStochasticLightcutTreeMultiGPUTester  = CStochasticLightcutTreeGPUTester<LightcutTreeMultiLeafExecutor,  NBL_LIGHTCUT_TREE_WEIGHT_MODE>;
using CStochasticLightcutTreeSingleGPUTester = CStochasticLightcutTreeGPUTester<LightcutTreeSingleLeafExecutor, NBL_LIGHTCUT_TREE_WEIGHT_MODE>;
// Geometric scenarios are instantiated once per weight mode (0..3) by main.cpp.
template<uint32_t Mode> 
using CStochasticLightcutTreeBelowPlaneGPUTester   = CStochasticLightcutTreeGPUTester<LightcutTreeBelowPlaneExecutor<Mode>,      Mode>;
template<uint32_t Mode> 
using CStochasticLightcutTreeDistFalloffGPUTester  = CStochasticLightcutTreeGPUTester<LightcutTreeDistanceFalloffExecutor<Mode>, Mode>;
template<uint32_t Mode> 
using CStochasticLightcutTreeInflatedBboxGPUTester = CStochasticLightcutTreeGPUTester<LightcutTreeInflatedBboxExecutor<Mode>,    Mode>;
template<uint32_t Mode> 
using CStochasticLightcutTreeDepth2GPUTester       = CStochasticLightcutTreeGPUTester<LightcutTreeDepth2Executor<Mode>,          Mode>;

#endif
