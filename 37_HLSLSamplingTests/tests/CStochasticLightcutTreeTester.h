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

	// Only the renderer's weight mode (LightcutTestWeightMode) is exercised. What pins it, beyond
	// the universal CPU==GPU / jacobian==1 / pdf>0 invariants:
	//   belowPlane:   orientFactor culls every child -> the ~0u sentinel with zero pdfs.
	//   distFalloff:  the bounding-sphere solid angle falls off as 1/dist^2 for small clusters,
	//                 giving a 4/5, 1/5 split that power-only and uniform weighting cannot produce.
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

	static float analyticChildWeight(const nbl::hlsl::float32_t3& bMin, const nbl::hlsl::float32_t3& bMax,
	                                 float power,
	                                 const nbl::hlsl::float32_t3& x, const nbl::hlsl::float32_t3& n)
	{
		if (!(power > 0.0f)) return 0.0f;

		const nbl::hlsl::float32_t3 ext{bMax.x - bMin.x, bMax.y - bMin.y, bMax.z - bMin.z};
		const double halfDiagSq = 0.25*(double(ext.x)*ext.x + double(ext.y)*ext.y + double(ext.z)*ext.z);

		const nbl::hlsl::float32_t3 c{0.5f*(bMin.x + bMax.x), 0.5f*(bMin.y + bMax.y), 0.5f*(bMin.z + bMax.z)};
		const nbl::hlsl::float32_t3 dc{c.x - x.x, c.y - x.y, c.z - x.z};
		const double centroidDistSq = double(dc.x)*dc.x + double(dc.y)*dc.y + double(dc.z)*dc.z;

		// Receiver-side cosine UPPER BOUND over the whole bbox: widen the centroid cosine by the bbox
		// angular radius alpha (sin alpha = halfDiag / distToCentroid) and take cos(max(phi - alpha, 0)).
		// orientFactor == 0 doubles as the below-horizon cull.
		const double distRefSq = std::max(centroidDistSq, halfDiagSq);
		const double invLen    = 1.0/std::sqrt(distRefSq);
		const double cosPhi    = std::clamp((double(n.x)*dc.x + double(n.y)*dc.y + double(n.z)*dc.z) * invLen, -1.0, 1.0);
		const double sinAlpha  = std::min(std::sqrt(halfDiagSq) * invLen, 1.0);
		const double phi       = std::acos(cosPhi);
		const double alpha     = std::asin(sinAlpha);
		const double orientFactor = std::max(std::cos(std::max(phi - alpha, 0.0)), 0.0);
		if (!(orientFactor > 0.0)) return 0.0f;

		// Bounding-sphere solid angle up to the 2pi that cancels in the sibling sum.
		return float(double(power) * (1.0 - std::cos(alpha)) * orientFactor);
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

		// BelowPlane scenario: every child sits below the tangent plane, so the orientation
		// cone bound kills them all and generate() must return the ~0u sentinel with zero
		// pdfs. The generic consistency / "PDFs > 0" / jacobian == 1 checks below don't apply
		// and would flag legitimate failure output as a bug.
		if constexpr (kIsBelowPlane)
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
				const uint32_t parentSlot = leafArr / 4u;          // 0..3, which root child
				const uint32_t leafSlot   = leafArr % 4u;          // 0..3, which child of that parent

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
			if (actual.generatedLeafHeap == 1u)
				pass &= verifyTestValue("LightcutTree(distFalloff)::close pdf == 4/5", 0.8f, actual.forwardPdf, iteration, seed, testType, 1e-3, 1e-3);
			else if (actual.generatedLeafHeap == 2u)
				pass &= verifyTestValue("LightcutTree(distFalloff)::far pdf == 1/5",   0.2f, actual.forwardPdf, iteration, seed, testType, 1e-3, 1e-3);
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

using CStochasticLightcutTreeMultiGPUTester  = CStochasticLightcutTreeGPUTester<LightcutTreeMultiLeafExecutor,  LightcutTestWeightMode>;
using CStochasticLightcutTreeSingleGPUTester = CStochasticLightcutTreeGPUTester<LightcutTreeSingleLeafExecutor, LightcutTestWeightMode>;
using CStochasticLightcutTreeBelowPlaneGPUTester   = CStochasticLightcutTreeGPUTester<LightcutTreeBelowPlaneExecutor<LightcutTestWeightMode>,      LightcutTestWeightMode>;
using CStochasticLightcutTreeDistFalloffGPUTester  = CStochasticLightcutTreeGPUTester<LightcutTreeDistanceFalloffExecutor<LightcutTestWeightMode>, LightcutTestWeightMode>;
using CStochasticLightcutTreeInflatedBboxGPUTester = CStochasticLightcutTreeGPUTester<LightcutTreeInflatedBboxExecutor<LightcutTestWeightMode>,    LightcutTestWeightMode>;
using CStochasticLightcutTreeDepth2GPUTester       = CStochasticLightcutTreeGPUTester<LightcutTreeDepth2Executor<LightcutTestWeightMode>,          LightcutTestWeightMode>;

#endif
