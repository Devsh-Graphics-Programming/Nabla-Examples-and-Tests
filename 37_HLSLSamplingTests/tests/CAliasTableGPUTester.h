#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_C_ALIAS_TABLE_GPU_TESTER_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_C_ALIAS_TABLE_GPU_TESTER_INCLUDED_

#include "nbl/examples/examples.hpp"
#include "app_resources/common/alias_table.hlsl"
#include "nbl/examples/Tester/ITester.h"
#include "SamplerTestHelpers.h"

// Shared GPU correctness harness for the packed alias variants. Labels for
// failed-field messages are selected from the Executor type at compile time.
template<typename Executor>
class CPackedAliasTableGPUTester final : public ITester<AliasTableInputValues, AliasTableTestResults, Executor>
{
	using base_t = ITester<AliasTableInputValues, AliasTableTestResults, Executor>;
	using R      = AliasTableTestResults;

	using typename base_t::TestType;
	using base_t::getRandomEngine;
	using base_t::verifyTestValue;
	using base_t::printTestFail;

	static constexpr bool kIsA = std::is_same_v<Executor, PackedAliasATestExecutor>;
	static constexpr const char* kGeneratedIdxName     = kIsA ? "PackedAliasA::generatedIndex"     : "PackedAliasB::generatedIndex";
	static constexpr const char* kForwardPdfName       = kIsA ? "PackedAliasA::forwardPdf"         : "PackedAliasB::forwardPdf";
	static constexpr const char* kBackwardPdfName      = kIsA ? "PackedAliasA::backwardPdf"        : "PackedAliasB::backwardPdf";
	static constexpr const char* kForwardWeightName    = kIsA ? "PackedAliasA::forwardWeight"      : "PackedAliasB::forwardWeight";
	static constexpr const char* kBackwardWeightName   = kIsA ? "PackedAliasA::backwardWeight"     : "PackedAliasB::backwardWeight";
	static constexpr const char* kJacobianName         = kIsA ? "PackedAliasA::jacobianProduct"    : "PackedAliasB::jacobianProduct";
	static constexpr const char* kPdfConsistencyName   = kIsA ? "PackedAliasA::pdf consistency"    : "PackedAliasB::pdf consistency";
	static constexpr const char* kWeightConsistencyName = kIsA ? "PackedAliasA::weight consistency" : "PackedAliasB::weight consistency";

public:
	CPackedAliasTableGPUTester(const uint32_t testBatchCount) : base_t(testBatchCount, WORKGROUP_SIZE) {}

private:
	AliasTableInputValues generateInputTestValues() override
	{
		std::uniform_real_distribution<float> uDist(0.0f, 1.0f);

		AliasTableInputValues input;
		input.u = uDist(getRandomEngine());
		return input;
	}

	AliasTableTestResults determineExpectedResults(const AliasTableInputValues& input) override
	{
		AliasTableTestResults expected;
		Executor              executor;
		executor(input, expected);
		return expected;
	}

	bool verifyTestResults(const AliasTableTestResults& expected, const AliasTableTestResults& actual, const size_t iteration, const uint32_t seed, TestType testType) override
	{
		bool pass = true;

		if (expected.generatedIndex != actual.generatedIndex)
		{
			pass = false;
			printTestFail(kGeneratedIdxName, float(expected.generatedIndex), float(actual.generatedIndex), iteration, seed, testType, 0.0, 0.0);
		}

		VERIFY_FIELDS(pass, expected, actual, iteration, seed, testType,
			FieldCheck{kForwardPdfName,     &R::forwardPdf,     1e-5, 1e-6},
			FieldCheck{kBackwardPdfName,    &R::backwardPdf,    1e-5, 1e-6},
			FieldCheck{kForwardWeightName,  &R::forwardWeight,  1e-5, 1e-6},
			FieldCheck{kBackwardWeightName, &R::backwardWeight, 1e-5, 1e-6});
		VERIFY_PDFS_POSITIVE(pass, actual, iteration, seed, testType,
			PdfCheck{kForwardPdfName,  &R::forwardPdf},
			PdfCheck{kBackwardPdfName, &R::backwardPdf});

		pass &= verifyTestValue(kJacobianName,          1.0f, actual.jacobianProduct, iteration, seed, testType, 1e-4, 1e-4);
		pass &= verifyTestValue(kPdfConsistencyName,    actual.forwardPdf, actual.backwardPdf, iteration, seed, testType, 1e-7, 1e-7);
		pass &= verifyTestValue(kWeightConsistencyName, actual.forwardWeight, actual.backwardWeight, iteration, seed, testType, 1e-7, 1e-7);

		return pass;
	}
};

using CPackedAliasAGPUTester = CPackedAliasTableGPUTester<PackedAliasATestExecutor>;
using CPackedAliasBGPUTester = CPackedAliasTableGPUTester<PackedAliasBTestExecutor>;

#endif
