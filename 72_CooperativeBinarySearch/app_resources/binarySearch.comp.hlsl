// Copyright (C) 2024-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#pragma wave shader_stage(compute)

#include "common.h"

#include "nbl/builtin/hlsl/glsl_compat/subgroup_ballot.hlsl"

using namespace nbl::hlsl;

[[vk::push_constant]] PushConstants Constants;
[[vk::binding(0)]] StructuredBuffer<uint> Histogram;
[[vk::binding(1)]] RWStructuredBuffer<uint> Output;


uint getNextPowerOfTwo(uint number) {
	return 2 << firstbithigh(number - 1);
}

uint getLaneWithFirstBitSet(bool condition) {
	uint4 ballot = WaveActiveBallot(condition);
	if (all(ballot == 0)) {
		return WaveGetLaneCount();
	}
	return nbl::hlsl::glsl::subgroupBallotFindLSB(ballot);
}

// findValue must be the same across the entire wave
// Could use something like WaveReadFirstLane to be fully sure
uint binarySearchLowerBoundFindValue(uint findValue, StructuredBuffer<uint> searchBuffer, uint searchBufferSize) {
	uint lane = WaveGetLaneIndex();
	
	uint left = 0;
	uint right = searchBufferSize - 1;

	uint32_t range = getNextPowerOfTwo(right - left);
	// do pivots as long as we can't coalesced load
	while (range > WaveGetLaneCount())
	{
		// there must be at least 1 gap between subsequent pivots 
		const uint32_t step = range / WaveGetLaneCount(); 
		const uint32_t halfStep = step >> 1;
		const uint32_t pivotOffset = lane * step+halfStep;
		const uint32_t pivotIndex = left + pivotOffset;

		uint4 notGreaterPivots = WaveActiveBallot(pivotIndex < right && !(findValue < searchBuffer[pivotIndex]));
		uint partition = nbl::hlsl::glsl::subgroupBallotBitCount(notGreaterPivots);
		// only move left if needed
		if (partition != 0)
			left += partition * step - halfStep;
		// if we go into final half partition, the range becomes less too
		range = partition != WaveGetLaneCount() ? step : halfStep;
	}

	uint threadSearchIndex = left + lane;
	bool laneValid = threadSearchIndex < searchBufferSize;
	uint histAtIndex = laneValid ? searchBuffer[threadSearchIndex] : -1;
	uint firstLaneGreaterThan = getLaneWithFirstBitSet(histAtIndex > findValue);

	return left + firstLaneGreaterThan - 1;
}

static const uint32_t GroupsharedSize = WorkgroupSize;
groupshared uint shared_groupSearchBufferMinIndex;
groupshared uint shared_groupSearchBufferMaxIndex;
groupshared uint shared_groupSearchValues[WorkgroupSize];

// Binary search using the entire workgroup, making it log32 or log64 (every iteration, the possible set of 
// values is divided by the number of lanes in a wave)
uint binarySearchLowerBoundCooperative(uint groupIndex, uint groupThread, StructuredBuffer<uint> searchBuffer, uint searchBufferSize) {
	uint minSearchValue = groupIndex.x * GroupsharedSize;
	uint maxSearchValue = ((groupIndex.x + 1) * GroupsharedSize) - 1;

	// On each workgroup, two subgroups do the search
	// - One searches for the minimum, the other searches for the maximum
	// - Store the minimum and maximum on groupshared memory, then do a barrier
	uint wave = groupThread / WaveGetLaneCount();
	if (wave < 2) {
		uint search = wave == 0 ? minSearchValue : maxSearchValue;
		uint searchResult = binarySearchLowerBoundFindValue(search, searchBuffer, searchBufferSize);
		if (WaveIsFirstLane()) {
			if (wave == 0) shared_groupSearchBufferMinIndex = searchResult;
			else shared_groupSearchBufferMaxIndex = searchResult;
		}
	}
	GroupMemoryBarrierWithGroupSync();

	// Since every instance has at least one triangle, we know that having workgroup values 
	// for each value in the range of minimum to maximum will suffice.

	// Write every value in the range to groupshared memory and barrier.
	uint idx = shared_groupSearchBufferMinIndex + groupThread.x;
	if (idx <= shared_groupSearchBufferMaxIndex) {
		shared_groupSearchValues[groupThread.x] = searchBuffer[idx];
	}
	GroupMemoryBarrierWithGroupSync();

	uint maxValueIndex = shared_groupSearchBufferMaxIndex - shared_groupSearchBufferMinIndex;

	uint searchValue = minSearchValue + groupThread;
	uint currentSearchValueIndex = 0;
	uint laneValue = shared_groupSearchBufferMaxIndex;
	while (currentSearchValueIndex <= maxValueIndex) {
		uint curValue = shared_groupSearchValues[currentSearchValueIndex];
		if (curValue > searchValue) {
			laneValue = shared_groupSearchBufferMinIndex + currentSearchValueIndex - 1;
			break;
		}
		currentSearchValueIndex ++;
	}

	return laneValue;
}

[numthreads(WorkgroupSize,1,1)]
void main(const uint3 thread : SV_DispatchThreadID, const uint3 groupThread : SV_GroupThreadID, const uint3 group : SV_GroupID)
{
    Output[thread.x] = binarySearchLowerBoundCooperative(group.x, groupThread.x, Histogram, Constants.EntityCount);
}