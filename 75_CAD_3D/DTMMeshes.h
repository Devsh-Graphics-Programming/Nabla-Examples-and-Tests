#pragma once

// pyramid
#if 1
core::vector<TriangleMeshVertex> DTMMainMeshVertices = {
	{ float64_t3(0.0, 100.0, 0.0) },
	{ float64_t3(-200.0, 10.0, -200.0) },
	{ float64_t3(200.0, 10.0, -100.0) },
	{ float64_t3(0.0, 100.0, 0.0) },
	{ float64_t3(200.0, 10.0, -100.0) },
	{ float64_t3(200.0, -20.0, 200.0) },
	{ float64_t3(0.0, 100.0, 0.0) },
	{ float64_t3(200.0, -20.0, 200.0) },
	{ float64_t3(-200.0, 10.0, 200.0) },
	{ float64_t3(0.0, 100.0, 0.0) },
	{ float64_t3(-200.0, 10.0, 200.0) },
	{ float64_t3(-200.0, 10.0, -200.0) },
};

core::vector<uint32_t> DTMMainMeshIndices = {
	0, 1, 2,
	3, 4, 5,
	6, 7, 8,
	9, 10, 11
};
#endif