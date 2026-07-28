#ifndef _PATHTRACER_40_PT_CONFIG_INCLUDED_
#define _PATHTRACER_40_PT_CONFIG_INCLUDED_

// Single home for every example-40 path-tracer compile-time knob.

// ---- variant selectors (CMake passes -D per pipeline variant; these are only the fallbacks) ----
// 0 = OBB clipped spherical pyramid, 1 = triangle uniform-area, 2 = triangle Arvo, 3 = triangle projected.
#ifndef NBL_NEE_LEAF_MODE
#define NBL_NEE_LEAF_MODE 0
#endif

// 1 in the deferred raygen variants, which only emit requests. The NEE compute pass compiles without
// it and gets the full estimator.
#ifndef NBL_NEE_DEFERRED
#define NBL_NEE_DEFERRED 0
#endif

// Emitter-selection proposal: 1 = power alias table (O(1)), 0 = stochastic light-cut tree descent.
#ifndef NBL_NEE_USE_ALIAS
#define NBL_NEE_USE_ALIAS 1
#endif

#define NBL_MIS_MODE_NEE_ONLY 0
#define NBL_MIS_MODE_BXDF_ONLY 1
#define NBL_MIS_MODE_BOTH 2
#ifndef NBL_MIS_MODE
#define NBL_MIS_MODE NBL_MIS_MODE_BOTH
#endif

// ---- NEE estimator knobs ----
// Visibility: 0 = two rays (emitter-geometry rejection + identity-skip shadow), 1 = one closest-hit ray.
// Triangle leaves default to the single closest-hit ray, which confirms the exact winning triangle via
// resolveHitEmitterID. OBB stays two-ray for the cheap per-emitter-TLAS early rejection.
#ifndef NBL_NEE_SINGLE_RAY
#if NBL_NEE_LEAF_MODE != 0
#define NBL_NEE_SINGLE_RAY 1
#else
#define NBL_NEE_SINGLE_RAY 0
#endif
#endif

#ifndef NEE_RIS_CANDIDATES
#define NEE_RIS_CANDIDATES 1
#endif
#ifndef NEE_LIGHT_CANDIDATES
#define NEE_LIGHT_CANDIDATES 1
#endif

// Geometry term in the RIS resample target: 2 = orient * bounded projected solid angle, 1 = orient/dist^2,
// 0 = orient only.
#ifndef NEE_GEOMTARGET_DISTANCE
#define NEE_GEOMTARGET_DISTANCE 2
#endif

// OBB silhouette fit (OBB, non-deferred only). 1 = spherical rotating-caliper rectangle, else longest-edge.
#if NBL_NEE_LEAF_MODE == 0 && !NBL_NEE_DEFERRED
#ifndef NBL_NEE_PROJECTED_SPHRECT
#define NBL_NEE_PROJECTED_SPHRECT 1
#endif
#ifndef NBL_NEE_CALIPER
#define NBL_NEE_CALIPER 0
#endif
#endif // OBB non-deferred

// ---- light-cut tree descent. Overrides the builtin stochastic_lightcut_tree.hlsl defaults, which works
// only because this file is included before it. ----
// Weight mode: 0 power*orient/dist^2, 1 power, 2 uniform, 3 power*orient, 4 power*projSolidAngle.
#ifndef NBL_LIGHTCUT_TREE_WEIGHT_MODE
#define NBL_LIGHTCUT_TREE_WEIGHT_MODE 4
#endif
#ifndef NBL_LIGHTCUT_TREE_PDF_FLOOR_ENABLED
#define NBL_LIGHTCUT_TREE_PDF_FLOOR_ENABLED 0
#endif
#ifndef NBL_LIGHTCUT_TREE_PDF_FLOOR
#define NBL_LIGHTCUT_TREE_PDF_FLOOR 1e-2
#endif
#ifndef NBL_LIGHTCUT_TREE_STOP_MAX_RATIO_ENABLED
#define NBL_LIGHTCUT_TREE_STOP_MAX_RATIO_ENABLED 0
#endif
#ifndef NBL_LIGHTCUT_TREE_STOP_MAX_RATIO
#define NBL_LIGHTCUT_TREE_STOP_MAX_RATIO 0.6
#endif

// ---- alias-index bit widths. C++ builder and HLSL sampler must agree, more than 2^Log2N leaves
// overflows the packed index and corrupts selection. ----
#ifndef NBL_LIGHTTREE_ALIAS_LOG2N_OBB
#define NBL_LIGHTTREE_ALIAS_LOG2N_OBB 16u
#endif
#ifndef NBL_LIGHTTREE_ALIAS_LOG2N_TRI
#define NBL_LIGHTTREE_ALIAS_LOG2N_TRI 22u
#endif
#ifndef NBL_LIGHTTREE_ALIAS_LOG2N
#if defined(NBL_NEE_LEAF_MODE) && (NBL_NEE_LEAF_MODE != 0)
#define NBL_LIGHTTREE_ALIAS_LOG2N NBL_LIGHTTREE_ALIAS_LOG2N_TRI
#else
#define NBL_LIGHTTREE_ALIAS_LOG2N NBL_LIGHTTREE_ALIAS_LOG2N_OBB
#endif
#endif

// ---- diagnostics (skew timings; keep 0 for benchmark runs) ----
#ifndef NBL_NEE_STATS
#define NBL_NEE_STATS 0
#endif
#ifndef NBL_NEE_PROPOSAL_PROBE
#define NBL_NEE_PROPOSAL_PROBE 0
#endif

#endif // _PATHTRACER_40_PT_CONFIG_INCLUDED_
