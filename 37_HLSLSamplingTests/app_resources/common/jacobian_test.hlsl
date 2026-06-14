#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_JACOBIAN_TEST_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_JACOBIAN_TEST_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/promote.hlsl>

using namespace nbl::hlsl;

// Negative sentinels signal "skipped" to the host verifier; the value encodes the reason.
static const float32_t JACOBIAN_SKIP_U_DOMAIN             = -1.0f;
static const float32_t JACOBIAN_SKIP_CREASE               = -2.0f;
static const float32_t JACOBIAN_SKIP_HEMI_BOUNDARY        = -3.0f;
static const float32_t JACOBIAN_SKIP_BWD_PDF_RANGE        = -4.0f;
static const float32_t JACOBIAN_SKIP_CODOMAIN_SINGULARITY = -5.0f;


template<typename Sampler, uint32_t DomainDim, uint32_t CodomainDim>
struct ForwardJacobianMeasure;

// Signed step that stays inside [0,1]: flip direction when u is in the upper half so u +/- eps
// never overshoots the domain. Magnitude is what matters (the stencil results take abs/length).
template<typename T>
T signedEps(T u, T eps)
{
   return u > T(0.5) ? -eps : eps;
}

template<typename Sampler>
struct ForwardJacobianMeasure<Sampler, 1, 1>
{
   using scalar_type   = typename Sampler::scalar_type;
   using domain_type   = typename Sampler::domain_type;
   using codomain_type = typename Sampler::codomain_type;
   using cache_type    = typename Sampler::cache_type;

   static scalar_type compute(Sampler _sampler, domain_type u, scalar_type eps, codomain_type L)
   {
      cache_type c;
      const codomain_type L_x = _sampler.generate(u + signedEps<scalar_type>(u, eps), c);
      return nbl::hlsl::abs<scalar_type>(L_x - L) / eps;
   }
};

template<typename Sampler>
struct ForwardJacobianMeasure<Sampler, 2, 2>
{
   using scalar_type   = typename Sampler::scalar_type;
   using domain_type   = typename Sampler::domain_type;
   using codomain_type = typename Sampler::codomain_type;
   using cache_type    = typename Sampler::cache_type;

   static scalar_type compute(Sampler _sampler, domain_type u, scalar_type eps, codomain_type L)
   {
      domain_type u_x = u;
      u_x[0] += signedEps<scalar_type>(u[0], eps);
      domain_type u_y = u;
      u_y[1] += signedEps<scalar_type>(u[1], eps);
      cache_type c;
      const codomain_type L_x = _sampler.generate(u_x, c);
      const codomain_type L_y = _sampler.generate(u_y, c);
      using matrix2_type      = matrix<scalar_type, 2, 2>;
      const scalar_type det   = nbl::hlsl::determinant<matrix2_type>(matrix2_type(L_x - L, L_y - L));
      return nbl::hlsl::abs<scalar_type>(det) / (eps * eps);
   }
};

template<typename Sampler>
struct ForwardJacobianMeasure<Sampler, 2, 3>
{
   using scalar_type   = typename Sampler::scalar_type;
   using domain_type   = typename Sampler::domain_type;
   using codomain_type = typename Sampler::codomain_type;
   using cache_type    = typename Sampler::cache_type;

   static scalar_type compute(Sampler _sampler, domain_type u, scalar_type eps, codomain_type L)
   {
      domain_type u_x = u;
      u_x[0] += signedEps<scalar_type>(u[0], eps);
      domain_type u_y = u;
      u_y[1] += signedEps<scalar_type>(u[1], eps);
      cache_type c;
      const codomain_type L_x = _sampler.generate(u_x, c);
      const codomain_type L_y = _sampler.generate(u_y, c);
      return nbl::hlsl::length(nbl::hlsl::cross(L_x - L, L_y - L)) / (eps * eps);
   }
};

// 3D domain: stencil perturbs u[0] and u[1] only, so the (2,3) body applies unchanged.
template<typename Sampler>
struct ForwardJacobianMeasure<Sampler, 3, 3> : ForwardJacobianMeasure<Sampler, 2, 3>
{
};


template<typename Sampler, uint32_t DomainDim>
struct DomainMarginCheck;

template<typename Sampler>
struct DomainMarginCheck<Sampler, 1>
{
   using scalar_type = typename Sampler::scalar_type;
   using domain_type = typename Sampler::domain_type;
   static bool outsideMargin(domain_type u, scalar_type margin)
   {
      return u < margin || u > scalar_type(1) - margin;
   }
};

template<typename Sampler>
struct DomainMarginCheck<Sampler, 2>
{
   using scalar_type = typename Sampler::scalar_type;
   using domain_type = typename Sampler::domain_type;
   static bool outsideMargin(domain_type u, scalar_type margin)
   {
      return u[0] < margin || u[0] > scalar_type(1) - margin || u[1] < margin || u[1] > scalar_type(1) - margin;
   }
};

// 3D domain: forward stencil only perturbs u[0] and u[1], so u[2] is irrelevant and (2) applies.
template<typename Sampler>
struct DomainMarginCheck<Sampler, 3> : DomainMarginCheck<Sampler, 2>
{
};

enum JacobianMode : uint32_t
{
   JACOBIAN_PLAIN             = 0,
   JACOBIAN_CONCENTRIC        = 1, // + concentric crease skip
   JACOBIAN_CONCENTRIC_UXFOLD = 2  // + crease + u.x=0.5 hemi-boundary skip
};

// marginFactor scales the u-domain skip to marginFactor * eps. Use > 1 only for samplers whose
// stencil bias extends past a single eps-step (e.g. Arvo spherical triangle: sinZ ~ sqrt(u.y)
// gives O(h/u.y) forward-diff bias, so u.y in [0, k*eps] must be skipped).
template<uint32_t Mode, typename Sampler>
float32_t computeJacobianProduct(Sampler _sampler, typename Sampler::domain_type u, float32_t eps, float32_t marginFactor)
{
   using scalar_type   = typename Sampler::scalar_type;
   using domain_type   = typename Sampler::domain_type;
   using codomain_type = typename Sampler::codomain_type;
   using cache_type    = typename Sampler::cache_type;

   NBL_IF_CONSTEXPR(Mode != JACOBIAN_PLAIN)
   {
      // Cast via float32_t2 so this block typechecks for scalar / vec2 / vec3 domains alike
      // (HLSL splats scalars, identity on vec2, .xy on vec3). 1D samplers never reach here.
      const float32_t2 uxy = (float32_t2)u;
      const float32_t ux   = uxy.x;
      const float32_t uy   = uxy.y;

      NBL_IF_CONSTEXPR(Mode == JACOBIAN_CONCENTRIC_UXFOLD)
      {
         if (nbl::hlsl::abs(ux - float32_t(0.5)) <= float32_t(2e-3))
            return JACOBIAN_SKIP_HEMI_BOUNDARY;
      }

      const bool uxFold = (Mode == JACOBIAN_CONCENTRIC_UXFOLD);
      // Empirical: the concentric C0 crease's stencil bias spreads wider than the 2*eps geometric
      // straddle band. Non-uxFold 6e-3 covers the disk-center residual for Projected samplers;
      // uxFold 1e-2 accounts for the doubled local_ux rate when u.x is folded.
      const float32_t creaseBand = uxFold ? float32_t(1e-2) : float32_t(6e-3);
      const float32_t local_ux   = uxFold ? nbl::hlsl::abs(float32_t(2) * ux - float32_t(1)) : ux;
      const float32_t a          = float32_t(2) * local_ux - float32_t(1);
      const float32_t b          = float32_t(2) * uy - float32_t(1);
      if (nbl::hlsl::abs(nbl::hlsl::abs(a) - nbl::hlsl::abs(b)) <= creaseBand)
         return JACOBIAN_SKIP_CREASE;
   }

   using margin_check_type = DomainMarginCheck<Sampler, vector_traits<domain_type>::Dimension>;
   if (margin_check_type::outsideMargin(u, scalar_type(eps * marginFactor)))
      return JACOBIAN_SKIP_U_DOMAIN;

   // Generate on a copy: some samplers mutate u through NBL_REF_ARG (e.g. ProjectedSphere
   // consumes u.z for hemisphere selection), and the perturbations below need the original u.
   cache_type cache;
   domain_type uGen      = u;
   const codomain_type L = _sampler.generate(uGen, cache);
   const scalar_type pdf = _sampler.forwardPdf(uGen, cache);

   using measure_type        = ForwardJacobianMeasure<Sampler, vector_traits<domain_type>::Dimension, vector_traits<codomain_type>::Dimension>;
   const scalar_type measure = measure_type::compute(_sampler, u, scalar_type(eps), L);

   return pdf * measure;
}


template<typename Sampler, uint32_t DomainDim, uint32_t CodomainDim>
struct InverseJacobianMeasure;

template<typename Sampler>
struct InverseJacobianMeasure<Sampler, 2, 2>
{
   using scalar_type   = typename Sampler::scalar_type;
   using domain_type   = typename Sampler::domain_type;
   using codomain_type = typename Sampler::codomain_type;

   static scalar_type compute(Sampler _sampler, codomain_type x, scalar_type eps)
   {
      const scalar_type twoEps = scalar_type(2) * eps;
      codomain_type x0_lo      = x;
      x0_lo[0] -= eps;
      codomain_type x0_hi = x;
      x0_hi[0] += eps;
      codomain_type x1_lo = x;
      x1_lo[1] -= eps;
      codomain_type x1_hi = x;
      x1_hi[1] += eps;
      domain_type u0_lo       = _sampler.generateInverse(x0_lo);
      domain_type u0_hi       = _sampler.generateInverse(x0_hi);
      domain_type u1_lo       = _sampler.generateInverse(x1_lo);
      domain_type u1_hi       = _sampler.generateInverse(x1_hi);
      const domain_type dudx0 = (u0_hi - u0_lo) / twoEps;
      const domain_type dudx1 = (u1_hi - u1_lo) / twoEps;
      using matrix2_type      = matrix<scalar_type, 2, 2>;
      const scalar_type det   = nbl::hlsl::determinant<matrix2_type>(matrix2_type(dudx0, dudx1));
      return nbl::hlsl::abs<scalar_type>(det);
   }
};

template<typename Sampler>
struct InverseJacobianMeasure<Sampler, 2, 3>
{
   using scalar_type   = typename Sampler::scalar_type;
   using domain_type   = typename Sampler::domain_type;
   using codomain_type = typename Sampler::codomain_type;

   static scalar_type compute(Sampler _sampler, codomain_type x, scalar_type eps)
   {
      const scalar_type twoEps = scalar_type(2) * eps;
      codomain_type t1, t2;
      const codomain_type up  = nbl::hlsl::abs<scalar_type>(x[2]) < scalar_type(0.999)
         ? codomain_type(scalar_type(0), scalar_type(0), scalar_type(1))
         : codomain_type(scalar_type(1), scalar_type(0), scalar_type(0));
      t1                      = nbl::hlsl::normalize(nbl::hlsl::cross(up, x));
      t2                      = nbl::hlsl::cross(x, t1);
      domain_type u_t1_lo     = _sampler.generateInverse(nbl::hlsl::normalize(x - t1 * eps));
      domain_type u_t1_hi     = _sampler.generateInverse(nbl::hlsl::normalize(x + t1 * eps));
      domain_type u_t2_lo     = _sampler.generateInverse(nbl::hlsl::normalize(x - t2 * eps));
      domain_type u_t2_hi     = _sampler.generateInverse(nbl::hlsl::normalize(x + t2 * eps));
      const domain_type dudt1 = (u_t1_hi - u_t1_lo) / twoEps;
      const domain_type dudt2 = (u_t2_hi - u_t2_lo) / twoEps;
      using matrix2_type      = matrix<scalar_type, 2, 2>;
      const scalar_type det   = nbl::hlsl::determinant<matrix2_type>(matrix2_type(dudt1, dudt2));
      return nbl::hlsl::abs<scalar_type>(det);
   }
};

template<typename Sampler>
float32_t computeInverseJacobianPdf(Sampler _sampler, typename Sampler::codomain_type sample, float32_t backwardPdf, float32_t pdfMin, float32_t pdfMax)
{
   using scalar_type   = typename Sampler::scalar_type;
   using domain_type   = typename Sampler::domain_type;
   using codomain_type = typename Sampler::codomain_type;

   if (backwardPdf < scalar_type(pdfMin) || backwardPdf > scalar_type(pdfMax))
      return JACOBIAN_SKIP_BWD_PDF_RANGE;

   using measure_type    = InverseJacobianMeasure<Sampler, vector_traits<domain_type>::Dimension, vector_traits<codomain_type>::Dimension>;
   const scalar_type eps = scalar_type(1e-3);
   return measure_type::compute(_sampler, sample, eps);
}

#endif
