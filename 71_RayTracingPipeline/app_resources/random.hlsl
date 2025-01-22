// Generate a random unsigned int from two unsigned int values, using 16 pairs
// of rounds of the Tiny Encryption Algorithm. See Zafar, Olano, and Curtis,
// "GPU Random Numbers via the Tiny Encryption Algorithm"
uint32_t tea(uint32_t val0, uint32_t val1)
{
  uint32_t v0 = val0;
  uint32_t v1 = val1;
  uint32_t s0 = 0;

  for(uint32_t n = 0; n < 16; n++)
  {
    s0 += 0x9e3779b9;
    v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
    v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
  }

  return v0;
}

// Generate a random unsigned int in [0, 2^24) given the previous RNG state
// using the Numerical Recipes linear congruential generator
uint32_t lcg(inout uint32_t prev)
{
  uint32_t LCG_A = 1664525u;
  uint32_t LCG_C = 1013904223u;
  prev       = (LCG_A * prev + LCG_C);
  return prev & 0x00FFFFFF;
}

// Generate a random float32_t in [0, 1) given the previous RNG state
float32_t rnd(inout uint32_t prev)
{
  return (float32_t(lcg(prev)) / float32_t(0x01000000));
}
