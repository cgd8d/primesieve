///
/// @file   PrimeGenerator.cpp
/// @brief  Generates the primes inside [start, stop] and stores them
///         in a vector. After the primes have been stored in the
///         vector primesieve::iterator iterates over the vector and
///         returns the primes. When there are no more primes left in
///         the vector PrimeGenerator generates new primes.
///
///         primesieve::iterator's next_prime() performance depends
///         on PrimeGenerator::fillNextPrimes(). Therefore
///         fillNextPrimes() is highly optimized using hardware
///         acceleration (e.g. CTZ, AVX512) whenever possible.
///
/// Copyright (C) 2024 Kim Walisch, <kim.walisch@gmail.com>
/// Copyright (C) 2022 @zielaj, https://github.com/zielaj
///
/// This file is distributed under the BSD License. See the COPYING
/// file in the top level directory.
///

#include <primesieve/Erat.hpp>
#include <primesieve/forward.hpp>
#include <primesieve/littleendian_cast.hpp>
#include <primesieve/macros.hpp>
#include <primesieve/PreSieve.hpp>
#include <primesieve/PrimeGenerator.hpp>
#include <primesieve/primesieve_error.hpp>
#include <primesieve/macros.hpp>
#include <primesieve/pmath.hpp>
#include <primesieve/popcnt.hpp>
#include <primesieve/Vector.hpp>
#include <primesieve/SievingPrimes.hpp>

#include <stdint.h>
#include <algorithm>
#include <limits>

#include <iostream>

#if defined(ENABLE_AVX512_VBMI2) || \
    defined(ENABLE_MULTIARCH_AVX512_VBMI2)
  #include <immintrin.h>
#endif

namespace {

/// First 128 primes
const primesieve::Array<uint64_t, 128> smallPrimes =
{
    2,   3,   5,   7,  11,  13,  17,  19,  23,  29,
   31,  37,  41,  43,  47,  53,  59,  61,  67,  71,
   73,  79,  83,  89,  97, 101, 103, 107, 109, 113,
  127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
  179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
  233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
  283, 293, 307, 311, 313, 317, 331, 337, 347, 349,
  353, 359, 367, 373, 379, 383, 389, 397, 401, 409,
  419, 421, 431, 433, 439, 443, 449, 457, 461, 463,
  467, 479, 487, 491, 499, 503, 509, 521, 523, 541,
  547, 557, 563, 569, 571, 577, 587, 593, 599, 601,
  607, 613, 617, 619, 631, 641, 643, 647, 653, 659,
  661, 673, 677, 683, 691, 701, 709, 719
};

/// Number of primes <= n
const primesieve::Array<uint8_t, 720> primePi =
{
    0,   0,   1,   2,   2,   3,   3,   4,   4,   4,   4,   5,   5,   6,   6,
    6,   6,   7,   7,   8,   8,   8,   8,   9,   9,   9,   9,   9,   9,  10,
   10,  11,  11,  11,  11,  11,  11,  12,  12,  12,  12,  13,  13,  14,  14,
   14,  14,  15,  15,  15,  15,  15,  15,  16,  16,  16,  16,  16,  16,  17,
   17,  18,  18,  18,  18,  18,  18,  19,  19,  19,  19,  20,  20,  21,  21,
   21,  21,  21,  21,  22,  22,  22,  22,  23,  23,  23,  23,  23,  23,  24,
   24,  24,  24,  24,  24,  24,  24,  25,  25,  25,  25,  26,  26,  27,  27,
   27,  27,  28,  28,  29,  29,  29,  29,  30,  30,  30,  30,  30,  30,  30,
   30,  30,  30,  30,  30,  30,  30,  31,  31,  31,  31,  32,  32,  32,  32,
   32,  32,  33,  33,  34,  34,  34,  34,  34,  34,  34,  34,  34,  34,  35,
   35,  36,  36,  36,  36,  36,  36,  37,  37,  37,  37,  37,  37,  38,  38,
   38,  38,  39,  39,  39,  39,  39,  39,  40,  40,  40,  40,  40,  40,  41,
   41,  42,  42,  42,  42,  42,  42,  42,  42,  42,  42,  43,  43,  44,  44,
   44,  44,  45,  45,  46,  46,  46,  46,  46,  46,  46,  46,  46,  46,  46,
   46,  47,  47,  47,  47,  47,  47,  47,  47,  47,  47,  47,  47,  48,  48,
   48,  48,  49,  49,  50,  50,  50,  50,  51,  51,  51,  51,  51,  51,  52,
   52,  53,  53,  53,  53,  53,  53,  53,  53,  53,  53,  54,  54,  54,  54,
   54,  54,  55,  55,  55,  55,  55,  55,  56,  56,  56,  56,  56,  56,  57,
   57,  58,  58,  58,  58,  58,  58,  59,  59,  59,  59,  60,  60,  61,  61,
   61,  61,  61,  61,  61,  61,  61,  61,  62,  62,  62,  62,  62,  62,  62,
   62,  62,  62,  62,  62,  62,  62,  63,  63,  63,  63,  64,  64,  65,  65,
   65,  65,  66,  66,  66,  66,  66,  66,  66,  66,  66,  66,  66,  66,  66,
   66,  67,  67,  67,  67,  67,  67,  68,  68,  68,  68,  68,  68,  68,  68,
   68,  68,  69,  69,  70,  70,  70,  70,  71,  71,  71,  71,  71,  71,  72,
   72,  72,  72,  72,  72,  72,  72,  73,  73,  73,  73,  73,  73,  74,  74,
   74,  74,  74,  74,  75,  75,  75,  75,  76,  76,  76,  76,  76,  76,  77,
   77,  77,  77,  77,  77,  77,  77,  78,  78,  78,  78,  79,  79,  79,  79,
   79,  79,  79,  79,  80,  80,  80,  80,  80,  80,  80,  80,  80,  80,  81,
   81,  82,  82,  82,  82,  82,  82,  82,  82,  82,  82,  83,  83,  84,  84,
   84,  84,  84,  84,  85,  85,  85,  85,  86,  86,  86,  86,  86,  86,  87,
   87,  87,  87,  87,  87,  87,  87,  88,  88,  88,  88,  89,  89,  90,  90,
   90,  90,  91,  91,  91,  91,  91,  91,  91,  91,  91,  91,  91,  91,  92,
   92,  92,  92,  92,  92,  92,  92,  93,  93,  93,  93,  94,  94,  94,  94,
   94,  94,  94,  94,  95,  95,  95,  95,  96,  96,  96,  96,  96,  96,  97,
   97,  97,  97,  97,  97,  97,  97,  97,  97,  97,  97,  98,  98,  99,  99,
   99,  99,  99,  99,  99,  99,  99,  99,  99,  99,  99,  99,  99,  99,  99,
   99, 100, 100, 100, 100, 100, 100, 101, 101, 101, 101, 101, 101, 101, 101,
  101, 101, 102, 102, 102, 102, 102, 102, 103, 103, 103, 103, 103, 103, 104,
  104, 105, 105, 105, 105, 105, 105, 106, 106, 106, 106, 106, 106, 106, 106,
  106, 106, 107, 107, 107, 107, 107, 107, 108, 108, 108, 108, 108, 108, 109,
  109, 110, 110, 110, 110, 110, 110, 111, 111, 111, 111, 111, 111, 112, 112,
  112, 112, 113, 113, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114,
  114, 115, 115, 115, 115, 115, 115, 115, 115, 115, 115, 116, 116, 117, 117,
  117, 117, 118, 118, 118, 118, 118, 118, 119, 119, 119, 119, 119, 119, 120,
  120, 121, 121, 121, 121, 121, 121, 121, 121, 121, 121, 121, 121, 122, 122,
  122, 122, 123, 123, 123, 123, 123, 123, 124, 124, 124, 124, 124, 124, 124,
  124, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 126, 126, 126, 126,
  126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 128
};

} // namespace

namespace primesieve {

PrimeGenerator::PrimeGenerator(uint64_t start,
                               uint64_t stop,
                               PreSieve& preSieve) :
  Erat(start, stop),
  preSieve_(preSieve)
{ }

uint64_t PrimeGenerator::maxCachedPrime()
{
  return smallPrimes.back();
}

std::size_t PrimeGenerator::getStartIdx() const
{
  std::size_t startIdx = 0;

  if (start_ > 1)
    startIdx = primePi[start_ - 1];

  return startIdx;
}

std::size_t PrimeGenerator::getStopIdx() const
{
  std::size_t stopIdx = 0;

  if (stop_ < maxCachedPrime())
    stopIdx = primePi[stop_];
  else
    stopIdx = smallPrimes.size();

  return stopIdx;
}

/// Used by iterator::prev_prime()
void PrimeGenerator::initPrevPrimes(Vector<uint64_t>& primes,
                                    std::size_t* size)
{
  auto resize = [](Vector<uint64_t>& primes,
                   std::size_t size)
  {
    // Avoids reallocation in fillPrevPrimes()
    size += 64;

    if (primes.empty())
      primes.resize(size);
    // When sieving backwards the number of primes inside [start, stop]
    // slowly increases in each new segment as there are more small
    // than large primes. Our new size has been calculated using
    // primeCountUpper(start, stop) which is usually too large by 4%
    // near 10^12 and by 2.5% near 10^19. Hence if the new size is less
    // than 1% larger than the old size we do not increase the primes
    // buffer as it will likely be large enough to fit all primes.
    else if (size > primes.size() &&
             (double) size / (double) primes.size() > 1.01)
    {
      // Prevent unnecessary copying when resizing
      primes.clear();
      primes.resize(size);
    }
  };

  std::size_t pix = primeCountUpper(start_, stop_);

  if (start_ <= maxCachedPrime())
  {
    std::size_t a = getStartIdx();
    std::size_t b = getStopIdx();
    ASSERT(a <= b);

    *size = (start_ <= 2) + b - a;
    resize(primes, std::max(*size, pix));
    std::size_t i = 0;

    if (start_ <= 2)
      primes[i++] = 0;

    std::copy(smallPrimes.begin() + a,
              smallPrimes.begin() + b,
              &primes[i]);
  }
  else
    resize(primes, pix);

  initErat();
}

/// Used by iterator::next_prime()
void PrimeGenerator::initNextPrimes(Vector<uint64_t>& primes,
                                    std::size_t* size)
{
  auto resize = [](Vector<uint64_t>& primes,
                   std::size_t size)
  {
    if (size > primes.size())
    {
      // Prevent unnecessary copying when resizing
      primes.clear();
      primes.resize(size);
    }
  };

  // A buffer of 1024 primes provides good
  // performance with little memory usage.
  std::size_t maxSize = 1 << 10;

  if (start_ <= maxCachedPrime())
  {
    std::size_t a = getStartIdx();
    std::size_t b = getStopIdx();
    *size = b - a;

    if (stop_ < maxCachedPrime() + 2)
      resize(primes, *size);
    else
    {
      // +64 is needed because our fillNextPrimes()
      // algorithm aborts as soon as there is not
      // enough space to store 64 more primes.
      std::size_t minSize = *size + 64;
      std::size_t pix = primeCountUpper(start_, stop_) + 64;
      pix = inBetween(minSize, pix, maxSize);
      pix = std::max(*size, pix);
      resize(primes, pix);
    }

    ASSERT(primes.size() >= *size);
    std::copy(smallPrimes.begin() + a,
              smallPrimes.begin() + b,
              primes.begin());
  }
  else
  {
    // +64 is needed because our fillNextPrimes()
    // algorithm aborts as soon as there is not
    // enough space to store 64 more primes.
    std::size_t minSize = 64;
    std::size_t pix = primeCountUpper(start_, stop_) + 64;
    pix = inBetween(minSize, pix, maxSize);
    resize(primes, pix);
  }

  initErat();
}

void PrimeGenerator::initErat()
{
  ASSERT(maxCachedPrime() >= 5);
  uint64_t startErat = maxCachedPrime() + 2;
  startErat = std::max(startErat, start_);
  isInit_ = true;

  if (startErat <= stop_ &&
      startErat < std::numeric_limits<uint64_t>::max())
  {
    preSieve_.init(startErat, stop_);
    int sieveSize = get_sieve_size();
    Erat::init(startErat, stop_, sieveSize, preSieve_, memoryPool_);
    sievingPrimes_.init(this, sieveSize, preSieve_, memoryPool_);
  }
}

void PrimeGenerator::sieveSegment()
{
  uint64_t sqrtHigh = isqrt(segmentHigh_);

  sieveIdx_ = 0;
  low_ = segmentLow_;

  if (!prime_)
    prime_ = sievingPrimes_.next();

  while (prime_ <= sqrtHigh)
  {
    addSievingPrime(prime_);
    prime_ = sievingPrimes_.next();
  }

  Erat::sieveSegment();
}

/// Used by iterator::prev_prime()
bool PrimeGenerator::sievePrevPrimes(Vector<uint64_t>& primes,
                                     std::size_t* size)
{
  if (!isInit_)
    initPrevPrimes(primes, size);

  if (hasNextSegment())
  {
    sieveSegment();
    return true;
  }

  // We have generated all primes inside [start, stop], we cannot
  // generate more primes using this PrimeGenerator. Therefore we
  // need to allocate a new PrimeGenerator in iterator.cpp.
  return false;
}

/// Used by iterator::next_prime()
bool PrimeGenerator::sieveNextPrimes(Vector<uint64_t>& primes,
                                     std::size_t* size)
{
  if (!isInit_)
    initNextPrimes(primes, size);

  if (hasNextSegment())
  {
    sieveSegment();
    return true;
  }

  // The next prime would be > 2^64
  if_unlikely(stop_ >= std::numeric_limits<uint64_t>::max())
    throw primesieve_error("cannot generate primes > 2^64");

  // We have generated all primes <= stop, we cannot generate
  // more primes using this PrimeGenerator. Therefore we
  // need to allocate a new PrimeGenerator in iterator.cpp.
  return false;
}

/// This method is used by iterator::prev_prime().
/// This method stores all primes inside [a, b] into the primes
/// vector. (b - a) is about sqrt(stop) so the memory usage is
/// quite large. Also after primesieve::iterator has iterated
/// over the primes inside [a, b] we need to generate new
/// primes which incurs an initialization overhead of O(sqrt(n)).
///
void PrimeGenerator::fillPrevPrimes(Vector<uint64_t>& primes,
                                    std::size_t* size)
{
  *size = 0;

  while (sievePrevPrimes(primes, size))
  {
    // Use local variables to prevent the compiler from
    // writing temporary results to memory.
    std::size_t i = *size;
    uint64_t low = low_;
    uint64_t sieveIdx = sieveIdx_;
    uint64_t sieveSize = sieve_.size();
    uint8_t* sieve = sieve_.data();

    while (sieveIdx < sieveSize)
    {
      // Each loop iteration can generate up to 64 primes,
      // so we have to make sure there is enough space
      // left in the primes vector.
      if_unlikely(i + 64 > primes.size())
        primes.resize(i + 64);

      uint64_t bits = littleendian_cast<uint64_t>(&sieve[sieveIdx]);
      std::size_t j = i;
      i += popcnt64(bits);

      do
      {
        primes[j+0] = nextPrime(bits, low); bits &= bits - 1;
        primes[j+1] = nextPrime(bits, low); bits &= bits - 1;
        primes[j+2] = nextPrime(bits, low); bits &= bits - 1;
        primes[j+3] = nextPrime(bits, low); bits &= bits - 1;
        j += 4;
      }
      while (j < i);

      low += 8 * 30;
      sieveIdx += 8;
    }

    low_ = low;
    sieveIdx_ = sieveIdx;
    *size = i;
  }
}


struct TestHist
{
  std::vector<uint64_t> v;
  uint64_t t;

  TestHist()
{
    v.assign(65,0);
    t=0;
}

~TestHist()
{
    std::cout<< "Hist results ("<<t<<" total hits):"<<std::endl;
    for(size_t i=0;i<65;i++){
        std::cout<<i<<"\t: "<<100*double(v[i])/t<<std::endl;
    }
}
};
TestHist aTestHist;

#if defined(ENABLE_DEFAULT)

/// This method is used by iterator::next_prime().
/// This method stores only the next few primes (~ 1000) in the
/// primes vector. Also for iterator::next_prime() there is no
/// recurring initialization overhead (unlike prev_prime()) for
/// this reason iterator::next_prime() runs up to 2x faster
/// than iterator::prev_prime().
///
void PrimeGenerator::fillNextPrimes_default(Vector<uint64_t>& primes, std::size_t* size)
{
  *size = 0;

  do
  {
    if (sieveIdx_ >= sieve_.size())
      if (!sieveNextPrimes(primes, size))
        return;

    // Use local variables to prevent the compiler from
    // writing temporary results to memory.
    std::size_t i = *size;
    std::size_t maxSize = primes.size();
    ASSERT(i + 64 <= maxSize);
    //uint64_t low = low_;
    uint64_t sieveIdx = sieveIdx_;
    uint64_t sieveSize = sieve_.size();
    uint8_t* sieve = sieve_.data();

    __m256i low_vec = _mm256_set1_epi64x(low_);
    __m256i low_step = _mm256_set1_epi64x(8 * 30);

    // Create bitvals_lookup helper register for AVX.
    /*uint64_t bitvals64_0 =
      ( 7ull      ) +
      (11ull <<  8) +
      (13ull << 16) +
      (17ull << 24) +
      (19ull << 32) +
      (23ull << 40) +
      (29ull << 48) +
      (31ull << 56);
    // Generate subsequent values by adding
    // multiples of 30 (= 1e) to each byte value.
    uint64_t bitvals_step = 0x1e1e1e1e1e1e1e1eULL;
    uint64_t bitvals64_1 = bitvals64_0 + bitvals_step;
    __m128i bitvals_half = _mm_set_epi64x(bitvals64_1, bitvals64_0); // may not be VEX coded
    __m256i bitvals_lookup = _mm256_set_m128i(bitvals_half, bitvals_half);

    // Also create helper with lookup of
    // multiples if 60.
    __m128i bitvals_lookuphi_half = _mm_set_epi8(
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
      (char)180, 120, 60, 0);
    __m256i bitvals_lookuphi = _mm256_set_m128i(bitvals_lookuphi_half, bitvals_lookuphi_half);

    // Pregenerate the shuffle indices that select out
    // groups of bit values. 
    __m128i idx_ungroup_tail_half = _mm_set_epi64x(12, 4);
    __m256i idx_ungroup_tail = _mm256_set_m128i(idx_ungroup_tail_half, idx_ungroup_tail_half);
    __m128i idx_ungroup_lead_half = _mm_set_epi64x(8, 0);
    __m256i idx_ungroup_lead = _mm256_set_m128i(idx_ungroup_lead_half, idx_ungroup_lead_half);
      */
    // Finally create logical mask to suppress
    // unintended byes. 
    //__m256i mask_bitvals = _mm256_set1_epi64x(0xFF);

    // Fill the buffer with at least (maxSize - 64) primes.
    // Each loop iteration can generate up to 64 primes
    // so we have to stop generating primes once there is
    // not enough space for 64 more primes.
    do
    {
      uint64_t bits = littleendian_cast<uint64_t>(&sieve[sieveIdx]);
      uint64_t bits_lz = bits;
      std::size_t j = i;
      size_t pc = popcnt64(bits);
      i += pc;
      //size_t lz_idx = i-4;
      //aTestHist.t++;
      //aTestHist.v[pc]++;

      // Make vector of value low.
      //__m256i low_vec = _mm256_set1_epi64x(low);

      /*
      if(pc < 4 and pc != 0)
      {
        while(j < i)
        {
          auto bitIndex = ctz64(bits); bits &= bits - 1;
          primes[j++] = low + bitValues[bitIndex];
        }
      }
      else
      {
        
      uint64_t* jptr = primes.data() + j;
      uint64_t* lptr = primes.data() + i - 4;
      */

      //for(size_t iter = (pc+7)/8; iter != 0; iter--)
      //size_t break_iter = (pc+3)/8;
      //for(size_t iter = 0; true; iter++)
      //{

        // Handle up to 14 primes in branchless code.
        // We want the compiler to interleave lzcnt
        // and tzcnt, which on most processors 
        // can run in parallel.  Compilers don't
        // seem to handle scheduling well at this level, 
        // but ordering the C++ code seems to help.

        auto bitIndexZ = 63ull xor __builtin_clzll(bits_lz);
        bits_lz = _bzhi_u64(bits_lz, bitIndexZ);
        auto bitIndex0 = ctz64(bits); bits &= bits - 1;
        __m128i bitVals_tail0_lo = _mm_cvtsi64_si128(
          bitValues[bitIndex0]
        );
        
        auto bitIndexY = 63ull xor __builtin_clzll(bits_lz);
        bits_lz = _bzhi_u64(bits_lz, bitIndexY);
        auto bitIndex1 = ctz64(bits); bits &= bits - 1;

        bitVals_tail0_lo = _mm_insert_epi64(
          bitVals_tail0_lo,
          bitValues[bitIndex1],
          1);
        __m128i bitVals_lead1_hi = _mm_set_epi64x(
          bitValues[bitIndexZ],
          bitValues[bitIndexY]
        );
        
        /*__m128i bitVals_tail0_lo = _mm_set_epi64x(
          bitValues[bitIndex1],
          bitValues[bitIndex0]
        );*/
        
        auto bitIndexX = 63ull xor __builtin_clzll(bits_lz);
        bits_lz = _bzhi_u64(bits_lz, bitIndexX);
        auto bitIndex2 = ctz64(bits); bits &= bits - 1;
        
        auto bitIndexW = 63ull xor __builtin_clzll(bits_lz);
        auto bitIndex3 = ctz64(bits); bits &= bits - 1;

        __m128i bitVals_lead1_lo = _mm_set_epi64x(
          bitValues[bitIndexX],
          bitValues[bitIndexW]
        );
        __m128i bitVals_tail0_hi = _mm_set_epi64x(
          bitValues[bitIndex3],
          bitValues[bitIndex2]
        );

        __m256i bitVals_lead1 = _mm256_set_m128i(
          bitVals_lead1_hi,
          bitVals_lead1_lo
        );
        __m256i bitVals_tail0 = _mm256_set_m128i(
          bitVals_tail0_hi,
          bitVals_tail0_lo
        );
        
        /*
        __m256i bitVals_lead1 = _mm256_set_epi64x(
          bitValues[bitIndexZ],
          bitValues[bitIndexY],
          bitValues[bitIndexX],
          bitValues[bitIndexW]
        );
        __m256i bitVals_tail0 = _mm256_set_epi64x(
          bitValues[bitIndex3],
          bitValues[bitIndex2],
          bitValues[bitIndex1],
          bitValues[bitIndex0]
        );*/


        auto bitIndex4 = ctz64(bits); bits &= bits - 1;
        bits_lz = _bzhi_u64(bits_lz, bitIndexW);
        auto bitIndexV = 63ull xor __builtin_clzll(bits_lz);

        auto bitIndex5 = ctz64(bits); bits &= bits - 1;
        bits_lz = _bzhi_u64(bits_lz, bitIndexV);
        auto bitIndexU = 63ull xor __builtin_clzll(bits_lz);

        auto bitIndex6 = ctz64(bits); bits &= bits - 1;
        auto bitIndex7 = ctz64(bits); //bits &= bits - 1;
        
        //bits_lz = _bzhi_u64(bits_lz, bitIndexU);
        __m128i bitVals_lead0 = _mm_set_epi64x(
          bitValues[bitIndexV],
          bitValues[bitIndexU]
        );
        __m256i bitVals_tail1 = _mm256_set_epi64x(
          bitValues[bitIndex7],
          bitValues[bitIndex6],
          bitValues[bitIndex5],
          bitValues[bitIndex4]
        );
        

        
        
        

        // Store 6 primes from lzcnt first.
        // If pc<6 then we don't need these, 
        // but to keep the code branch-free
        // we use a destination address that
        // will do no harm.
        // Compiler should implement with
        // branch-free cmov instruction.
        size_t lz_dest = (pc < 6) ? j : i-6;
        
        __m256i nextPrimes_lead1 = _mm256_add_epi64(bitVals_lead1, low_vec);
        _mm256_storeu_si256((__m256i*)(primes.data()+lz_dest+2), nextPrimes_lead1);
        
        __m128i nextPrimes_lead0 = _mm_add_epi64(bitVals_lead0, _mm256_castsi256_si128(low_vec));
        _mm_storeu_si128((__m128i*)(primes.data()+lz_dest), nextPrimes_lead0);

        // Load bit indices into ymm register. 
        // Warning: if a compiler implements this
        // with legacy SSE instructions, expect a
        // performance regression.  I believe that
        // compilers with AVX enabled will typically
        // disable legacy SSE and use VEX-coded equivalents. 
        //__m256i bitIndices = _mm256_set_epi64x(bitIndex3, bitIndex2, bitIndex1, bitIndex0);
        /*__m256i bitIndices = _mm256_set_epi32(
          bitIndex3, bitIndexZ, bitIndex2, bitIndexY, // high half
          bitIndex1, bitIndexX, bitIndex0, bitIndexW // low half
          );

        // look up offset values from bit positions
        __m256i bitIndices_hi = _mm256_srli_epi32(bitIndices, 4);
        __m256i bitVals_lo = _mm256_shuffle_epi8(bitvals_lookup, bitIndices);
        //__m256i bitVals_lo = _mm256_and_si256(bitVals_lo_um, mask_bitvals);
        __m256i bitVals_hi = _mm256_shuffle_epi8(bitvals_lookuphi, bitIndices_hi);
        __m256i bitVals = _mm256_add_epi64(bitVals_lo, bitVals_hi);
*/
        // extract lead and tail bits in groups of 4,
        // add low, and store.
        //__m256i bitVals_tail_um = _mm256_shuffle_epi8(bitVals, idx_ungroup_tail);
        //__m256i bitVals_tail = _mm256_and_si256(bitVals_tail_um, mask_bitvals);

        
        __m256i nextPrimes_tail0 = _mm256_add_epi64(bitVals_tail0, low_vec);
        _mm256_storeu_si256((__m256i*)(primes.data()+j), nextPrimes_tail0);
        
        __m256i nextPrimes_tail1 = _mm256_add_epi64(bitVals_tail1, low_vec);
        _mm256_storeu_si256((__m256i*)(primes.data()+j+4), nextPrimes_tail1);

        /*if(not std::is_sorted(primes.data()+j+4*iter, primes.data()+j+std::min(4*iter+4,pc)))
        {
            std::cout << "tail primes not sorted." << std::endl;
            std::exit(1);
        }*/
        //if(pc >= 4)
        //if(iter == break_iter) break;
        //{
        //__m256i bitVals_lead_um = _mm256_shuffle_epi8(bitVals, idx_ungroup_lead);
        //__m256i bitVals_lead = _mm256_and_si256(bitVals_lead_um, mask_bitvals);

        /*__m256i bitVals_lead = _mm256_set_epi64x(
          bitValues[bitIndexZ],
          bitValues[bitIndexY],
          bitValues[bitIndexX],
          bitValues[bitIndexW]
        );
        __m256i nextPrimes_lead = _mm256_add_epi64(bitVals_lead, low_vec);
        _mm256_storeu_si256((__m256i*)(primes.data()+i-4-4*iter), nextPrimes_lead);*/
        /*if(not std::is_sorted(primes.data()+i-4-4*iter, primes.data()+i-4*iter))
        {
            std::cout << "lead primes not sorted." << std::endl;
            std::exit(1);
        }*/
        /*j += 4;
       for(size_t iter = 8; iter < pc; iter += 2)
        {
            primes[j  ] = nextPrime(bits, low); bits &= bits - 1;
            primes[j+1] = nextPrime(bits, low); bits &= bits - 1;
            j += 2;
        }*/
       // jptr += 4;
       // lptr -= 4;
      //}
      //}

      // We have handled up to 14 primes
      // which is usually enough. 
      // In the unlikely event that it
      // isn't, use this code (almost surely
      // triggering branch misprediction).
      for(size_t iter = 0; iter + 14 < pc; iter++) [[unlikely]]
      {
        bits &= bits - 1;
        auto bitIndex = ctz64(bits);
        uint64_t this_low = _mm_cvtsi128_si64(_mm256_castsi256_si128(low_vec));
        primes[j+8+iter] = this_low + bitValues[bitIndex];// nextPrime(bits, low); bits &= bits - 1;
      }

    /*if(not std::is_sorted(primes.data()+i-pc, primes.data()+i))
        {
            std::cout << "group of primes not sorted." << std::endl;
            std::exit(1);
        }*/

      //low += 8 * 30;

      low_vec = _mm256_add_epi64(low_vec, low_step);
        
      sieveIdx += 8;
    }
    while (i <= maxSize - 64 &&
           sieveIdx < sieveSize);

    // Recommended to avoid mixing SSE and
    // AVX instructions, which has a penalty 
    // on many processors.
    _mm256_zeroupper();

    //low_ = low;

    low_ = _mm_cvtsi128_si64(_mm256_castsi256_si128(low_vec));
    sieveIdx_ = sieveIdx;
    *size = i;
  }
  while (*size == 0);
}

#endif

#if defined(ENABLE_AVX512_VBMI2) || \
    defined(ENABLE_MULTIARCH_AVX512_VBMI2)

/// This algorithm converts 1 bits from the sieve array into primes
/// using AVX512. The algorithm is a modified version of the AVX512
/// algorithm which converts 1 bits into bit indexes from:
/// https://branchfree.org/2018/05/22/bits-to-indexes-in-bmi2-and-avx-512
/// https://github.com/kimwalisch/primesieve/pull/109
///
/// Our algorithm is optimized for sparse bitstreams that are
/// distributed relatively evenly. While processing a 64-bit word
/// from the sieve array there are if checks that skip to the next
/// loop iteration once all 1 bits have been processed. In my
/// benchmarks this algorithm ran about 10% faster than the default
/// fillNextPrimes() algorithm which uses __builtin_ctzll().
///
#if defined(ENABLE_MULTIARCH_AVX512_VBMI2)
  __attribute__ ((target ("avx512f,avx512vbmi,avx512vbmi2")))
#endif
void PrimeGenerator::fillNextPrimes_avx512(Vector<uint64_t>& primes, std::size_t* size)
{
  *size = 0;

  do
  {
    if (sieveIdx_ >= sieve_.size())
      if (!sieveNextPrimes(primes, size))
        return;

    // Use local variables to prevent the compiler from
    // writing temporary results to memory.
    std::size_t i = *size;
    std::size_t maxSize = primes.size();
    ASSERT(i + 64 <= maxSize);
    uint64_t low = low_;
    uint64_t sieveIdx = sieveIdx_;
    uint64_t sieveSize = sieve_.size();
    uint8_t* sieve = sieve_.data();

    __m512i avxBitValues = _mm512_set_epi8(
      (char) 241, (char) 239, (char) 233, (char) 229,
      (char) 227, (char) 223, (char) 221, (char) 217,
      (char) 211, (char) 209, (char) 203, (char) 199,
      (char) 197, (char) 193, (char) 191, (char) 187,
      (char) 181, (char) 179, (char) 173, (char) 169,
      (char) 167, (char) 163, (char) 161, (char) 157,
      (char) 151, (char) 149, (char) 143, (char) 139,
      (char) 137, (char) 133, (char) 131, (char) 127,
      (char) 121, (char) 119, (char) 113, (char) 109,
      (char) 107, (char) 103, (char) 101, (char)  97,
      (char)  91, (char)  89, (char)  83, (char)  79,
      (char)  77, (char)  73, (char)  71, (char)  67,
      (char)  61, (char)  59, (char)  53, (char)  49,
      (char)  47, (char)  43, (char)  41, (char)  37,
      (char)  31, (char)  29, (char)  23, (char)  19,
      (char)  17, (char)  13, (char)  11, (char)   7
    );

    __m512i bytes_0_to_7   = _mm512_setr_epi64( 0,  1,  2,  3,  4,  5,  6,  7);
    __m512i bytes_8_to_15  = _mm512_setr_epi64( 8,  9, 10, 11, 12, 13, 14, 15);
    __m512i bytes_16_to_23 = _mm512_setr_epi64(16, 17, 18, 19, 20, 21, 22, 23);
    __m512i bytes_24_to_31 = _mm512_setr_epi64(24, 25, 26, 27, 28, 29, 30, 31);
    __m512i bytes_32_to_39 = _mm512_setr_epi64(32, 33, 34, 35, 36, 37, 38, 39);
    __m512i bytes_40_to_47 = _mm512_setr_epi64(40, 41, 42, 43, 44, 45, 46, 47);
    __m512i bytes_48_to_55 = _mm512_setr_epi64(48, 49, 50, 51, 52, 53, 54, 55);
    __m512i bytes_56_to_63 = _mm512_setr_epi64(56, 57, 58, 59, 60, 61, 62, 63);

    while (sieveIdx < sieveSize)
    {
      // Each iteration processes 8 bytes from the sieve array
      uint64_t bits64 = *(uint64_t*) &sieve[sieveIdx];
      uint64_t primeCount = popcnt64(bits64);

      // Prevent _mm512_storeu_si512() buffer overrun
      if (i + primeCount >= maxSize - 7)
        break;

      __m512i base = _mm512_set1_epi64(low);
      uint64_t* primes64 = &primes[i];

      // These variables are not used anymore during this
      // iteration, increment for next iteration.
      i += primeCount;
      low += 8 * 30;
      sieveIdx += 8;

      // Convert 1 bits from the sieve array (bits64) into prime
      // bit values (bytes) using the avxBitValues lookup table and
      // move all non zero bytes (bit values) to the beginning.
      __m512i bitValues = _mm512_maskz_compress_epi8(bits64, avxBitValues);

      // Convert the first 8 bytes (prime bit values)
      // into eight 64-bit prime numbers.
      __m512i vprimes0 = _mm512_maskz_permutexvar_epi8(0x0101010101010101ull, bytes_0_to_7, bitValues);
      vprimes0 = _mm512_add_epi64(base, vprimes0);
      _mm512_storeu_si512(&primes64[0], vprimes0);

      if (primeCount <= 8)
        continue;

      __m512i vprimes1 = _mm512_maskz_permutexvar_epi8(0x0101010101010101ull, bytes_8_to_15, bitValues);
      vprimes1 = _mm512_add_epi64(base, vprimes1);
      _mm512_storeu_si512(&primes64[8], vprimes1);

      if (primeCount <= 16)
        continue;

      __m512i vprimes2 = _mm512_maskz_permutexvar_epi8(0x0101010101010101ull, bytes_16_to_23, bitValues);
      vprimes2 = _mm512_add_epi64(base, vprimes2);
      _mm512_storeu_si512(&primes64[16], vprimes2);

      if (primeCount <= 24)
        continue;

      __m512i vprimes3 = _mm512_maskz_permutexvar_epi8(0x0101010101010101ull, bytes_24_to_31, bitValues);
      vprimes3 = _mm512_add_epi64(base, vprimes3);
      _mm512_storeu_si512(&primes64[24], vprimes3);

      if (primeCount <= 32)
        continue;

      __m512i vprimes4 = _mm512_maskz_permutexvar_epi8(0x0101010101010101ull, bytes_32_to_39, bitValues);
      vprimes4 = _mm512_add_epi64(base, vprimes4);
      _mm512_storeu_si512(&primes64[32], vprimes4);

      if (primeCount <= 40)
        continue;

      __m512i vprimes5 = _mm512_maskz_permutexvar_epi8(0x0101010101010101ull, bytes_40_to_47, bitValues);
      vprimes5 = _mm512_add_epi64(base, vprimes5);
      _mm512_storeu_si512(&primes64[40], vprimes5);

      if (primeCount <= 48)
        continue;

      __m512i vprimes6 = _mm512_maskz_permutexvar_epi8(0x0101010101010101ull, bytes_48_to_55, bitValues);
      vprimes6 = _mm512_add_epi64(base, vprimes6);
      _mm512_storeu_si512(&primes64[48], vprimes6);

      if (primeCount <= 56)
        continue;

      __m512i vprimes7 = _mm512_maskz_permutexvar_epi8(0x0101010101010101ull, bytes_56_to_63, bitValues);
      vprimes7 = _mm512_add_epi64(base, vprimes7);
      _mm512_storeu_si512(&primes64[56], vprimes7);
    }

    low_ = low;
    sieveIdx_ = sieveIdx;
    *size = i;
  }
  while (*size == 0);
}

#endif

} // namespace
