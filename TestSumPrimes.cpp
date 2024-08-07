
#include <primesieve.hpp>
#include <iostream>
#include <numeric>
#include <string>

int main(int argc, char *argv[])
{
  assert(argc == 2);
  uint64_t StartExp = std::stoul(argv[1]);
  
  uint64_t start = 1ull << StartExp;
  uint64_t stop = start + (1ull << 40);
  uint64_t acc = 0;
  primesieve::iterator it(start);
  it.generate_next_primes();

  for (; it.primes_[it.size_ - 1] < stop; it.generate_next_primes())
    acc = std::accumulate(
      it.primes_,
      it.primes_ + it.size_,
      acc);
  for (std::size_t i = 0; it.primes_[i] < stop; i++)
    acc += it.primes_[i];

  std::cout << "Sum of primes in ["
    << start << ", " << stop
    << ") (mod 2^64) = " << acc
    << std::endl;

  return 0;
}

