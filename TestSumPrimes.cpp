


uint64_t main()
{
  uint64_t start = 0;
  uint64_t stop = 1ull << 38;
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

  return acc;
}

