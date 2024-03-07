// Copyright Alex Schutz 2023-24

#pragma once

#include <algorithm>
#include <deque>
#include <iostream>
#include <random>
#include <unordered_map>
#include <vector>

namespace CTP {

/**
 * @brief Sampling of a discrete probability distribution
 *
 * Implements Vose's Alias Method which allows for O(1) generation of samples
 * from a discrete probability distribution.
 *
 * @tparam T
 */
template <class T, class Hash = std::hash<T>, class KeyEqual = std::equal_to<T>>
class DiscreteSample {
 public:
  /// @brief Construct the discrete sampler
  /// @param probabilities a map from items to probabilities. Sum of
  /// probabilities is assumed to be 1.
  DiscreteSample(
      const std::unordered_map<T, double, Hash, KeyEqual>& probabilities,
      uint64_t seed = std::random_device{}())
      : _n(probabilities.size()), _generator(seed), _uniform(0.0, 1.0) {
    _items.reserve(probabilities.size());
    _item_probs.reserve(probabilities.size());
    for (auto kv : probabilities) {
      _items.push_back(kv.first);
      _item_probs.push_back(kv.second);
    }

    initialiseArrays();
  }

  /// @brief Return a sample from the distribution
  T sample() const noexcept {
    const size_t i = _n * _uniform(_generator);
    const double u = _uniform(_generator);
    return (u > _prob.at(i)) ? _items.at(_alias[i]) : _items.at(i);
  }

 private:
  size_t _n;
  std::vector<T> _items;
  std::vector<double> _item_probs;
  mutable std::mt19937 _generator;  // mersenne_twister_engine for RNG
  mutable std::uniform_real_distribution<double> _uniform;
  std::vector<size_t> _alias;
  std::vector<double> _prob;

  /// @brief initialise the _alias and _prob vectors using Vose's Alias Method
  /// See https://www.keithschwarz.com/darts-dice-coins/
  void initialiseArrays() {
    // initialise arrays and stacks
    _alias.resize(_n, -1);
    _prob.resize(_n, 0.0);
    std::deque<size_t> small, large;

    // scale probabilities then assign to stack based on size
    for (auto& p : _item_probs) p *= _n;
    for (size_t i = 0; i < _n; ++i) {
      const double p = _item_probs.at(i);
      if (p < 1)
        small.push_back(i);
      else
        large.push_back(i);
    }

    // pair small and large probabilities to assign aliases
    while (!small.empty() && !large.empty()) {
      const int l = small.back();
      small.pop_back();
      const int g = large.back();
      large.pop_back();
      const double p_l = _item_probs.at(l);
      const double p_g = _item_probs.at(g);
      _prob[l] = p_l;
      _alias[l] = g;
      _item_probs[g] = (p_g + p_l) - 1;
      if (_item_probs.at(g) < 1)
        small.push_back(g);
      else
        large.push_back(g);
    }
    // deal with remaining items (must be prob 1)
    while (!large.empty()) {
      const int g = large.back();
      large.pop_back();
      _prob[g] = 1.0;
    }

    // this is only possible due to numerical instability
    while (!small.empty()) {
      const int l = small.back();
      small.pop_back();
      _prob[l] = 1.0;
    }
  }
};

}  // namespace CTP
