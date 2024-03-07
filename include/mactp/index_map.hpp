// Copyright Alex Schutz 2024

#pragma once
#include <algorithm>
#include <memory>
#include <unordered_map>
#include <vector>

namespace CTP {

/**
 * @brief Creates a two-way look up between a list of T and their
 * indices.
 */
template <typename T, typename Hash = std::hash<T>>
class IndexMap {
 public:
  IndexMap(const std::vector<T> &v) {
    indexToElement = v;
    size_t i = 0;
    for (const auto &e : v) elementToIndex.insert({e, i++});
  }

  /// @brief Return the index of the given element within the list
  size_t getIndex(const T &element) const { return elementToIndex.at(element); }

  /// @brief Get the element at the given index
  const T &at(size_t index) const { return indexToElement.at(index); }

  size_t size() const { return indexToElement.size(); }

  /// @brief Return the map of elements to indices
  const std::unordered_map<T, size_t, Hash> &map() const {
    return elementToIndex;
  }

  /// @brief Return a vector of pointers to elements in order
  const std::vector<T> &vector() const { return indexToElement; }

 private:
  std::unordered_map<T, size_t, Hash> elementToIndex;
  std::vector<T> indexToElement;
};

}  // namespace CTP
