// Copyright Alex Schutz 2024

#pragma once
#include <mactp/index_map.hpp>
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

namespace CTP {

class StateSpace {
 public:
  StateSpace(const std::map<std::string, std::vector<int>> &factors)
      : _sf_map(mapStateFactors(factors)),
        _size(calcSize()),
        prodSF(calculateProdSF()) {}

  size_t size() const { return _size; }

  int combineIndices(const std::vector<int> &i) const;
  std::vector<int> splitIndices(int i) const;

  /// @brief Retrieve the state number given a state
  int stateIndex(const std::map<std::string, int> &state) const;

  /// @brief Retrieve the state given a state number
  std::map<std::string, int> at(int sI) const;

  /**
   * @brief Return the index within the state factor of the element of state
   * number sI
   *
   * For example, with state factors
   * ```
   * "loc_name": ["river", "land"]
   * "height": [0.0, 0.1, 0.2]
   * ```
   * then the state `sI` = 3 corresponds to {"land", 0.0}
   * so getStateFactorIndex(3, "loc_name") returns 1
   * and getStateFactorIndex(3, "height") returns 0
   */
  int getStateFactorIndex(int sI, std::string sf_name) const;

  /// @brief Return the element of the given state factor for state number sI
  int getStateFactorElem(int sI, std::string sf_name) const;

  /**
   * @brief Given state number sI, return the number of the state where the
   * index of the given state factor is set to`new_sf_elem_idx`
   *
   * * For example, with state factors
   * ```
   * "loc_name": ["river", "land"]
   * "height": [0.0, 0.1, 0.2]
   * ```
   * the state `sI` = 3 corresponds to {"land", 0}.
   * To change the state to {"land", 0.2}, we want to update the index of the
   * "height" state factor to 2.
   * So we call `updateStateFactorIndex(3, "height", 2)`
   * and get a returned state index of 5.
   */
  int updateStateFactorIndex(int sI, std::string sf_name,
                             int new_sf_elem_idx) const;

  /// @brief Given state number sI, return the number of the state where the
  /// element of the given state factor is set to `new_elem`
  int updateStateFactor(int sI, std::string sf_name, int new_elem) const;

  const std::map<std::string, IndexMap<int>> &map() const { return _sf_map; }

 private:
  std::map<std::string, IndexMap<int>> _sf_map;  // ordered by name
  size_t _size;
  std::map<std::string, int> prodSF;  // cumulative product of sf lengths

  /// @brief create a map of state factor names to index maps
  std::map<std::string, IndexMap<int>> mapStateFactors(
      const std::map<std::string, std::vector<int>> &factors) const;

  /// @brief calculate a cumulative product of state factor lengths
  std::map<std::string, int> calculateProdSF() const;

  size_t calcSize() const;
};

}  // namespace CTP
