// Copyright Alex Schutz 2024

#pragma once
#include <limits>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace CTP {

class InfMap {
 private:
  std::unordered_map<int, double> _map;

 public:
  InfMap() = default;
  const std::unordered_map<int, double>& map() const { return _map; }
  double& operator[](int key) {
    auto it = _map.find(key);
    if (it != _map.end()) return it->second;
    return _map.insert({key, std::numeric_limits<double>::infinity()})
        .first->second;
  }
  auto begin() { return _map.begin(); }
  auto end() { return _map.end(); }
  auto size() const { return _map.size(); }
  auto empty() const { return _map.empty(); }
  void clear() { _map.clear(); }
};

/**
 * @brief Implements the Shortest Path Faster Algorithm (SPFA)
 *
 * Inherit from this class to calculate the shortest path from source to all
 * reachable nodes, according to edges given by `getEdges`.
 */
class ShortestPathFasterAlgorithm {
 private:
  mutable InfMap d;
  mutable std::unordered_map<int, bool> inQueue;
  mutable std::unordered_map<int, int> depth;
  mutable std::unordered_map<int, std::pair<int, int>> predecessor;

  void initParams() const;

 public:
  ShortestPathFasterAlgorithm() = default;

  /**
   * @brief Get the edges and weights out of `node`.
   *
   * Returns a list of destination nodes, associated edge costs and edge number.
   *
   * The edge number is only used to label which edge is taken when
   * reconstructing the path, and can be set arbitrarily.
   */
  virtual std::vector<std::tuple<int, double, int>> getEdges(
      int node) const = 0;

  /**
   * @brief Calculate the shortest path between source and each node up to
   * maximum depth N
   *
   * Returns a map of <destination, cost> pairs and a map of <node,
   * <predecessor_node, edge>> pairs. Use the latter with `reconstructPath`.
   */
  std::tuple<std::unordered_map<int, double>,
             std::unordered_map<int, std::pair<int, int>>>
  calculate(int source, int N) const;

  /// @brief Reconstruct path of <node, next_edge> to target
  std::vector<std::pair<int, int>> reconstructPath(
      int target,
      const std::unordered_map<int, std::pair<int, int>>& paths) const;
};

}  // namespace CTP
