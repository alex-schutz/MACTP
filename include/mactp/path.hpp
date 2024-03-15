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
  double& operator[](int key) { return _map[key]; }
  const double& operator[](int key) const {
    const static double inf = std::numeric_limits<double>::infinity();
    auto it = _map.find(key);
    return (it != _map.end()) ? it->second : inf;
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
  int source;
  InfMap d;
  std::unordered_map<int, bool> inQueue;
  std::unordered_map<int, int> count;
  std::unordered_map<int, std::pair<int, int>> predecessor;

  void calculate(int N);

 public:
  ShortestPathFasterAlgorithm(int source, int max_nodes) : source(source) {
    calculate(max_nodes);
  }

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

  /// @brief Return the solution of the shortest path between source and each
  /// reachable node
  const std::unordered_map<int, double>& solution() const { return d.map(); }

  /// @brief Reconstruct path of <node, next_edge> to target
  std::vector<std::pair<int, int>> reconstructPath(int target) const;
};

}  // namespace CTP
