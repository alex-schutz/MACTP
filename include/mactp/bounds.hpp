// Copyright Alex Schutz 2024

#pragma once
#include <SimInterface.h>

#include <mactp/path.hpp>
#include <unordered_set>

namespace CTP {

/**
 * @brief Method to calculate the shortest path in a deterministic POMDP from a
 * given state to a terminal state.
 */
class PathToTerminal : public ShortestPathFasterAlgorithm {
 public:
  PathToTerminal(SimInterface* sim) : sim(sim) {}

  /// @brief Return the best terminal state and associated reward reachable from
  /// source within max_depth steps
  std::tuple<int, double> path(int source, int max_depth) const;

  std::vector<std::tuple<int, double, int>> getEdges(int state) const override;

 private:
  SimInterface* sim;
  mutable std::unordered_set<int> terminalStates;
};

}  // namespace CTP
