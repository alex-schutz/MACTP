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
  PathToTerminal(SimInterface* sim, int source, int max_depth)
      : ShortestPathFasterAlgorithm(source, max_depth), sim(sim) {}

  // Highest reward path from a given state to a terminal state
  double path() const { return -solution().at(-1); }

  // The terminal state with the highest reward reachable from this state
  int terminalState() const {
    const std::vector<std::pair<int, int>> p = reconstructPath(-1);
    return p.at(p.size() - 2).first;
  }

  std::vector<std::tuple<int, double, int>> getEdges(int state) const override;

 private:
  SimInterface* sim;
  mutable std::unordered_set<int> terminalStates;
};

}  // namespace CTP
