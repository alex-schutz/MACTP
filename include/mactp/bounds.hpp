// Copyright Alex Schutz 2024

#pragma once
#include <SimInterface.h>

#include <mactp/path.hpp>
#include <unordered_map>
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
  /// source within max_depth steps. Does not account for discount factor.
  std::tuple<int, double> path(int source, int max_depth) const;

  std::vector<std::tuple<int, double, int>> getEdges(int state) const override;

 private:
  SimInterface* sim;
  mutable std::unordered_set<int> terminalStates;
};

/**
 * @brief Return an upper bound for the value of the belief.
 *
 * The upper bound value of the belief is the expected sum of each possible
 * state's MDP value
 *
 * @param belief A map from state indices to probabilities.
 * @param sim A POMDP simulator object
 * @param max_depth Max depth of a simulation run
 */
double beliefUpperBound(const std::unordered_map<int, double>& belief,
                        SimInterface* sim, int max_depth);

}  // namespace CTP
