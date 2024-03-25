// Copyright Alex Schutz 2024

#pragma once
#include <SimInterface.h>

#include <mactp/path.hpp>
#include <memory>
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

  struct PathNode {
    int action;
    std::unordered_set<int> states;
    std::shared_ptr<PathNode> nextNode;
    std::vector<std::shared_ptr<PathNode>> prevNodes;
  };

  /// @brief Return the starting action node for each previously calculated path
  /// leading to a sequence of action nodes, where common sub-paths have been
  /// combined
  std::unordered_map<int, std::shared_ptr<PathNode>> buildPathTree() const;

 private:
  SimInterface* sim;
  mutable std::unordered_set<int> terminalStates;
  mutable std::unordered_map<int, std::vector<std::pair<int, int>>> paths;

  std::shared_ptr<PathNode> createPathNode(
      int action, const std::unordered_set<int>& states) const;

  std::shared_ptr<PathNode> findActionChild(std::shared_ptr<PathNode> node,
                                            int action) const;

  std::shared_ptr<PathNode> findOrCreateNode(std::shared_ptr<PathNode> nextNode,
                                             int action) const;
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
