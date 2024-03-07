// Copyright Alex Schutz 2023-24

#pragma once
#include <SimInterface.h>

#include <mactp/sample.hpp>
#include <mactp/statespace.hpp>
#include <random>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace CTP {

struct pairhash {
 public:
  template <typename T, typename U>
  std::size_t operator()(const std::pair<T, U> &x) const {
    return std::hash<T>()(x.first) ^ std::hash<U>()(x.second);
  }
};

class MACTP : public SimInterface {
 private:
  int N;
  std::vector<int> _nodes;
  std::vector<std::pair<int, int>> _edges;
  std::unordered_map<std::pair<int, int>, double, pairhash> _stochastic_edges;
  std::vector<int> _abs_goals;
  std::vector<int> _nonabs_goals;
  std::unordered_map<int, int> _goal_ach;
  double _move_reward, _idle_reward, _service_reward, _bad_action_reward;
  double _discount_factor;
  mutable std::mt19937 _rng;

  StateSpace stateSpace;
  StateSpace individualObservationSpace;
  StateSpace observationSpace;
  StateSpace actionSpace;
  DiscreteSample<int> initialStateDist;

 public:
  /**
   * @brief Construct a new MACTP object
   *
   * @param N number of agents
   * @param nodes set of (non-negative) node numbers (first node is considered
   * the origin)
   * @param edges set of all edges between nodes (undirected,
   * include only one way with pairs in ascending node order)
   * @param stochastic_edges bloackable edges & probability of being blocked
   * (undirected, include only one way with pairs in ascending node order)
   * @param abs_goals set of absorbing goals
   * @param nonabs_goals set of non-absorbing goals
   * @param goal_ach <goal, k> where k is the number of times a goal must be
   * serviced
   * @param move_reward single agent reward for moving between nodes
   * @param idle_reward single agent reward for idling
   * @param service_reward single agent reward for servicing a (non-achieved)
   * goal
   * @param bad_action_reward single agent reward for performing an invalid
   * action
   * @param discount_factor discount factor (within [0, 1])
   * @param seed seed for rng
   */
  MACTP(int N, std::vector<int> nodes, std::vector<std::pair<int, int>> edges,
        std::unordered_map<std::pair<int, int>, double, pairhash>
            stochastic_edges,
        std::vector<int> abs_goals, std::vector<int> nonabs_goals,
        std::unordered_map<int, int> goal_ach, double move_reward,
        double idle_reward, double service_reward, double bad_action_reward,
        double discount_factor, uint64_t seed = std::random_device{}())
      : N(N),
        _nodes(nodes),
        _edges(edges),
        _stochastic_edges(stochastic_edges),
        _abs_goals(abs_goals),
        _nonabs_goals(nonabs_goals),
        _goal_ach(goal_ach),
        _move_reward(move_reward),
        _idle_reward(idle_reward),
        _service_reward(service_reward),
        _bad_action_reward(bad_action_reward),
        _discount_factor(discount_factor),
        _rng(seed),
        stateSpace(initialiseStateSpace()),
        individualObservationSpace(initialiseIndividualObservationSpace()),
        observationSpace(initialiseObservationSpace()),
        actionSpace(initialiseActionSpace()),
        initialStateDist(initialiseStartDist(seed)) {}

  std::tuple<int, int, double, bool> Step(int sI, int aI) override;

  int SampleStartState() override { return initialStateDist.sample(); }
  int GetSizeOfObs() const override { return observationSpace.size(); }
  int GetSizeOfA() const override { return actionSpace.size(); }
  double GetDiscount() const override { return _discount_factor; }
  int GetNbAgent() const override { return N; }

  int IndividualToJointActionIndex(
      std::vector<int> &action_indices) const override {
    return actionSpace.combineIndices(action_indices);
  }

  std::vector<int> JointToIndividualObsIndices(int JoI) const override {
    return observationSpace.splitIndices(JoI);
  }

  std::vector<int> JointToIndividualActionIndices(int JaI) const override {
    return actionSpace.splitIndices(JaI);
  }

  int GiveLocalObsIndex(int JoI, int agentI) const override {
    return observationSpace.getStateFactorIndex(JoI, std::to_string(agentI));
  }

  int GiveLocalActionIndex(int JaI, int agentI) const override {
    return actionSpace.getStateFactorIndex(JaI, std::to_string(agentI));
  }

  int GetSizeOfJointObs() const override { return observationSpace.size(); }

  int GetSizeOfLocalObs(int agentI) const override {
    return observationSpace.map().at(std::to_string(agentI)).size();
  }

  int GetSizeOfLocalA(int agentI) const override {
    return actionSpace.map().at(std::to_string(agentI)).size();
  }

 private:
  StateSpace initialiseStateSpace() const;
  StateSpace initialiseIndividualObservationSpace() const;
  StateSpace initialiseObservationSpace() const;
  StateSpace initialiseActionSpace() const;
  DiscreteSample<int> initialiseStartDist(uint64_t seed) const;

  /// @brief Return true if a and b are connected by an unblocked edge in state.
  /// A node is considered adjacent to itself
  bool nodesAdjacent(int a, int b, int state) const;
  /// @brief Return true if the goal has been serviced by enough agents
  bool goalAchieved(int goal, int state) const;
  /// @brief Return true if agent can perform action in state
  bool validAction(int agent, int action, int state) const;
  /// @brief Return the reward for agent performing action in state. Also return
  /// the next state sNext
  double applyAgentActionToState(int state, int agent, int action,
                                 int &sNext) const;
  /// @brief Return the reward for performing joint action aI in state sI. Also
  /// return the next state sNext
  double applyActionToState(int sI, int aI, int &sNext) const;
  /// @brief Return the index of the local observation of agent in state
  int localObservation(int state, int agent) const;
  /// @brief Return a matrix indicating if agent i and agent j are able to
  /// communicate given the state
  std::vector<std::vector<bool>> communicationAvailable(int state) const;
  /// @brief Return the individual observation number produced by combining
  /// observations ob1 and ob2
  int communicateObservation(int ob1, int ob2) const;
  /// @brief Return the observation number given the state
  int observeState(int state) const;
  /// @brief Return true if all goals have been achieved or cannot be achieved
  bool checkComplete(int state) const;
  /// @brief Compute the set of goals which are reachable from the origin
  std::vector<int> computeReachableGoals(int state) const;
  /// @brief Return true if the given node is a goal node
  bool isGoal(int loc) const;
  ///@brief Return true if dest is reachable from source
  bool reachable(int source, int dest, int state) const;
};

}  // namespace CTP
