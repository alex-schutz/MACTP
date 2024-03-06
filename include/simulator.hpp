// Copyright Alex Schutz 2023-24

#pragma once
#include <SimInterface.h>

#include <random>
#include <sample.hpp>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace CTP {

class MACTP : public SimInterface {
 private:
  int N;
  std::vector<int> _nodes;
  std::unordered_map<int, int> _edges;
  std::unordered_map<std::pair<int, int>, double> _stochastic_edges;
  std::vector<int> _abs_goals;
  std::vector<int> _nonabs_goals;
  std::unordered_map<int, int> _goal_ach;
  double _move_reward, _idle_reward, _service_reward, _bad_action_reward;
  double _discount_factor;
  mutable std::mt19937 _rng;

  StateSpace stateSpace;
  StateSpace observationSpace;
  StateSpace actionSpace;
  DiscreteSample<int> initialStateDist;

 public:
  /**
   * @brief Construct a new MACTP object
   *
   * @param N number of agents
   * @param nodes set of node numbers
   * @param edges set of edges between nodes (directed, include both ways)
   * @param stochastic_edges edges & probability of being blocked (undirected,
   * include only one way)
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
  MACTP(int N, std::vector<int> nodes, std::unordered_map<int, int> edges,
        std::unordered_map<std::pair<int, int>, double> stochastic_edges,
        std::vector<int> abs_goals, std::vector<int> nonabs_goals,
        std::unordered_map<int, int> goal_ach, double move_reward,
        double idle_reward, double service_reward, double finished_reward,
        double bad_action_reward, double discount_factor,
        uint64_t seed = std::random_device{}())
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
        observationSpace(initialiseObservationSpace()),
        actionSpace(initialiseActionSpace()),
        initialStateDist(initialiseStartDist(seed)) {}

  tuple<int, int, double, bool> Step(int sI, int aI) override;

  int SampleStartState() override { return initialStateDist.sample(); }
  int GetSizeOfObs() const override { return observationSpace.size(); }
  int GetSizeOfA() const override { return actionSpace.size(); }
  double GetDiscount() const override { return _discount_factor; }
  int GetNbAgent() const override { return N; }

  int IndividualToJointActionIndex(vector<int> &action_indices) const override {
    return actionSpace.combineIndices(action_indices);
  }

  vector<int> JointToIndividualObsIndices(int JoI) const override {
    return observationSpace.splitIndices(JoI);
  }

  vector<int> JointToIndividualActionIndices(int JaI) const override {
    return actionSpace.splitIndices(JaI);
  }

  int GiveLocalObsIndex(int JoI, int agentI) const override {
    return observationSpace.nthFactorIndex(JoI, agentI);
  }

  int GiveLocalActionIndex(int JaI, int agentI) const override {
    return actionSpace.nthFactorIndex(JaI, agentI);
  }

  int GetSizeOfJointObs() const override { return observationSpace.size(); }

  int GetSizeOfLocalObs(int agentI) const override {
    return observationSpace.at(agentI).size();
  }

  int GetSizeOfLocalA(int agentI) const override {
    return actionSpace.at(agentI).size();
  }

  int GetNextNI(int nI, int oI, int size_optimizing_fsc,
                vector<vector<vector<double>>> &Eta_fsc);

  double PolicyEvaluation(
      vector<vector<vector<double>>> &eta_fsc_optimizing_agent,
      vector<FSCNode> &FscNodes_optimizing_agent) override;

 private:
  StateSpace initialiseStateSpace() const;
  StateSpace initialiseObservationSpace() const;
  StateSpace initialiseActionSpace() const;
  DiscreteSample<int> initialiseStartDist(uint64_t seed) const;

  double applyActionToState(int sI, int aI, int &sNext) const;
  int observeState(int sNext) const;
  bool checkComplete(int sNext) const;
};

}  // namespace CTP
