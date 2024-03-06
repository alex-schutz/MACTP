// Copyright Alex Schutz 2023-24

#include <algorithm>
#include <simulator.hpp>

namespace CTP {

tuple<int, int, double, bool> MACTP::Step(int sI, int aI) {
  int sNext;
  const double reward = applyActionToState(sI, aI, sNext);
  const int oI = observeState(sNext);
  const bool finished = checkComplete(sNext);
  // sI_next, oI, Reward, Done
  return std::tuple<int, int, double, bool>(sNext, oI, reward, finished);
}

int MACTP::GetNextNI(int nI, int oI, int size_optimizing_fsc,
                     vector<vector<vector<double>>> &Eta_fsc) {
  std::uniform_real_distribution<double> unif(0, 1);
  double random_p = unif(_rng);
  double sum_p = 0.0;
  for (int n_newI = 0; n_newI < size_optimizing_fsc; n_newI++) {
    double pr_trans = Eta_fsc[nI][oI][n_newI];
    sum_p += pr_trans;
    if (sum_p >= random_p) {
      return n_newI;
    }
  }

  // self node
  return nI;
}

double MACTP::PolicyEvaluation(
    vector<vector<vector<double>>> &eta_fsc_optimizing_agent,
    vector<FSCNode> &FscNodes_optimizing_agent) {
  double sum_accumlated_rewards = 0.0;
  int restart_evaluation = 1;
  double discount = GetDiscount();
  double epsilon = 0.0001;

  int size_optimizing_fsc = FscNodes_optimizing_agent.size();
  double average_acc_rewards = 0;

  tuple<int, int, double, bool> res_step;
  int i = 0;
  while (i < restart_evaluation) {
    i += 1;
    int depth = 0;
    int sampled_eI_start = SampleStartState();
    int eI = sampled_eI_start;
    int nI_optimizing_agent = 0;
    int oI = -1;
    double total_discount = pow(discount, depth);
    double temp_res = 0;
    while (total_discount > epsilon) {
      int aI = FscNodes_optimizing_agent[nI_optimizing_agent].best_aI;
      res_step = Step(eI, aI);
      double reward = get<2>(res_step);
      temp_res += reward * total_discount;
      total_discount *= discount;
      eI = get<0>(res_step);
      oI = get<1>(res_step);
      nI_optimizing_agent =
          GetNextNI(nI_optimizing_agent, oI, size_optimizing_fsc,
                    eta_fsc_optimizing_agent);
      depth += 1;
    }
    sum_accumlated_rewards += temp_res;
    average_acc_rewards = sum_accumlated_rewards / i;
  }

  return average_acc_rewards;
}

static std::string agentLoc2str(int a) { return "a" + std::to_string(a); }

static std::string edge2str(int i, int j) {
  return "e" + std::to_string(i) + "_" + std::to_string(j);
}

static std::string agentGoal2str(int agent, int goal) {
  return "g" + std::to_string(goal) + "_a" + std::to_string(agent);
}

StateSpace MACTP::initialiseStateSpace() const {
  std::map<std::string, std::vector<int>> state_factors;
  // location of each agent
  for (int a = 0; a < N; ++a) state_factors[agentLoc2str(a)] = _nodes;

  // stochastic edge status
  for (const auto &[edge, _] : _stochastic_edges)
    state_factors[edge2str(edge.first, edge.second)] = {0, 1};

  // goal service status (per agent)
  for (int a = 0; a < N; ++a) {
    for (const auto &g : _abs_goals)
      state_factors[agentGoal2str(a, g)] = {0, 1};
    for (const auto &g : _nonabs_goals)
      state_factors[agentGoal2str(a, g)] = {0, 1};
  }
  return StateSpace(state_factors);
}

StateSpace MACTP::initialiseIndividualObservationSpace() const {
  // agent can observe any element from state space or -1 (unknown)
  std::map<std::string, std::vector<int>> observation_factors;
  // location of each agent
  std::vector<int> agent_locs = _nodes;
  agent_locs.push_back(-1);
  for (int a = 0; a < N; ++a) observation_factors[agentLoc2str(a)] = agent_locs;

  // stochastic edge status
  for (const auto &[edge, _] : _stochastic_edges)
    observation_factors[edge2str(edge.first, edge.second)] = {0, 1, -1};

  // goal service status (per agent)
  for (int a = 0; a < N; ++a) {
    for (const auto &g : _abs_goals)
      observation_factors[agentGoal2str(a, g)] = {0, 1, -1};
    for (const auto &g : _nonabs_goals)
      observation_factors[agentGoal2str(a, g)] = {0, 1, -1};
  }
  return StateSpace(observation_factors);
}

StateSpace MACTP::initialiseObservationSpace() const {
  // observation space consists of index into individual observation space for
  // each agent
  std::vector<int> obs_indices;
  for (int i = 0; i < individualObservationSpace.size(); ++i)
    obs_indices.push_back(i);

  std::map<std::string, std::vector<int>> observation_factors;
  for (int a = 0; a < N; ++a)
    observation_factors[std::to_string(a)] = obs_indices;

  return StateSpace(observation_factors);
}

StateSpace MACTP::initialiseActionSpace() const {
  // individual agent actions (all agents homogeneous):
  // move to (or stay at) node i: i
  // service goal: -1
  std::vector<int> agent_actions = _nodes;
  agent_actions.push_back(-1);

  std::map<std::string, std::vector<int>> agent_factors;
  for (int a = 0; a < N; ++a) agent_factors[std::to_string(a)] = agent_actions;

  return StateSpace(agent_factors);
}

DiscreteSample<int> MACTP::initialiseStartDist(uint64_t seed) const {
  std::map<std::string, int> known_state;

  // agents start at origin (first node in list)
  for (int a = 0; a < N; ++a) known_state[agentLoc2str(a)] = _nodes[0];

  // goals start unserviced
  for (int a = 0; a < N; ++a) {
    for (const auto &g : _abs_goals) known_state[agentGoal2str(a, g)] = 0;
    for (const auto &g : _nonabs_goals) known_state[agentGoal2str(a, g)] = 0;
  }

  // calculate probability of state based on independent probability of edges
  std::unordered_map<int, double> initial_states;
  const int n = _stochastic_edges.size();
  for (int i = 0; i < (1 << n); ++i) {  // compute all bool permutations
    std::map<std::string, int> status = known_state;
    double probability = 1.0f;
    int j = 0;
    for (const auto &[edge, prob] : _stochastic_edges) {
      const bool traversable = static_cast<bool>(i & (1 << j));
      status[edge2str(edge.first, edge.second)] = traversable;
      probability *= traversable ? (1.0f - prob) : prob;
      ++j;
    }
    initial_states[stateSpace.stateIndex(status)] = probability;
  }
  return DiscreteSample<int>(initial_states);
}

bool MACTP::validAction(int agent, int action, int state) const {
  int loc = stateSpace.getStateFactorElem(state, agentLoc2str(agent));
  if (loc == action) return true;  // idle always allowed
  if (action == -1)                // service goal only if not already serviced
    return stateSpace.getStateFactorElem(state, agentGoal2str(agent, loc)) == 0;

  // check if agent is stuck in absorbing goal (cannot move)
  for (const auto &g : _abs_goals)
    if (stateSpace.getStateFactorElem(state, agentGoal2str(agent, g)) == 1)
      return false;

  // move action, check if edge exists
  return nodesAdjacent(loc, action, state);
}

double MACTP::applyAgentActionToState(int state, int agent, int action,
                                      int &sNext) const {
  sNext = state;
  if (!validAction(agent, action, state)) return _bad_action_reward;

  int loc = stateSpace.getStateFactorElem(state, agentLoc2str(agent));
  if (action == -1) {  // service goal
    sNext = stateSpace.updateStateFactor(state, agentGoal2str(agent, loc), 1);
    // return idle reward if goal was already achieved
    return goalAchieved(loc, state) ? _idle_reward : _service_reward;
  }
  // move/idle
  sNext = stateSpace.updateStateFactor(state, agentLoc2str(agent), action);
  if (loc == action) return _idle_reward;
  return _move_reward;
}

double MACTP::applyActionToState(int sI, int aI, int &sNext) const {
  double reward = 0;
  int nextState = sI;
  const std::vector<int> actions = actionSpace.splitIndices(aI);
  for (int a = 0; a < N; ++a)
    reward += applyAgentActionToState(nextState, a, actions[a], nextState);

  sNext = nextState;
  return reward;
}

bool MACTP::goalAchieved(int goal, int state) const {
  int k = 0;
  for (int a = 0; a < N; ++a)
    k += stateSpace.getStateFactorElem(state, agentGoal2str(a, goal));

  return k >= _goal_ach.at(goal);
}

bool MACTP::nodesAdjacent(int a, int b, int state) const {
  if (std::find(_edges.begin(), _edges.end(), std::pair(a, b)) == _edges.end())
    return false;  // edge does not exist

  // check if edge is stochastic
  const auto stoch_edge = a < b ? std::pair(a, b) : std::pair(b, a);
  const auto stoch_ptr = _stochastic_edges.find(stoch_edge);
  if (stoch_ptr == _stochastic_edges.end()) return true;  // deterministic edge

  // check if edge is unblocked
  return stateSpace.getStateFactorElem(
             state, edge2str(stoch_edge.first, stoch_edge.second)) ==
         1;  // traversable
}

int MACTP::localObservation(int state, int agent) const {
  std::map<std::string, int> observation;

  const int loc = stateSpace.getStateFactorElem(state, agentLoc2str(agent));
  for (int a = 0; a < N; ++a)
    observation[agentLoc2str(a)] = (a == agent) ? loc : -1;

  // stochastic edge status
  for (const auto &[edge, _] : _stochastic_edges) {
    if (nodesAdjacent(loc, edge.first, state) ||
        nodesAdjacent(loc, edge.second, state)) {
      const int status = stateSpace.getStateFactorElem(
          state, edge2str(edge.first, edge.second));
      observation[edge2str(edge.first, edge.second)] = status;
    } else
      observation[edge2str(edge.first, edge.second)] = -1;
  }
  // goal service status (per agent)
  for (int a = 0; a < N; ++a) {
    if (a == agent) {
      for (const auto &g : _abs_goals)
        observation[agentGoal2str(a, g)] =
            stateSpace.getStateFactorElem(state, agentGoal2str(a, g));
      for (const auto &g : _nonabs_goals)
        observation[agentGoal2str(a, g)] =
            stateSpace.getStateFactorElem(state, agentGoal2str(a, g));
    } else {
      for (const auto &g : _abs_goals) observation[agentGoal2str(a, g)] = -1;
      for (const auto &g : _nonabs_goals) observation[agentGoal2str(a, g)] = -1;
    }
  }
}

int MACTP::communicateObservation(int ob1, int ob2) const {
  int ob = ob1;
  const std::vector<int> ob2_components =
      individualObservationSpace.splitIndices(ob2);
  int i = 0;
  for (const auto &[name, _] : individualObservationSpace.map())
    individualObservationSpace.updateStateFactorIndex(ob, name, i++);
  return ob;
}

int MACTP::observeState(int sNext) const {}

// all REACHABLE goals complete
// all agents stuck in absorbing goals
bool MACTP::checkComplete(int sNext) const {}

std::vector<int> MACTP::computeReachableGoals(int state) const {}

}  // namespace CTP
