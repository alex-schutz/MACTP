// Copyright Alex Schutz 2023-24

#include <algorithm>
#include <mactp/simulator.hpp>
#include <sstream>
#include <unordered_set>

namespace CTP {

std::tuple<int, int, double, bool> MACTP::Step(int sI, int aI) {
  int sNext;
  const double reward = applyActionToState(sI, aI, sNext);
  const int oI = observeState(sNext);
  const bool finished = checkComplete(sNext);
  // sI_next, oI, Reward, Done
  return std::tuple<int, int, double, bool>(sNext, oI, reward, finished);
}

static std::string agentLoc2str(int a) { return "a" + std::to_string(a); }

static std::string edge2str(std::pair<int, int> e) {
  return "e" + std::to_string(e.first) + "_" + std::to_string(e.second);
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
    state_factors[edge2str(edge)] = {0, 1};

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
    observation_factors[edge2str(edge)] = {0, 1, -1};

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
      status[edge2str(edge)] = traversable;
      probability *= traversable ? (1.0f - prob) : prob;
      ++j;
    }
    initial_states[stateSpace.stateIndex(status)] = probability;
  }
  return DiscreteSample<int>(initial_states);
}

bool MACTP::isGoal(int loc) const {
  if (std::find(_abs_goals.cbegin(), _abs_goals.cend(), loc) !=
      _abs_goals.cend())
    return true;
  return (std::find(_nonabs_goals.cbegin(), _nonabs_goals.cend(), loc) !=
          _nonabs_goals.cend());
}

bool MACTP::validAction(int agent, int action, int state) const {
  int loc = stateSpace.getStateFactorElem(state, agentLoc2str(agent));
  if (loc == action) return true;  // idle always allowed

  // check if agent is stuck in absorbing goal (cannot move/service other goals)
  for (const auto &g : _abs_goals)
    if (stateSpace.getStateFactorElem(state, agentGoal2str(agent, g)) == 1)
      return false;

  if (action == -1) {  // service goal only if not already serviced
    if (!isGoal(loc)) return false;
    return stateSpace.getStateFactorElem(state, agentGoal2str(agent, loc)) == 0;
  }

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
  for (int a = 0; a < N; ++a)
    reward += applyAgentActionToState(
        nextState, a, actionSpace.getStateFactorElem(aI, std::to_string(a)),
        nextState);

  sNext = nextState;
  return reward;
}

bool MACTP::goalAchieved(int goal, int state) const {
  if (!isGoal(goal)) return false;
  int k = 0;
  for (int a = 0; a < N; ++a)
    k += stateSpace.getStateFactorElem(state, agentGoal2str(a, goal));

  return k >= _goal_ach.at(goal);
}

bool MACTP::nodesAdjacent(int a, int b, int state) const {
  if (a == b) return true;
  const auto edge = a < b ? std::pair(a, b) : std::pair(b, a);
  if (std::find(_edges.begin(), _edges.end(), edge) == _edges.end())
    return false;  // edge does not exist

  // check if edge is stochastic
  const auto stoch_ptr = _stochastic_edges.find(edge);
  if (stoch_ptr == _stochastic_edges.end()) return true;  // deterministic edge

  // check if edge is unblocked
  return stateSpace.getStateFactorElem(state, edge2str(edge)) ==
         1;  // traversable
}

int MACTP::localObservation(int state, int agent) const {
  std::map<std::string, int> observation;

  const int loc = stateSpace.getStateFactorElem(state, agentLoc2str(agent));
  for (int a = 0; a < N; ++a)
    observation[agentLoc2str(a)] = (a == agent) ? loc : -1;

  // stochastic edge status
  for (const auto &[edge, _] : _stochastic_edges) {
    if (loc == edge.first || loc == edge.second) {
      const int status = stateSpace.getStateFactorElem(state, edge2str(edge));
      observation[edge2str(edge)] = status;
    } else {
      observation[edge2str(edge)] = -1;
    }
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
  return individualObservationSpace.stateIndex(observation);
}

int MACTP::communicateObservation(int ob1, int ob2) const {
  std::map<std::string, int> ob = individualObservationSpace.at(ob1);
  for (auto &[name, f] : ob)
    if (f == -1) f = individualObservationSpace.getStateFactorElem(ob2, name);
  return individualObservationSpace.stateIndex(ob);
}

std::vector<std::vector<bool>> MACTP::communicationAvailable(int state) const {
  std::vector<int> agent_locs;
  for (int a = 0; a < N; ++a)
    agent_locs.push_back(stateSpace.getStateFactorElem(state, agentLoc2str(a)));

  std::vector<std::vector<bool>> comms;
  for (const auto &ei : agent_locs) {
    std::vector<bool> comms_i;
    for (const auto &ej : agent_locs)
      comms_i.push_back(nodesAdjacent(ei, ej, state));
    comms.push_back(comms_i);
  }
  return comms;
}

int MACTP::observeState(int state) const {
  const auto comms = communicationAvailable(state);

  std::vector<int> local_obs;
  for (int a = 0; a < N; ++a) local_obs.push_back(localObservation(state, a));

  std::vector<int> agent_obs = local_obs;
  for (int ai = 0; ai < N; ++ai) {
    for (int aj = 0; aj < N; ++aj) {
      if (ai == aj) continue;
      if (comms[ai][aj])
        agent_obs[ai] = communicateObservation(agent_obs[ai], local_obs[aj]);
    }
  }
  return observationSpace.combineIndices(agent_obs);
}

bool MACTP::checkComplete(int state) const {
  const std::vector<int> reachable_goals = computeReachableGoals(state);
  std::vector<int> unachieved_goals;
  for (const auto &g : reachable_goals)
    if (!goalAchieved(g, state)) unachieved_goals.push_back(g);
  if (unachieved_goals.empty()) return true;  // all reachable goals achieved

  for (const auto &g : unachieved_goals) {
    for (int a = 0; a < N; ++a) {
      // check if any agent could still service the unachieved goal if it was
      // there (has not already serviced this goal or an absorbing goal)
      const int locState =
          stateSpace.updateStateFactor(state, agentLoc2str(a), g);
      if (validAction(a, -1, locState)) return false;
    }
  }
  return true;
}

bool MACTP::reachable(int source, int dest, int state) const {
  std::unordered_set<int> toDoSet = {source};
  std::unordered_set<int> doneSet;
  while (!toDoSet.empty()) {
    const int node = *toDoSet.begin();
    toDoSet.erase(toDoSet.begin());
    doneSet.insert(node);
    for (const auto &n : _nodes) {
      if (n == node) continue;
      if (!nodesAdjacent(n, node, state)) continue;
      if (n == dest) return true;
      if (std::find(doneSet.cbegin(), doneSet.cend(), n) == doneSet.cend()) {
        toDoSet.insert(n);
      }
    }
  }

  return false;
}

std::vector<int> MACTP::computeReachableGoals(int state) const {
  std::vector<int> all_goals = _abs_goals;
  all_goals.insert(all_goals.end(), _nonabs_goals.begin(), _nonabs_goals.end());

  std::vector<int> reachable_goals;
  for (const auto &g : all_goals)
    if (reachable(_nodes[0], g, state)) reachable_goals.push_back(g);

  return reachable_goals;
}

std::ostream &operator<<(std::ostream &os,
                         const std::map<std::string, int> &map) {
  os << "{\n";
  for (const auto &pair : map) os << pair.first << ": " << pair.second << ",\n";
  os << "}";
  return os;
}

std::string MACTP::stateToString(int state) const {
  std::stringstream ss;
  ss << stateSpace.at(state);
  return ss.str();
}
std::string MACTP::actionToString(int action) const {
  std::stringstream ss;
  ss << actionSpace.at(action);
  return ss.str();
}
std::string MACTP::observationToString(int obs) const {
  std::stringstream ss;
  ss << "{\n";
  for (const auto [agent, ob] : observationSpace.at(obs))
    ss << agent << ": " << individualObservationSpace.at(ob) << ",\n";
  ss << "}";
  return ss.str();
}

}  // namespace CTP
