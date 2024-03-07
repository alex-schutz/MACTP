// Copyright Alex Schutz 2023-24

#include <simulator.hpp>

using namespace CTP;

// 6 nodes, connected in a hexagon
const std::vector<int> nodes = {0, 1, 2, 3, 4, 5};
const std::vector<std::pair<int, int>> edges = {{0, 1}, {1, 2}, {2, 3},
                                                {3, 4}, {4, 5}, {0, 5}};

// 1 stochastic edge
const std::unordered_map<std::pair<int, int>, double> stochastic_edges = {
    {{1, 2}, 0.5}};

// 1 goal, service once
const std::vector<int> abs_goals = {2};
const std::vector<int> nonabs_goals = {0};
const std::unordered_map<int, int> goal_ach = {{2, 1}};

// 1 agent
const int N = 1;

// rewards
double move_reward = -1;
double idle_reward = -1;
double service_reward = 10;
double bad_action_reward = -50;
double discount_factor = 0.95;

const auto ctp = MACTP(N, nodes, edges, stochastic_edges, abs_goals,
                       nonabs_goals, goal_ach, move_reward, idle_reward,
                       service_reward, bad_action_reward, discount_factor);
