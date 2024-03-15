#include <gtest/gtest.h>

#include <iostream>
#include <mactp/bounds.hpp>
#include <mactp/simulator.hpp>

using namespace CTP;

MACTP generateSim() {
  // 6 nodes, connected in a hexagon
  const std::vector<int> nodes = {0, 1, 2, 3, 4, 5};
  const std::vector<std::pair<int, int>> edges = {{0, 1}, {1, 2}, {2, 3},
                                                  {3, 4}, {4, 5}, {0, 5}};

  const std::unordered_map<std::pair<int, int>, double, pairhash>
      stochastic_edges = {{{1, 2}, 0.5}};

  const std::vector<int> abs_goals = {2};
  const std::vector<int> nonabs_goals = {1};
  const std::unordered_map<int, int> goal_ach = {{1, 1}, {2, 1}};

  const int N = 1;

  double move_reward = -1;
  double idle_reward = -1;
  double service_reward = 10;
  double bad_action_reward = -50;
  double discount_factor = 0.95;

  return MACTP(N, nodes, edges, stochastic_edges, abs_goals, nonabs_goals,
               goal_ach, move_reward, idle_reward, service_reward,
               bad_action_reward, discount_factor);
}

TEST(PathToTerminalTest, Path) {
  MACTP sim = generateSim();
  const auto ptt = PathToTerminal(&sim);

  {  // create terminal state, path should be length 1
    const auto& [goal_node, reward] = ptt.path(23, 20);
    EXPECT_EQ(goal_node, 23);
    EXPECT_EQ(reward, 0.0);
  }
  {
    // create start state with unblocked edge, actions should be
    // 0 -> 1 -> -1 -> 2 -> -1
    const auto& [goal_node, reward] = ptt.path(1, 20);
    EXPECT_EQ(goal_node, 23);
    EXPECT_EQ(reward, 18.0);
  }
  {
    // create start state with blocked edge, actions should be
    // 0 -> 1 -> -1 -> 0 -> 5 -> 4 -> 3 -> 2 -> -1
    const auto& [goal_node, reward] = ptt.path(0, 20);
    EXPECT_EQ(goal_node, 19);
    EXPECT_EQ(reward, 14.0);
  }
}
