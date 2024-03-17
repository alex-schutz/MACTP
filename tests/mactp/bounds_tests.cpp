#include <gtest/gtest.h>

#include <iostream>
#include <mactp/bounds.hpp>
#include <mactp/simulator.hpp>

using namespace CTP;

MACTP generateSim(const std::vector<int>& nodes,
                  const std::vector<std::pair<int, int>>& edges,
                  const std::unordered_map<std::pair<int, int>, double,
                                           pairhash>& stochastic_edges,
                  const std::vector<int>& abs_goals,
                  const std::vector<int>& nonabs_goals,
                  const std::unordered_map<int, int>& goal_ach) {
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
  // 6 nodes, connected in a hexagon
  const std::vector<int> nodes = {0, 1, 2, 3, 4, 5};
  const std::vector<std::pair<int, int>> edges = {{0, 1}, {1, 2}, {2, 3},
                                                  {3, 4}, {4, 5}, {0, 5}};
  const std::unordered_map<std::pair<int, int>, double, pairhash>
      stochastic_edges = {{{1, 2}, 0.5}};
  const std::vector<int> abs_goals = {2};
  const std::vector<int> nonabs_goals = {1};
  const std::unordered_map<int, int> goal_ach = {{1, 1}, {2, 1}};
  MACTP sim = generateSim(nodes, edges, stochastic_edges, abs_goals,
                          nonabs_goals, goal_ach);
  const auto ptt = PathToTerminal(&sim);

  {  // create terminal state, path should be length 1
    const auto& [goal_node, reward] = ptt.path(23, 20);
    EXPECT_EQ(goal_node, 23);
    EXPECT_EQ(reward, 0.0);
  }
  {
    // create start state with unblocked edge
    // states should be
    // 4 -> 12 -> 14 -> 22 -> 23
    // actions should be
    // 1 -> -1 -> 2 -> -1
    const auto& [goal_node, reward] = ptt.path(4, 20);
    EXPECT_EQ(goal_node, 23);
    EXPECT_EQ(reward, 18.0);

    const std::vector<std::pair<int, int>> path_exp = {
        {4, 1}, {12, 6}, {14, 2}, {22, 6}, {23, -1}};
    const auto [costs, pred] = ptt.calculate(4, 20);
    const auto path = ptt.reconstructPath(goal_node, pred);
    EXPECT_EQ(path, path_exp);
  }
  {
    // create start state with blocked edge
    // states should be
    // 0 -> 8 -> 10 -> 2 -> 42 -> 34 -> 26 -> 18 -> 19
    // actions should be
    // 1 -> -1 -> 0 -> 5 -> 4 -> 3 -> 2 -> -1
    const auto& [goal_node, reward] = ptt.path(0, 20);
    EXPECT_EQ(goal_node, 19);
    EXPECT_EQ(reward, 14.0);

    const std::vector<std::pair<int, int>> path_exp = {
        {0, 1},  {8, 6},  {10, 0}, {2, 5},  {42, 4},
        {34, 3}, {26, 2}, {18, 6}, {19, -1}};
    const auto [costs, pred] = ptt.calculate(0, 20);
    const auto path = ptt.reconstructPath(goal_node, pred);
    EXPECT_EQ(path, path_exp);
  }
}

TEST(PathToTerminalTest, BlockedGoalPath) {
  const std::vector<int> nodes = {0, 1, 2};
  const std::vector<std::pair<int, int>> edges = {{0, 1}, {1, 2}};
  const std::unordered_map<std::pair<int, int>, double, pairhash>
      stochastic_edges = {{{1, 2}, 0.5}};
  const std::vector<int> abs_goals = {2};
  const std::vector<int> nonabs_goals = {1};
  const std::unordered_map<int, int> goal_ach = {{1, 1}, {2, 1}};
  MACTP sim = generateSim(nodes, edges, stochastic_edges, abs_goals,
                          nonabs_goals, goal_ach);
  const auto ptt = PathToTerminal(&sim);
  {
    // create start state with unblocked edge
    // states should be
    // 4 -> 12 -> 14 -> 22 -> 23
    // actions should be
    // 1 -> -1 -> 2 -> -1
    const auto& [goal_node, reward] = ptt.path(4, 20);
    EXPECT_EQ(goal_node, 23);
    EXPECT_EQ(reward, 18.0);

    const std::vector<std::pair<int, int>> path_exp = {
        {4, 1}, {12, 3}, {14, 2}, {22, 3}, {23, -1}};
    const auto [costs, pred] = ptt.calculate(4, 20);
    const auto path = ptt.reconstructPath(goal_node, pred);
    EXPECT_EQ(path, path_exp);
  }
  {
    // create start state with blocked edge
    // end state will be 10 (cannot reach goal 2)
    const auto& [end_node, reward] = ptt.path(0, 20);
    EXPECT_EQ(end_node, 10);
    EXPECT_EQ(reward, 9.0);
  }
}

TEST(PathToTerminalTest, UnachievableGoalPath) {
  const std::vector<int> nodes = {0, 1, 2};
  const std::vector<std::pair<int, int>> edges = {{0, 1}, {1, 2}};
  const std::unordered_map<std::pair<int, int>, double, pairhash>
      stochastic_edges = {};
  const std::vector<int> abs_goals = {};
  const std::vector<int> nonabs_goals = {1, 2};
  const std::unordered_map<int, int> goal_ach = {{1, 2}, {2, 1}};
  MACTP sim = generateSim(nodes, edges, stochastic_edges, abs_goals,
                          nonabs_goals, goal_ach);
  const auto ptt = PathToTerminal(&sim);
  {
    // end state will be 11
    const auto& [end_node, reward] = ptt.path(0, 20);
    EXPECT_EQ(end_node, 11);
    EXPECT_EQ(reward, 18.0);
  }
}
