#include <gtest/gtest.h>

#include <iostream>
#include <mactp/path.hpp>

using namespace CTP;

class MockGraph : public ShortestPathFasterAlgorithm {
 public:
  MockGraph() = default;

  std::vector<std::tuple<int, double, int>> getEdges(int node) const override {
    if (node == 1) {
      return {{2, 3.0, 1}, {3, 6.0, 2}};
    } else if (node == 2) {
      return {{3, 2.0, 3}};
    } else if (node == 3) {
      return {{1, -1.0, 4}, {3, 0.5, 5}};
    }
    return {};
  }
};

TEST(ShortestPathFasterAlgorithmTest, Constructor) {
  MockGraph graph;
  {
    const auto [costs, pred] = graph.calculate(1, 10);
    EXPECT_EQ(costs.size(), 3);
    EXPECT_EQ(costs.at(1), 0.0);
    EXPECT_EQ(costs.at(2), 3.0);
    EXPECT_EQ(costs.at(3), 5.0);

    const auto [costs2, pred2] = graph.calculate(1, 1);
    EXPECT_EQ(costs2.size(), 3);
    EXPECT_EQ(costs.at(1), 0.0);
    EXPECT_EQ(costs2.at(2), 3.0);
    EXPECT_EQ(costs2.at(3), 6.0);
  }

  {
    const auto [costs, pred] = graph.calculate(3, 10);
    EXPECT_EQ(costs.size(), 3);
    EXPECT_EQ(costs.at(1), -1.0);
    EXPECT_EQ(costs.at(2), 2.0);
    EXPECT_EQ(costs.at(3), 0.0);
  }

  {
    const auto [costs, pred] = graph.calculate(4, 10);
    EXPECT_EQ(costs.size(), 1);
    EXPECT_EQ(costs.at(4), 0.0);
  }
}

TEST(ShortestPathFasterAlgorithmTest, ReconstructPath) {
  MockGraph graph;
  const auto [costs, pred] = graph.calculate(1, 10);
  const auto path = graph.reconstructPath(3, pred);
  EXPECT_EQ(path.size(), 3);
  EXPECT_EQ(path[0].first, 1);
  EXPECT_EQ(path[0].second, 1);
  EXPECT_EQ(path[1].first, 2);
  EXPECT_EQ(path[1].second, 3);
  EXPECT_EQ(path[2].first, 3);
  EXPECT_EQ(path[2].second, -1);
}
