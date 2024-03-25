// Copyright Alex Schutz 2024

#include <assert.h>

#include <mactp/bounds.hpp>

namespace CTP {

std::vector<std::tuple<int, double, int>> PathToTerminal::getEdges(
    int state) const {
  std::vector<std::tuple<int, double, int>> edges;
  for (int a = 0; a < sim->GetSizeOfA(); ++a) {
    const auto& [sNext, o, reward, done] = sim->Step(state, a);
    edges.push_back({sNext, -reward, a});
    if (done) terminalStates.insert(sNext);
  }
  if (terminalStates.contains(state)) edges.push_back({-1, 0, -1});
  return edges;
}

std::tuple<int, double> PathToTerminal::path(int source, int max_depth) const {
  const auto [costs, pred] = calculate(source, max_depth);
  if (costs.contains(-1)) {
    const std::vector<std::pair<int, int>> p = reconstructPath(-1, pred);
    paths[source] = p;
    if (p.size() > 1) return {p.at(p.size() - 2).first, -costs.at(-1)};
  }
  const auto best_move = std::max_element(
      costs.cbegin(), costs.cend(),
      [](const std::pair<int, double>& p1, const std::pair<int, double>& p2) {
        return p1.second < p2.second;
      });
  return *best_move;
}

std::unordered_map<int, std::shared_ptr<PathToTerminal::PathNode>>
PathToTerminal::buildPathTree() const {
  std::shared_ptr<PathNode> rootNode = createPathNode(-1, {});  // dummy root
  std::unordered_map<int, std::shared_ptr<PathNode>> startNodes;

  for (const auto& [source, path] : paths) {
    std::shared_ptr<PathNode> nextNode = rootNode;

    for (auto it = path.rbegin(); it != path.rend(); ++it) {
      int action = it->second;
      int state = it->first;
      if (it == path.rbegin()) {
        assert(action == -1);
        assert(state == -1);
        continue;
      }

      std::shared_ptr<PathNode> currentNode =
          findOrCreateNode(nextNode, action);
      currentNode->states.insert(state);
      nextNode = currentNode;
    }
    startNodes[source] = nextNode;
  }

  // disconnect the dummy root node
  for (const auto& child : rootNode->prevNodes) child->nextNode = nullptr;

  return startNodes;
}

std::shared_ptr<PathToTerminal::PathNode> PathToTerminal::createPathNode(
    int action, const std::unordered_set<int>& states) const {
  return std::make_shared<PathNode>(PathNode{action, states});
}

std::shared_ptr<PathToTerminal::PathNode> PathToTerminal::findActionChild(
    std::shared_ptr<PathNode> node, int action) const {
  for (const auto& n : node->prevNodes)
    if (n->action == action) return n;
  return nullptr;
}

std::shared_ptr<PathToTerminal::PathNode> PathToTerminal::findOrCreateNode(
    std::shared_ptr<PathNode> nextNode, int action) const {
  const auto node = findActionChild(nextNode, action);
  if (!node) {
    std::shared_ptr<PathNode> newNode = createPathNode(action, {});
    newNode->nextNode = nextNode;
    nextNode->prevNodes.push_back(newNode);
    return newNode;
  }
  return node;
}

double beliefUpperBound(const std::unordered_map<int, double>& belief,
                        SimInterface* sim, int max_depth) {
  auto ptt = PathToTerminal(sim);

  double V_upper_bound = 0.0;
  for (const auto& [state, prob] : belief)
    V_upper_bound += prob * std::get<1>(ptt.path(state, max_depth));

  return V_upper_bound;
}

}  // namespace CTP
