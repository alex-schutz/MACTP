// Copyright Alex Schutz 2024

#include <mactp/bounds.hpp>

namespace CTP {

std::vector<std::tuple<int, double, int>> PathToTerminal::getEdges(
    int state) const {
  std::vector<std::tuple<int, double, int>> edges;
  if (terminalStates.contains(state)) edges.push_back({-1, 0, -1});
  for (int a = 0; a < sim->GetSizeOfA(); ++a) {
    const auto& [sNext, o, reward, done] = sim->Step(state, a);
    edges.push_back({sNext, -reward, a});
    if (done) terminalStates.insert(sNext);
  }
  return edges;
}

std::tuple<int, double> PathToTerminal::path(int source, int max_depth) const {
  const auto [costs, pred] = calculate(source, max_depth);
  if (costs.contains(-1)) {
    const std::vector<std::pair<int, int>> p = reconstructPath(-1, pred);
    if (p.size() > 1) return {p.at(p.size() - 2).first, -costs.at(-1)};
  }
  const auto best_move = std::max_element(
      costs.cbegin(), costs.cend(),
      [](const std::pair<int, double>& p1, const std::pair<int, double>& p2) {
        return p1.second < p2.second;
      });
  return *best_move;
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
