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

}  // namespace CTP
