// Copyright Alex Schutz 2024

#include <algorithm>
#include <mactp/path.hpp>
#include <queue>
#include <stdexcept>

namespace CTP {

std::vector<std::pair<int, int>> ShortestPathFasterAlgorithm::reconstructPath(
    int target) const {
  std::vector<std::pair<int, int>> path;
  std::pair<int, int> current = {target, -1};
  while (current.first != source) {
    path.push_back(current);
    const auto currentPtr = predecessor.find(current.first);
    if (currentPtr == predecessor.cend()) break;
    current = currentPtr->second;
  }
  if (current.first == source) path.push_back(current);

  std::reverse(path.begin(), path.end());
  return path;
}

void ShortestPathFasterAlgorithm::calculate(int N) {
  std::deque<int> q;
  q.push_back(source);
  d[source] = 0.0;
  inQueue[source] = true;
  ++count[source];

  while (!q.empty()) {
    int u = q.front();
    q.pop_front();
    inQueue[u] = false;

    // Process each edge
    for (auto [v, w, label] : getEdges(u)) {
      if (d[u] + w < d[v]) {
        d[v] = d[u] + w;
        predecessor[v] = {u, label};

        // Add v to queue
        if (!inQueue[v]) {
          if (d[v] < d[q.front()])
            q.push_front(v);
          else
            q.push_back(v);

          inQueue[v] = true;
          ++count[v];
          if (count[v] >= N) return;  // max depth reached
        }
      }
    }
  }
}

}  // namespace CTP
