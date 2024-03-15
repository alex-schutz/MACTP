// Copyright Alex Schutz 2024

#include <algorithm>
#include <iostream>
#include <mactp/path.hpp>
#include <queue>
#include <stdexcept>

namespace CTP {

std::vector<std::pair<int, int>> ShortestPathFasterAlgorithm::reconstructPath(
    int target,
    const std::unordered_map<int, std::pair<int, int>>& paths) const {
  std::vector<std::pair<int, int>> path;
  std::pair<int, int> current = {target, -1};
  while (true) {
    path.push_back(current);
    const auto currentPtr = predecessor.find(current.first);
    if (currentPtr == predecessor.cend()) break;
    current = currentPtr->second;
  }

  std::reverse(path.begin(), path.end());
  return path;
}

void ShortestPathFasterAlgorithm::initParams() const {
  d = {};
  inQueue = {};
  depth = {};
  predecessor = {};
}

std::tuple<std::unordered_map<int, double>,
           std::unordered_map<int, std::pair<int, int>>>
ShortestPathFasterAlgorithm::calculate(int source, int N) const {
  initParams();

  std::deque<int> q;
  q.push_back(source);
  d[source] = 0.0;
  inQueue[source] = true;
  depth[source] = 0;

  while (!q.empty()) {
    int u = q.front();
    q.pop_front();
    inQueue[u] = false;

    // Process each edge
    for (auto [v, w, label] : getEdges(u)) {
      depth[v] = std::max(depth[v], depth[u] + 1);
      if (depth[v] > N) return {d.map(), predecessor};  // max depth reached
      if (d[u] + w < d[v]) {
        d[v] = d[u] + w;
        predecessor[v] = {u, label};

        // Add v to queue
        if (!inQueue[v]) {
          if (q.empty() || d[v] < d[q.front()])
            q.push_front(v);
          else
            q.push_back(v);
          inQueue[v] = true;
        }
      }
    }
  }
  return {d.map(), predecessor};
}

}  // namespace CTP
