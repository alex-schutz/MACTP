#include "statespace.hpp"

#include <assert.h>

namespace CTP {

int StateSpace::combineIndices(const std::vector<int> &i) const {
  int j = 0;
  int s = 0;
  for (const auto &[name, sf] : _sf_map) {
    s += i[j++] * prodSF.at(name);
  }
  assert(j == i.size());
  return s;
}

std::vector<int> StateSpace::splitIndices(int i) const {
  std::vector<int> s;
  for (const auto &[name, sf] : _sf_map)
    s.push_back(getStateFactorIndex(i, name));
  return s;
}

int StateSpace::getStateFactorIndex(int sI, std::string sf_name) const {
  return (sI / prodSF.at(sf_name)) % _sf_map.at(sf_name).size();
}

int StateSpace::getStateFactorElem(int sI, std::string sf_name) const {
  return _sf_map.at(sf_name).at(getStateFactorIndex(sI, sf_name));
}

int StateSpace::updateStateFactorIndex(int sI, std::string sf_name,
                                       int new_sf_elem_idx) const {
  const int curr_idx = getStateFactorIndex(sI, sf_name);
  const int delta = new_sf_elem_idx - curr_idx;
  return sI + delta * prodSF.at(sf_name);
}

int StateSpace::updateStateFactor(int sI, std::string sf_name,
                                  int new_elem) const {
  return updateStateFactorIndex(sI, sf_name,
                                _sf_map.at(sf_name).getIndex(new_elem));
}

int StateSpace::stateIndex(const std::map<std::string, int> &state) const {
  int s = 0;
  for (const auto &[name, sf] : _sf_map)
    s += sf.getIndex(state.at(name)) * prodSF.at(name);
  return s;
}

std::map<std::string, int> StateSpace::at(int sI) const {
  std::map<std::string, int> s;
  for (const auto &[name, sf] : _sf_map) s[name] = getStateFactorElem(sI, name);
  return s;
}

std::map<std::string, IndexMap<int>> StateSpace::mapStateFactors(
    const std::map<std::string, std::vector<int>> &factors) const {
  std::map<std::string, IndexMap<int>> map;
  for (const auto &[name, vals] : factors)
    map.emplace(name, IndexMap<int>(vals));
  return map;
}

std::map<std::string, int> StateSpace::calculateProdSF() const {
  std::vector<int> _prodSF_vec = {1};
  for (auto sf = _sf_map.crbegin(); sf != _sf_map.crend(); ++sf)
    _prodSF_vec.insert(_prodSF_vec.begin(), _prodSF_vec[0] * sf->second.size());

  std::map<std::string, int> _prodSF;
  int i = 1;
  for (const auto &[name, sf] : _sf_map) {
    _prodSF[name] = _prodSF_vec[i++];
  }

  return _prodSF;
}

size_t StateSpace::calcSize() const {
  size_t p = 1;
  for (const auto &[name, sf] : _sf_map) p *= sf.size();
  return p;
}
}  // namespace CTP
