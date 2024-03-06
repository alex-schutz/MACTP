// Copyright Alex Schutz 2023-24

#include <simulator.hpp>

namespace CTP {

tuple<int, int, double, bool> MACTP::Step(int sI, int aI) {
  int sNext;
  const double reward = applyActionToState(sI, aI, sNext);
  const int oI = observeState(sNext);
  const bool finished = checkComplete(sNext);
  // sI_next, oI, Reward, Done
  return std::tuple<int, int, double, bool>(sNext, oI, reward, finished);
}

int MACTP::GetNextNI(int nI, int oI, int size_optimizing_fsc,
                     vector<vector<vector<double>>> &Eta_fsc) {
  std::uniform_real_distribution<double> unif(0, 1);
  double random_p = unif(_rng);
  double sum_p = 0.0;
  for (int n_newI = 0; n_newI < size_optimizing_fsc; n_newI++) {
    double pr_trans = Eta_fsc[nI][oI][n_newI];
    sum_p += pr_trans;
    if (sum_p >= random_p) {
      return n_newI;
    }
  }

  // self node
  return nI;
}

double MACTP::PolicyEvaluation(
    vector<vector<vector<double>>> &eta_fsc_optimizing_agent,
    vector<FSCNode> &FscNodes_optimizing_agent) {
  double sum_accumlated_rewards = 0.0;
  int restart_evaluation = 1;
  double discount = GetDiscount();
  double epsilon = 0.0001;

  int size_optimizing_fsc = FscNodes_optimizing_agent.size();
  double average_acc_rewards = 0;

  tuple<int, int, double, bool> res_step;
  int i = 0;
  while (i < restart_evaluation) {
    i += 1;
    int depth = 0;
    int sampled_eI_start = SampleStartState();
    int eI = sampled_eI_start;
    int nI_optimizing_agent = 0;
    int oI = -1;
    double total_discount = pow(discount, depth);
    double temp_res = 0;
    while (total_discount > epsilon) {
      int aI = FscNodes_optimizing_agent[nI_optimizing_agent].best_aI;
      res_step = Step(eI, aI);
      double reward = get<2>(res_step);
      temp_res += reward * total_discount;
      total_discount *= discount;
      eI = get<0>(res_step);
      oI = get<1>(res_step);
      nI_optimizing_agent =
          GetNextNI(nI_optimizing_agent, oI, size_optimizing_fsc,
                    eta_fsc_optimizing_agent);
      depth += 1;
    }
    sum_accumlated_rewards += temp_res;
    average_acc_rewards = sum_accumlated_rewards / i;
  }

  return average_acc_rewards;
}

StateSpace MACTP::initialiseStateSpace() const {}

StateSpace MACTP::initialiseObservationSpace() const {}

StateSpace MACTP::initialiseActionSpace() const {}

DiscreteSample<int> MACTP::initialiseStartDist(uint64_t seed) const {}

double MACTP::applyActionToState(int sI, int aI, int &sNext) const {}

int MACTP::observeState(int sNext) const {}

bool MACTP::checkComplete(int sNext) const {}

}  // namespace CTP
