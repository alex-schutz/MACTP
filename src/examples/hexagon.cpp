// Copyright Alex Schutz 2023-24

#include <mactp/simulator.hpp>

#include "BeliefParticles.h"
#include "ExtendedGenerativeModel.h"
#include "FSC.h"
#include "MCJESP.h"
#include "ParserDecPOMDPSparse.h"
#include "ParserPOMDPSparse.h"
#include "Planner.h"
#include "SimModel.h"
#include "TreeNode.h"

using namespace std;
using namespace CTP;

const vector<string> heuristic_type_names = {"MA", "MD", "MS"};
const vector<double> params_pomcp_c = {2.0, 5.0, 2.0, 2.0, 0.5};
const vector<int> params_pomcp_nb_rollout = {1, 200, 100, 200, 1};
const vector<int> params_pomcp_depth = {20, 40, 20, 15, 10};
const double pomcp_epsilon = 0.01;

std::ostream &operator<<(std::ostream &os,
                         const std::map<std::string, int> &myMap) {
  os << "{\n";
  for (const auto &pair : myMap) {
    os << pair.first << ": " << pair.second << ",\n";
  }
  os << "}";
  return os;
}

int main() {
  // 6 nodes, connected in a hexagon
  const std::vector<int> nodes = {0, 1, 2, 3, 4, 5};
  const std::vector<std::pair<int, int>> edges = {{0, 1}, {1, 2}, {2, 3},
                                                  {3, 4}, {4, 5}, {0, 5}};

  // 1 stochastic edge
  const std::unordered_map<std::pair<int, int>, double, pairhash>
      stochastic_edges = {{{1, 2}, 0.5}};

  // 1 goal, service once
  const std::vector<int> abs_goals = {2};
  const std::vector<int> nonabs_goals = {};
  const std::unordered_map<int, int> goal_ach = {{2, 2}};

  // 2 agents
  const int N = 2;

  // rewards
  double move_reward = -1;
  double idle_reward = -1;
  double service_reward = 10;
  double bad_action_reward = -50;
  double discount_factor = 0.95;

  // ----- 2. Loading your Dec-POMDP simulator -------
  auto simulator = MACTP(N, nodes, edges, stochastic_edges, abs_goals,
                         nonabs_goals, goal_ach, move_reward, idle_reward,
                         service_reward, bad_action_reward, discount_factor);
  const std::string decpomdp_name = "hexagon";

  SimInterface *sim = new MACTP(simulator);

  // ----- 3. (Optional) Tuning parameters -------
  int restart = 1;
  double pomcp_c = 3.0;        // default
  int nb_rollout = 1;          // default
  double epsilon = 0.01;       // default
  int max_pomcp_depth = 15;    // default
  double timeout = 5;          // default
  int max_fsc_node_size = 10;  // default
  int heuristic_type = 0;      // 0 is default M-A; 1 is M-D; 2 is M-S
  bool random_init = false;
  int max_fsc_size = 5;
  double max_belief_gap = 0.1;
  // -----------------------------------

  string outfile;
  string out_name = "MCJESP_log_" + decpomdp_name + "_";
  outfile += out_name + "maxFsc_" + to_string(max_fsc_node_size) + "_" +
             to_string(restart) + heuristic_type_names[heuristic_type] + ".csv";
  ofstream outlogs;
  outlogs.open(outfile.c_str());
  outlogs << "Restart"
          << ","
          << "Iteration"
          << ","
          << "AgentI"
          << ","
          << "Value"
          << ","
          << "IterTime"
          << ","
          << "V_max_Simulation, TotalTime, Iterations, FSC size" << endl;

  for (int i = 0; i < restart; i++) {
    MCJESP mcjesp(&simulator);
    mcjesp.Init(max_fsc_node_size, max_pomcp_depth, pomcp_c, nb_rollout,
                timeout, epsilon, max_belief_gap);

    if (!random_init) {
      mcjesp.Init_heuristic(sim, heuristic_type);
    } else {
      mcjesp.Init_random(max_fsc_size);
    }

    mcjesp.Plan(i, outlogs);
    vector<FSC> final_fscs = mcjesp.GetFinalFSCs();
    for (size_t j = 0; j < final_fscs.size(); j++) {
      string fsc_path = "./TempFiles/" + decpomdp_name + "_fsc" + to_string(j) +
                        "timeout" + to_string(timeout) + "maxN" +
                        to_string(max_fsc_node_size) + "restart" + to_string(i);
      final_fscs[j].ExportFSC(fsc_path);
    }
  }
  delete sim;

  return 0;
}
