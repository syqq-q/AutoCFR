import attr
import numpy as np
from autocfr.program.program_types import nptype
import copy
from PokerRL.game._.tree.PublicTree import PublicTree, PublicTreeHUNL
from PokerRL.game.wrappers import HistoryEnvBuilder
from PokerRL.rl.rl_util import get_env_cls_from_str
from PokerRL.game.games import DiscretizedNLHoldemSubGame
# from scripts.load_best_algorithm import load_best_algorithm
from autocfr.cfr.cfr_algorithm import load_algorithm


class CFRHunlSolver:
    def __init__(self, 
                 name,
                 chief_handle,
                 game_cls,
                 agent_bet_set,
                 algorithm,
                 other_agent_bet_set=None,
                 starting_stack_sizes=None
                 ):
        self._name = name
        self._n_seats = 2

        self._chief_handle = chief_handle

        if starting_stack_sizes is None:
            self._starting_stack_sizes = [game_cls.DEFAULT_STACK_SIZE]
        else:
            self._starting_stack_sizes = copy.deepcopy(starting_stack_sizes)
        self._game_cls_str = game_cls.__name__

        if other_agent_bet_set is None:
            self._env_args = [
                game_cls.ARGS_CLS(n_seats=self._n_seats,
                                  starting_stack_sizes_list=[start_chips for _ in range(self._n_seats)],
                                  bet_sizes_list_as_frac_of_pot=agent_bet_set, 
                                  )
                for start_chips in self._starting_stack_sizes
            ]
        else:
            self._env_args = [
                game_cls.ARGS_CLS(n_seats=self._n_seats,
                                  starting_stack_sizes_list=[start_chips for _ in range(self._n_seats)],
                                  bet_sizes_list_as_frac_of_pot=agent_bet_set,
                                  other_bet_sizes_list_as_frac_of_pot=other_agent_bet_set
                                  )
                for start_chips in self._starting_stack_sizes
            ]
        self._env_bldrs = [
            HistoryEnvBuilder(env_cls=get_env_cls_from_str(self._game_cls_str),
                              env_args=self._env_args[s])

            for s in range(len(self._starting_stack_sizes))
        ]
        
        # self.game = game
        self.algorithm = algorithm
        # self.algorithm_of_round4 = load_best_algorithm(31)[1]
        self.algorithm_of_round4 = load_algorithm("dcfr_plus")
        self.iter_count = None
        if issubclass(game_cls, DiscretizedNLHoldemSubGame):
            self.tree_cls = PublicTreeHUNL
        else:
            self.tree_cls = PublicTree
        self._trees = [
            self.tree_cls(env_bldr=self._env_bldrs[idx],
                       stack_size=self._env_args[idx].starting_stack_sizes_list,
                       stop_at_street=None)
            for idx in range(len(self._env_bldrs))
        ]

        for tree in self._trees:
            tree.build_tree()
            print("Tree with stack size", tree.stack_size, "has", tree.n_nodes, "nodes out of which", tree.n_nonterm,
                  "are non-terminal.")
        
        self._exps_curr_total = [
            self._chief_handle.create_experiment(
                self._name + "_Curr_S" + str(self._starting_stack_sizes[s]) + "_total_" + type(self.algorithm).__name__)
            for s in range(len(self._starting_stack_sizes))
        ]

        self._exps_avg_total = [
            self._chief_handle.create_experiment(
                self._name + "_Avg_total_S" + str(self._starting_stack_sizes[s]) + "_" + type(self.algorithm).__name__)
            for s in range(len(self._starting_stack_sizes))
        ]

        self._exp_all_averaged_curr_total = self._chief_handle.create_experiment(
            self._name + "_Curr_total_averaged_" + type(self.algorithm).__name__)

        self._exp_all_averaged_avg_total = self._chief_handle.create_experiment(
            self._name + "_Avg_total_averaged_" + type(self.algorithm).__name__)
        self.reset()

    def reset(self):
        self.iter_count = 0
        for p in range(self._n_seats):
            self._reset_player(p_id=p)
        for t_idx in range(len(self._trees)):
            self._trees[t_idx].fill_uniform_random()

        self._compute_cfv()
        # self._log_curr_strat_expl()
    
    def _reset_player(self, p_id):
        def __reset(_node, _p_id):
            if _node.p_id_acting_next == _p_id:
                # regrets and strategies only need to be stored for one player at each node
                _node.data = {
                    "regret": None,
                    "avg_strat": None
                }
                _node.strategy = None
                _node.avg_strat_sum = None

            for c in _node.children:
                __reset(c, _p_id=_p_id)

        for t_idx in range(len(self._trees)):
            __reset(self._trees[t_idx].root, _p_id=p_id)
    
    def iteration(self):
        self.iter_count += 1
        print("iteration:", self.iter_count)
        for i in range(self._n_seats):
            self._compute_cfv()
            self._compute_regrets(p_id=i)
            self._update_trees(current_player=i)
            self._update_reach_probs()

        self._compute_cfv()
        self._evaluate_avg_strats()
    
    def _update_trees(self, current_player):
        for t_idx in range(len(self._trees)):
            def get_average_strategy(cumu_strategy):
                vec = cumu_strategy
                size = vec.shape
                N = size[1]
                p_sum = np.expand_dims(np.sum(vec, axis=1), axis=1).repeat(N, axis=1)
                with np.errstate(divide='ignore', invalid='ignore'):
                    res = np.where(
                        p_sum > 0.0,
                        vec / p_sum,
                        np.full(shape=size, fill_value=1.0/N, dtype=np.float32)
                    )
                return res
            def execute_program(_node, current_player, algorithm):
                input_values_of_names = {
                    "ins_regret": nptype(_node.ins_regret),
                    "reach_prob": nptype(np.expand_dims(_node.reach_probs[current_player], axis=1)),#结构不一样
                    "iters": nptype(self.iter_count),
                    "cumu_regret": nptype(_node.data["regret"]),
                    "strategy": nptype(_node.strategy),
                    "cumu_strategy": nptype(_node.avg_strat_sum),
                }
                
                #print("ins_regret:", nptype(_node.ins_regret))
                #print("\ncumu_strategy:", nptype(_node.avg_strat_sum))

                output_values_of_names = self.algorithm.execute(input_values_of_names)
                _node.ins_regret *= 0
                #_node.reach_probs[current_player].fill(0)
                _node.data["regret"] = output_values_of_names["cumu_regret"]
                _node.strategy = output_values_of_names["strategy"]
                _node.avg_strat_sum = output_values_of_names["cumu_strategy"]
                _node.data["avg_strat"] = get_average_strategy(_node.avg_strat_sum)

                #print("\nreach_prob:", nptype(_node.reach_probs[current_player]))
                #print("\niters:", nptype(self.iter_count))
                #print("\ncumu_regret:", nptype(_node.data["regret"]))
                #print("\nstrategy:", nptype(_node.strategy))
                #print("\ncumu_strategy:", nptype(_node.data["avg_strat_sum"]))
            def _update(_node, current_player):
                if _node.p_id_acting_next == current_player:
                    #判断当前轮次，决定用什么算法更新，传一个algorithm参数
                    # print("current_round:", _node.current_round)
                    if _node.current_round == 2:
                        algo = self.algorithm
                    elif _node.current_round == 3:
                        algo = self.algorithm_of_round4
                    execute_program(_node=_node, current_player=current_player, algorithm=algo)
                for c in _node.children:
                    _update(_node=c, current_player=current_player)
            
            _node = self._trees[t_idx].root
            _update(_node=_node, current_player=current_player)

    def _compute_cfv(self):
        for t_idx in range(len(self._trees)):
            self._trees[t_idx].compute_ev()

    def _compute_regrets(self, p_id):
        for t_idx in range(len(self._trees)):
            def __compute_evs(_node):
                # EV of each action
                N_ACTIONS = len(_node.children)
                ev_all_actions = np.zeros(shape=(self._env_bldrs[t_idx].rules.RANGE_SIZE, N_ACTIONS), dtype=np.float32)
                for i, child in enumerate(_node.children):
                    ev_all_actions[:, i] = child.ev[p_id]

                # EV if playing by curr strat
                strat_ev = _node.ev[p_id]
                strat_ev = np.expand_dims(strat_ev, axis=-1).repeat(N_ACTIONS, axis=-1)

                return strat_ev, ev_all_actions      
            def _fill_regrets(_node):
                if _node.p_id_acting_next == p_id:
                    strat_ev, ev_all_actions = __compute_evs(_node=_node)
                    if self.iter_count == 1:
                        _node.data["regret"] = np.zeros_like(ev_all_actions)
                    _node.ins_regret = ev_all_actions - strat_ev
                for c in _node.children:
                    _fill_regrets(c)
            _fill_regrets(self._trees[t_idx].root)          

    def _evaluate_avg_strats(self):
        expl_totals = []
        for t_idx in range(len(self._trees)):
            METRIC = self._env_bldrs[t_idx].env_cls.WIN_METRIC
            eval_tree = self.tree_cls(env_bldr=self._env_bldrs[t_idx],
                                   stack_size=self._env_args[t_idx].starting_stack_sizes_list,
                                   stop_at_street=None,
                                   is_debugging=False,
                                   )
            eval_tree.build_tree()

            def _fill(_node_eval, _node_train):
                if _node_eval.p_id_acting_next != eval_tree.CHANCE_ID and (not _node_eval.is_terminal):
                    _node_eval.strategy = np.copy(_node_train.data["avg_strat"])
                    #print("\n_node_eval.strategy:", _node_eval.strategy)
                    #print("\nsum of strategy:", np.sum(_node_eval.strategy, axis=1))
                    assert np.allclose(np.sum(_node_eval.strategy, axis=1), 1, atol=0.0001)

                for c_eval, c_train in zip(_node_eval.children, _node_train.children):
                    _fill(_node_eval=c_eval, _node_train=c_train)

            # sets up some stuff; we overwrite strategy afterwards
            eval_tree.fill_uniform_random()

            # fill with strat
            _fill(_node_eval=eval_tree.root, _node_train=self._trees[t_idx].root)
            eval_tree.update_reach_probs()

            # compute EVs
            eval_tree.compute_ev()

            eval_tree.export_to_file(name=self._name + "_Avg_" + str(self.iter_count))

            # log
            expl_p = [
                float(eval_tree.root.exploitability[p]) * self._env_bldrs[t_idx].env_cls.EV_NORMALIZER
                for p in range(eval_tree.n_seats)
            ]
            expl_total = sum(expl_p) / eval_tree.n_seats
            expl_totals.append(expl_total)

            self._chief_handle.add_scalar(self._exps_avg_total[t_idx],
                                          "Evaluation/" + METRIC, self.iter_count, expl_total)

        expl_total_averaged = sum(expl_totals) / float(len(expl_totals))
        self.expl = expl_total_averaged
        self._chief_handle.add_scalar(self._exp_all_averaged_avg_total,
                                      "Evaluation/" + METRIC, self.iter_count, expl_total_averaged)

    def _update_reach_probs(self):
        for t_idx in range(len(self._trees)):
            self._trees[t_idx].update_reach_probs()

    def average_policy(self):
        def wrap(h):
            feature = h.information_state_string()
            s = self.states[feature]
            average_policy = {}
            for index, a in enumerate(h.legal_actions()):
                average_policy[a] = s.average_strategy[index]
            return average_policy
        return wrap
