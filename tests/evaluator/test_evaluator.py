import time
import pytest
from autocfr.cfr.cfr_algorithm import load_algorithm
from autocfr.cfr.cfr_solver import CFRSolver
from autocfr.evaluator.evaluator import evaluate_algorithm, Evaluator, VecEvaluator, GroupVecEvaluator
from autocfr.utils import load_game
from PokerRL.game import bet_sets
from PokerRL.game.games import DiscretizedNLHoldemSubGame3, DiscretizedNLHoldemSubGame4
from PokerRL.rl.base_cls.workers.ChiefBase import ChiefBase

def get_game_config():
    game_config = dict(
        long_name="kuhn_poker",
        game_name="kuhn_poker",
        params={"players": 2},
        max_score=1.2,
        iterations=1
    )
    return game_config


def test_compute_conv():
    game_config = get_game_config()
    # game = load_game(game_config)
    algorithm = load_algorithm("cfr")
    solver = ConservativeSolver(game, algorithm)
    # chief = ChiefBase(t_prof=None)
    # game_name = game_config["subgame_name"]
    # name = "{}_{}".format("cfr", game_name)
    # solver = CFRSolver(
    #     name=name,
    #     algorithm=algorithm,
    #     game_cls=game_config[game_name],
    #     agent_bet_set=bet_sets.B_3,
    #     other_agent_bet_set=bet_sets.B_2,
    #     chief_handle=chief
    #     )
    solver.iteration()
    conv = solver.expl / 1000
    print("conv:", conv)
    assert pytest.approx(conv, 1e-3) == 0.45833


def test_evaluate_algorithm():
    game_config = get_game_config()
    algorithm = load_algorithm("cfr")
    result = evaluate_algorithm(game_config, algorithm)
    print("\nresult:", result)
    assert result["status"] == "succ"
    #assert pytest.approx(result["conv"], 1e-3) == 0.06869
    #assert result["game_config"]["long_name"] == "kuhn_poker"

    algorithm = load_algorithm("cfr_error")
    result = evaluate_algorithm(game_config, algorithm)
    assert result["status"] == "fail"


def test_evaluator():
    evaluator = Evaluator(0)
    game_config = get_game_config()
    algorithm = load_algorithm("cfr")
    task = {"agent_index": 1, "algorithm": algorithm, "game_config": game_config}
    result = evaluator.run(task)
    assert result["status"] == "succ"
    assert result["agent_index"] == 1
    assert result["worker_index"] == 0
    #assert pytest.approx(result["conv"], 1e-3) == 0.06869


def atest_vec_evaluator():
    import ray
    ray.init()
    vec_evaluator = VecEvaluator(1)
    game_config = get_game_config()
    algorithm = load_algorithm("cfr")
    vec_evaluator.eval_algorithm(
        1, algorithm, game_config
    )
    for i in range(3):
        time.sleep(1)
        result = vec_evaluator.get_evaluating_result()
        if result is not None:
            assert result["status"] == "succ"
            assert result["agent_index"] == 1
            assert result["game_config"]["long_name"] == "kuhn_poker"
            assert pytest.approx(result["conv"], 1e-3) == 0.06869
    ray.shutdown()


def atest_group_vec_evaluator():
    import ray
    ray.init()
    vec_evaluator = GroupVecEvaluator(2)
    game_configs = [
        get_game_config(),
        get_game_config(),
    ]
    algorithm = load_algorithm("cfr")
    vec_evaluator.eval_algorithm_parallel(
        1, algorithm, game_configs
    )
    algorithm = load_algorithm("cfr_error")
    vec_evaluator.eval_algorithm_parallel(
        1, algorithm, game_configs
    )
    for i in range(3):
        time.sleep(1)
        result = vec_evaluator.get_evaluating_result()
        if result is not None:
            print(result)
    ray.shutdown()
