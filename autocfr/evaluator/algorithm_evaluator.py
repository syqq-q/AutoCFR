import multiprocessing
import numpy as np
import pandas as pd
from pathlib import Path
from autocfr.cfr.cfr_solver import CFRSolver
from autocfr.cfr.conservative import ConservativeSolver
from autocfr.cfr.radical import RadicalSolver
from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability
from autocfr.utils import load_game, load_module, save_df

class AlgorithmEvaluator:
    def __init__(self, game_configs, algorithm, algo_name, num_iters=1000, eval_freq=20, print_freq=20, verbose=True):
        self.game_configs = game_configs
        self.algorithm = algorithm
        self.algo_name = algo_name
        self.num_iters = num_iters
        self.eval_freq = eval_freq
        self.print_freq = print_freq
        self.df = pd.DataFrame(
            columns=["step", "exp", "algorithm_name", "game_name", "log_exp"]
        )
        self.verbose = verbose

    def evaluate(self):
        # pool = multiprocessing.Pool(len(self.game_configs))
        # for game_config in self.game_configs:
        #     pool.apply_async(
        #         self.evaluate_run,
        #         args=(game_config, self.algorithm, self.algo_name),
        #         callback=self.record,
        #     )
        # pool.close()
        # pool.join()
        for game_config in self.game_configs:
            self.evaluate_run(game_config, self.algorithm, self.algo_name)

        # self.evaluate_run(self.game_configs, self.algorithm, self.algo_name)

    def evaluate_run(self, game_config, algorithm, algo_name):
        game_name = game_config["long_name"]
        game = load_game(game_config)
        solver = ConservativeSolver(game, algorithm)
        conv = self.calc_conv(game, solver)
        steps = [0]
        convs = [conv]
        if "iterations" in game_config:
            iters = game_config["iterations"]
        else:
            iters = self.num_iters
        for i in range(1, iters + 1):
            solver.iteration()
            if i % self.eval_freq == 0:
                conv = self.calc_conv(game, solver)
                steps.append(i)
                convs.append(conv)
                if self.verbose:
                    if i % self.print_freq == 0:
                        print(game_name, i, conv)
                file = Path("models/games") / game_name / "{}_{}.csv".format(algo_name, game_name)
                file.parent.mkdir(parents=True, exist_ok=True)
                algo_df = pd.DataFrame(
                    data=dict(
                        step=steps,
                        exp=convs,
                    )
                )
                algo_df["game_name"] = game_name
                algo_df["algorithm_name"] = algo_name
                algo_df["log_exp"] = np.log(algo_df["exp"])
                algo_df.to_csv(file, index=False)
        results = dict(
            game_name=game_config["long_name"],
            algo_name=algo_name,
            data=dict(steps=steps, convs=convs),
        )
        return results

    def calc_conv(self, game, solver):
        conv = exploitability.exploitability(
            game,
            policy.tabular_policy_from_callable(game, solver.average_policy()),
        )
        return conv

    def record(self, result):
        algo_df = pd.DataFrame(
            data=dict(
                step=result["data"]["steps"],
                exp=result["data"]["convs"],
            )
        )
        algo_df["game_name"] = result["game_name"]
        algo_df["algorithm_name"] = result["algo_name"]
        algo_df["log_exp"] = np.log(algo_df["exp"])
        self.df = pd.concat([self.df, algo_df])

