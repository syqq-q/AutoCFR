import math
import pandas as pd
from pathlib import Path

class Standing:
    def __init__(self, game_configs):
        self.load_baseline_score(game_configs)

    def load_baseline_score(self, game_configs):
        game_names = [config["long_name"] for config in game_configs]
        self.baseline_score = pd.concat([df for df in self.read_games(game_names)])
    
    def read_games(self, game_names):
        for game_name in game_names:
            csv_file = Path(__file__).parent.parent / "models" / "games" / game_name / "CFR_{}.csv".format(game_name)
            df = pd.read_csv(csv_file)
            yield df
            csv_file = Path(__file__).parent.parent / "models" / "games" / game_name / "DCFR_{}.csv".format(game_name)
            df = pd.read_csv(csv_file)
            yield df

    def score(self, exp, game_config):
        iters = game_config["iterations"]
        game_name = game_config["long_name"]
        cfr_exp = self.get_exp("CFR", game_name, iters)
        dcfr_exp = self.get_exp("DCFR", game_name, iters)
        score = (math.log(cfr_exp) - math.log(exp)) / (
            math.log(cfr_exp) - math.log(dcfr_exp)
        )
        score = min(score, game_config["max_score"])
        return score

    def get_exp(self, algorithm_name, game_name, iters):
        df = self.baseline_score
        exp = df[
            (df.algorithm_name == algorithm_name)
            & (df.game_name == game_name)
            & (df.step == iters)
        ].iloc[0]["exp"]
        return exp

