import copy
import numpy as np
import pickle
import random
from collections import deque
from pathlib import Path

from autocfr.exp import ex
from autocfr.generator.hash_encoding import hash_encoding
from autocfr.generator.early_hurdle import early_hurdle


class Agent:
    total_index = [0, 0]
    agent_dict_0 = {}
    agent_dict_1 = {}
    dict_name = {0: agent_dict_0, 1: agent_dict_1}
    # agent_dict_2 = {}
    # agent_dict_3 = {}
    # agent_dict_4 = {}

    @classmethod
    def get_agent(cls, agent_index, game_index):
        # print("total_index:", cls.total_index)
        # print("agent_dict_0:", cls.agent_dict_0)
        # print("agent_dict_1:", cls.agent_dict_1)
        if agent_index not in cls.dict_name[game_index]:
            raise Exception("No Agent {} in game {}".format(agent_index, game_index))
        # return cls.agent_dict[agent_index]
        return cls.dict_name[game_index][agent_index]

    def __init__(self, algorithm, game_index, hash_code=None, early_hurdle_score=None):
        self.index = Agent.total_index[game_index]
        # Agent.agent_dict[Agent.total_index] = self
        # Agent.total_index += 1
        Agent.dict_name[game_index][Agent.total_index[game_index]] = self
        Agent.total_index[game_index] += 1
        self.game_index = game_index
        self.algorithm = algorithm
        self.hash_code = hash_code
        self.early_hurdle_score = early_hurdle_score
        self.scores = {}
        # self.weights = {}

    def gen_hash_code(self):
        if self.hash_code is None:
            self.hash_code = hash_encoding.hash_encoding(self.algorithm)

    def gen_early_hurdle_score(self):
        if self.early_hurdle_score is None:
            result = early_hurdle.early_hurdle_score(self.algorithm)
            self.early_hurdle_score = result["score"]

    def copy_score(self, other_agent):
        self.scores = copy.deepcopy(other_agent.scores)
        # self.weights = copy.deepcopy(other_agent.weights)

    def set_score(self, env_name, score):
        self.scores[env_name] = score
        # self.weights[env_name] = weight

    @property
    def ave_score(self):
        if not self.have_evaluated():
            return -1000
        total_score = 0
        total_weight = 0
        for env_name, score in self.scores.items():
            # weight = self.weights[env_name]
            # total_score += score * weight
            total_score += score
            total_weight += 1
        score = total_score / total_weight
        return score

    def have_evaluated(self):
        return len(self.scores) > 0

    @ex.capture
    def save(self, _run, game_name, prefix=None):
        if prefix:
            prefix += "_"
        else:
            prefix = ""
        run_id = _run._id

        file = (
            Path(__file__).parent.parent
            / "logs"
            / str(run_id)
            / "{}algorithms".format(prefix)
            /"game_{}".format(game_name)
            / "algorithms_{}.pkl".format(self.index)
        )
        file.parent.mkdir(parents=True, exist_ok=True)
        with file.open("wb") as f:
            pickle.dump(self.algorithm, f)

        file = (
            Path(__file__).parent.parent
            / "logs"
            / str(run_id)
            / "{}images".format(prefix)
            /"game_{}".format(game_name)
            / "images_{}".format(self.index)
        )
        file.parent.mkdir(parents=True, exist_ok=True)
        self.algorithm.visualize(abs_name=str(file))

    def __repr__(self):
        return "\nindex: {}, score: {}, early_hurdle_score: {}\n {}\n".format(
            self.index, self.ave_score, self.early_hurdle_score, self.algorithm
        )


class AgentCounter:
    def __init__(self):
        self.generating = 0
        self.early_hurdle = 0
        self.func_equiv = 0
        self.evaluaing = 0
        self.succ = 0
        self.fail = 0
        self.check_fail = 0
        self.drop = 0

    def cum_check_fail(self, num):
        self.check_fail += num

    def cum_generating(self, num=1):
        self.generating += num

    def generating_to_func_equv(self, num=1):
        self.generating -= num
        self.func_equiv += num

    def generating_to_early_hurdle(self, num=1):
        self.generating -= num
        self.early_hurdle += num

    def generaing_to_evaluating(self, num=1):
        self.generating -= num
        self.evaluaing += num

    def generaing_to_drop(self, num=1):
        self.generating -= num
        self.drop += num

    def evaluating_to_succ(self, num=1):
        self.evaluaing -= num
        self.succ += num

    def evaluating_to_fail(self, num=1):
        self.evaluaing -= num
        self.fail += num

    def state(self):
        state = {
            "generating": self.generating,
            "early_hurdle": self.early_hurdle,
            "func_equiv": self.func_equiv,
            "evaluaing": self.evaluaing,
            "succ": self.succ,
            "fail": self.fail,
            "check_fail": self.check_fail,
            "drop": self.drop,
        }
        return state

    def info(self):
        state = self.state()
        info = ", ".join([k + ": " + str(v) for k, v in state.items()])
        return info


class Population:
    def __init__(self, population_size, tournament_size):
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.popu = deque(maxlen=population_size)
        self.hash_agents = {}

    def add_agent(self, agent):
        if not agent.have_evaluated():
            raise Exception("you shoule evaluate agent before adding agent")
        agent.gen_hash_code()
        agent.gen_early_hurdle_score()
        self.hash_agents[agent.hash_code] = agent
        self.popu.append(agent)

    def compete(self):
        cur_population_size = len(self.popu)
        sample_num = min(cur_population_size, self.tournament_size)
        agent_indexes = random.sample(list(range(cur_population_size)), sample_num)
        winner = max(
            [self.popu[index] for index in agent_indexes],
            key=lambda agent: agent.ave_score,
        )
        return winner

    def print(self):
        info = ""
        for agent in self.popu:
            info += str(agent.ave_score) + " "
        info += str(self.best_agent.ave_score)
        print(info)

    def check_func_equal_previous_agents(self, agent):
        if agent.hash_code not in self.hash_agents:
            return False
        func_equal_agent = self.get_func_equal_agent(agent)
        if abs(func_equal_agent.early_hurdle_score - agent.early_hurdle_score) > 1e-6:
            print("func equal eroror!!!")
            return False
        return True

    def get_func_equal_agent(self, agent):
        return self.hash_agents[agent.hash_code]

    @property
    def max_score(self):
        return max(agent.ave_score for agent in self.popu)

    @property
    def max_early_hurdle_score(self):
        return max(agent.early_hurdle_score for agent in self.popu)

    @property
    def early_hurdle_threshold(self, percentile=75):
        return min(
            np.percentile(
                [agent.early_hurdle_score for agent in self.popu], percentile
            ),
            0.5,
        )

    @property
    def max_agent_index(self):
        return max(agent.index for agent in self.popu)

    def get_score(self, percentile):
        return np.percentile([agent.ave_score for agent in self.popu], percentile)

    @classmethod
    def save(cls, popu, filename):
        with open(filename, "wb") as f:
            pickle.dump(popu, f)

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as f:
            popu = pickle.load(f)
        return popu

    def __len__(self):
        return len(self.popu)

    def __getitem__(self, index):
        return self.popu[index]

    @classmethod
    def init_from_file(cls, init_population_file):
        population = cls.load(init_population_file)
        Agent.total_index = population.max_agent_index + 1
        return population
