import pickle
import pickletools
import numpy as np
from autocfr.utils import load_game
import attr
from autocfr.cfr.cfr_algorithm import load_algorithm
# file_path1 = "models/algorithms/cfr.pkl"
# file_path2 = "models/algorithms/cfr_plus.pkl"
# with open(file_path1, "rb") as f:
#     popu1 = pickle.load(f)
#     print(popu1)
# with open(file_path2, "rb") as f:
#     popu2 = pickle.load(f)
#     print(popu2)

# class State:
#     @classmethod
#     def from_history(cls):
#             feature = "h.information_state_string()"
#             reach = 0
#             na = 2
#             ins_regret = np.zeros([na])
#             cumu_regret = np.zeros([na])
#             strategy = np.ones([na]) / na
#             cumu_strategy = np.zeros([na])
#             average_strategy = np.ones([na]) / na
#             return cls(
#                 feature,
#                 reach,
#                 ins_regret,
#                 cumu_regret,
#                 strategy,
#                 cumu_strategy,
#                 average_strategy
#             )

# print(State.from_history())
# @attr.s
# class State:
#     player = attr.ib()
#     feature = attr.ib()
#     na = attr.ib()
#     reach = attr.ib()
#     ins_regret = attr.ib()
#     cumu_regret = attr.ib()
#     strategy = attr.ib()
#     cumu_strategy = attr.ib()
#     average_strategy = attr.ib()
#     @classmethod
#     def from_history(cls, h):
#         player = h.current_player()
#         feature = h.information_state_string()
#         na = len(h.legal_actions())
#         reach = 0
#         ins_regret = np.zeros([na])
#         cumu_regret = np.zeros([na])
#         strategy = np.ones([na]) / na
#         cumu_strategy = np.zeros([na])
#         average_strategy = np.ones([na]) / na
#         return cls(
#             player,
#             feature,
#             na,
#             reach,
#             ins_regret,
#             cumu_regret,
#             strategy,
#             cumu_strategy,
#             average_strategy,
#         )
# game_config = {
#             "long_name": "kuhn_poker",
#             "game_name": "kuhn_poker",
#             "params": {"players": 2},
#         }

# game = load_game(game_config)
# states = {}
# def init_states(h):
#     if h.is_terminal():
#         return
#     if h.is_chance_node():
#         for a, p in h.chance_outcomes():
#             init_states(h.child(a))
#         return
#     feature = h.information_state_string()
#     print("information_state_string:", feature)
#     if states.get(feature) is None:
#         states[feature] = State.from_history(h) #states是一个字典，键为当前的信息集字符串，值是一个对象实例
#     for a in h.legal_actions():
#         init_states(h.child(a))
# h = game.new_initial_state()
# init_states(h)
# print("state:", states)
# # print(h.information_state_string()) # player = -1, 0 = 0(string)

# vec = np.zeros((2, 3), dtype=np.float32)
# size = vec.shape
# N = size[1]
# print("size:", size)
# p_sum = np.expand_dims(np.sum(vec, axis=1), axis=1).repeat(N, axis=1)
# res = np.where(
#     p_sum > 0.0,
#     vec / p_sum,
#     np.full(shape=size, fill_value=1.0/N, dtype=np.float32)
#     )
# print("\nres:", res)

# import importlib
# name = "autocfr.vanilla_cfr:DCFRPlusSolver"
# if ":" in name:
#     mod_name, attr_name = name.split(":")
# else:
#     li = name.split(".")
#     mod_name, attr_name = ".".join(li[:-1]), li[-1]
# mod = importlib.import_module(mod_name)
# fn = getattr(mod, attr_name)
# print(fn)

file_path = "models/algorithms/dcfr_plus.pkl"
algorithm = load_algorithm("dcfr_plus")
with open(file_path, 'wb') as f:
    pickle.dump(algorithm, f)

