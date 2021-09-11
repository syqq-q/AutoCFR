import numpy as np
import pickle
from pathlib import Path


class HashEncoding:
    def __init__(self):
        self.init_random_sample()

    def load_samples(self, filename):
        path = Path(__file__).parent.parent.parent / "models" / filename
        with path.open("rb") as f:
            random_samples = pickle.load(f)
        return random_samples

    def init_random_sample(self):
        self.random_samples = []
        filenames = [
            "random_inputs.pkl",
            "algo_inputs_a.pkl",
        ]
        for filename in filenames:
            self.random_samples.extend(self.load_samples(filename))

    def hash_encoding(self, algorithm):
        """根据10个随机产生的输入，将程序的输出连接起来，得到程序的hash编码。在调用hash_encoding前应先调用program_check，保证程序执行不出错

        Args:
            algorithm: 待编码的输入程序

        Returns:
            hash_code (str): 程序的hash编码
        """
        hash_code = ""
        for input_values_of_names in self.random_samples:
            output_values_of_names = algorithm.execute(input_values_of_names)
            changed_fields = ["cumu_regret", "strategy", "cumu_strategy"]
            for k, v in output_values_of_names.items():
                if k in changed_fields:
                    hash_code += "".join([str(round(x, 5))[:8] for x in v])
        return hash_code


hash_encoding = HashEncoding()
