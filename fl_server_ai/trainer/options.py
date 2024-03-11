from dataclasses import dataclass


@dataclass
class TrainerOptions:
    skip_model_tests: bool = False
    model_test_after_each_round: bool = True
    delete_local_models_after_aggregation: bool = True
