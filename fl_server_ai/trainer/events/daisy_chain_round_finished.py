from .training_round_finished import TrainingRoundFinished
from .. import model_trainer


class DaisyChainRoundFinished(TrainingRoundFinished):

    def __init__(self, trainer: "model_trainer.ModelTrainer"):
        super().__init__(trainer)
        self.trainer.options.model_test_after_each_round = False

    def handle(self):
        # the round increment is not done yet, therefore `model.round + 1`
        if (self.training.model.round + 1) >= self.training.target_num_updates:
            # local client training is finished, let's do the aggregation
            super().handle()  # also does the round increment
            return

        # local client training is not finished yet, but we reach a daisy chaining period
        # => no aggregation, just send the permutated client models back for further training
        # also see `FedDCModelTrainer.handle()`

        self.training.model.round += 1
        self.training.model.save()
