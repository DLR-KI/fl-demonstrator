from .base import ModelTrainerEvent


class ModelTestFinished(ModelTrainerEvent):

    def next(self):
        if self.training.model.round < self.training.target_num_updates:
            self.trainer.start_round()
        else:
            self.trainer.finish()

    def handle(self):
        # currently do nothing
        # Potentially, one could aggregate all common metrics here.
        pass
