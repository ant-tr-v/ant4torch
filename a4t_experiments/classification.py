import ignite.engine
import torch.nn as nn
import torch.optim as optim

from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import Accuracy, Loss

from .abstract import Experiment


class Classification(Experiment):
    def __init__(self):
        super().__init__()
        self._criterion = nn.CrossEntropyLoss()
        self._val_metrics = {}
        self._optimizer = None

    def _update_metrics(self):
        self._val_metrics = {
            "accuracy": Accuracy(),
            "loss": Loss(self._criterion)
        }

    def _update_optimizer(self, model: nn.Module):
        # TODO: choose optimizer and its settings according to model setup
        kw_args = {
            'lr': 1e-3,
            'weight_decay': 1e-5
        }
        self._optimizer = optim.AdamW(model.parameters(), **kw_args)

    def _update_handlers(self, model, trainer_engine: ignite.engine.Engine, evaluator_engine: ignite.engine.Engine):
        # TODO: choose handlers according to configuration
        @trainer_engine.on(Events.EPOCH_COMPLETED)
        def evaluate_model(trainer):
            eval_state = evaluator_engine.run(self.test_loader)
            log = f'Epoch #{trainer_engine.state.epoch}:\n\t' + ' '.join(f'{k}: {v}' for k, v in eval_state.metrics.items())
            print(log)

    # overrides of Experiment

    def _configure(self):
        if self.train_loader is None or self.test_loader is None:
            raise RuntimeError('Experiment is not ready because data loaders not provided')
        self._update_metrics()

    def _evaluate_a_network(self, model):
        evaluator_engine = create_supervised_evaluator(model, metrics=self._val_metrics, device=self.device)
        state = evaluator_engine.run()
        return state.metrics

    def _train_a_network(self, model):
        # updating everything model-dependent: optimizer, scheduler etc...
        self._update_optimizer(model)
        # setting up ignite engines
        evaluator_engine = create_supervised_evaluator(model, metrics=self._val_metrics, device=self.device)
        trainer_engine = create_supervised_trainer(model, self._optimizer, self._criterion, device=self.device)
        # applying all the required handlers to trainer_engine
        self._update_handlers(model, trainer_engine, evaluator_engine)
        trainer_engine.run(self.train_loader, max_epochs=self.n_epochs)