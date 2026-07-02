"""
IA/callbacks.py — Callbacks Keras-like pour le Trainer.

Hiérarchie :
  Callback (base)
    ├── EarlyStopping
    ├── ModelCheckpoint
    ├── ProgressPrinter
    └── CSVLogger

Le Trainer appelle, dans cet ordre, à chaque étape :
  on_train_begin(trainer)
  on_epoch_begin(epoch, trainer)
  on_epoch_end(epoch, metrics, trainer)  # metrics = {'loss':..., 'accuracy':...}
  on_train_end(trainer)

Un callback peut interrompre l'entraînement en mettant trainer.stop_training = True.
"""
import os
import csv
import time
from typing import Any, Dict, List, Optional


class Callback:
    """Classe de base. Surchargez les méthodes voulues."""

    def on_train_begin(self, trainer): pass
    def on_train_end(self, trainer): pass
    def on_epoch_begin(self, epoch: int, trainer): pass
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer): pass


class CallbackList:
    """Container qui dispatch les événements à une liste de callbacks."""

    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []

    def on_train_begin(self, trainer):
        for cb in self.callbacks:
            cb.on_train_begin(trainer)

    def on_train_end(self, trainer):
        for cb in self.callbacks:
            cb.on_train_end(trainer)

    def on_epoch_begin(self, epoch, trainer):
        for cb in self.callbacks:
            cb.on_epoch_begin(epoch, trainer)

    def on_epoch_end(self, epoch, metrics, trainer):
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, metrics, trainer)


class EarlyStopping(Callback):
    """Arrête l'entraînement quand une métrique cesse de s'améliorer.

    Args:
        monitor: métrique à surveiller ('loss' par défaut).
        patience: nombre d'époques sans amélioration avant arrêt.
        min_delta: seuil minimum de variation pour être considéré comme amélioration.
        mode: 'min' (amélioration = baisse) ou 'max' (amélioration = hausse).
    """

    def __init__(self, monitor: str = 'loss', patience: int = 10,
                 min_delta: float = 0.0, mode: str = 'min'):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = float('inf') if mode == 'min' else -float('inf')
        self.wait = 0
        self.stopped_epoch = 0

    def on_train_begin(self, trainer):
        self.best = float('inf') if self.mode == 'min' else -float('inf')
        self.wait = 0

    def on_epoch_end(self, epoch, metrics, trainer):
        current = metrics.get(self.monitor)
        if current is None:
            return
        if self.mode == 'min':
            improved = current < self.best - self.min_delta
        else:
            improved = current > self.best + self.min_delta
        if improved:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                trainer.stop_training = True
                self.stopped_epoch = epoch


class ModelCheckpoint(Callback):
    """Sauvegarde le modèle (au format .gy) à chaque amélioration.

    Args:
        filepath: chemin du fichier .gy.
        monitor: métrique à surveiller.
        save_best_only: si True, ne sauvegarde que si la métrique s'améliore.
        mode: 'min' ou 'max'.
    """

    def __init__(self, filepath: str, monitor: str = 'loss',
                 save_best_only: bool = True, mode: str = 'min'):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.best = float('inf') if mode == 'min' else -float('inf')

    def on_train_begin(self, trainer):
        self.best = float('inf') if self.mode == 'min' else -float('inf')

    def on_epoch_end(self, epoch, metrics, trainer):
        # Nécessite que trainer expose save_model(filepath)
        if not hasattr(trainer, '_current_model_data'):
            return
        current = metrics.get(self.monitor)
        if current is None:
            return
        if self.save_best_only:
            if self.mode == 'min':
                improved = current < self.best
            else:
                improved = current > self.best
            if improved:
                self.best = current
                trainer._save_current(self.filepath)
        else:
            trainer._save_current(self.filepath)


class ProgressPrinter(Callback):
    """Affiche la progression à intervalle régulier.

    Args:
        interval: affiche toutes les N époques.
        show_metrics: liste des métriques à afficher (défaut: ['loss', 'accuracy']).
    """

    def __init__(self, interval: int = 10, show_metrics: Optional[List[str]] = None):
        super().__init__()
        self.interval = interval
        self.show_metrics = show_metrics or ['loss', 'accuracy']
        self.start_time = 0.0

    def on_train_begin(self, trainer):
        self.start_time = time.time()
        print(f"[Trainer] Démarrage entraînement : {trainer.model_type}, "
              f"epochs={trainer.epochs}, lr={trainer.lr}")

    def on_epoch_end(self, epoch, metrics, trainer):
        if epoch % self.interval != 0 and epoch != trainer.epochs - 1:
            return
        elapsed = time.time() - self.start_time
        parts = [f"epoch {epoch:>4d}/{trainer.epochs}"]
        for m in self.show_metrics:
            if m in metrics:
                parts.append(f"{m}={metrics[m]:.4f}")
        parts.append(f"({elapsed:.1f}s)")
        print("[Trainer] " + "  ".join(parts))

    def on_train_end(self, trainer):
        elapsed = time.time() - self.start_time
        print(f"[Trainer] Terminé en {elapsed:.1f}s  "
              f"→  {trainer.model_path}")


class CSVLogger(Callback):
    """Journalise les métriques dans un fichier CSV.

    Args:
        filename: chemin du fichier CSV.
        append: si True, ajoute au fichier existant ; sinon écrase.
    """

    def __init__(self, filename: str, append: bool = False):
        super().__init__()
        self.filename = filename
        self.append = append
        self._file = None
        self._writer = None
        self._fieldnames = None

    def on_train_begin(self, trainer):
        mode = 'a' if self.append else 'w'
        self._file = open(self.filename, mode, newline='', encoding='utf-8')
        self._writer = None
        self._fieldnames = None

    def on_epoch_end(self, epoch, metrics, trainer):
        row = {'epoch': epoch, **metrics}
        if self._writer is None:
            self._fieldnames = list(row.keys())
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
            if not self.append or self._file.tell() == 0:
                self._writer.writeheader()
        # Complète les champs manquants
        for k in self._fieldnames:
            row.setdefault(k, '')
        self._writer.writerow(row)
        self._file.flush()

    def on_train_end(self, trainer):
        if self._file:
            self._file.close()
            self._file = None


# ============================================================================
# Factory utilitaire
# ============================================================================

def default_callbacks(verbose: bool = True, checkpoint_path: Optional[str] = None,
                      early_stopping_patience: int = 0) -> List[Callback]:
    """Construit une liste de callbacks par défaut."""
    cbs: List[Callback] = []
    if verbose:
        cbs.append(ProgressPrinter(interval=10))
    if early_stopping_patience > 0:
        cbs.append(EarlyStopping(patience=early_stopping_patience))
    if checkpoint_path:
        cbs.append(ModelCheckpoint(checkpoint_path, save_best_only=True))
    return cbs
