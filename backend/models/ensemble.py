"""
BT3068 — Ensemble Classification Framework
=============================================
Combines multiple CNN models for robust multi-disease detection:
  - MobileNetV2 (3.4M params — lightweight)
  - DenseNet121 (8M params — dense connectivity)
  - ResNet101 (44.5M params — deep residual)

Ensemble Methods:
  - Soft Voting (probability-weighted average)
  - Stacking Meta-Learner (learned optimal combination)
  - MRLA Optimization (Mutation Rate-Based Lion Algorithm)

Reference: Cohen's κ = 0.809, 3× performance gain over single CNN classifiers
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


# ══════════════════════════════════════════
# Individual Model Builders
# ══════════════════════════════════════════

def build_mobilenetv2(num_classes, input_shape=(380, 380, 3), freeze_base=True):
    """Build MobileNetV2 classifier (3.4M params — lightweight)."""
    base = keras.applications.MobileNetV2(
        weights='imagenet', include_top=False, input_shape=input_shape
    )
    base.trainable = not freeze_base

    inputs = keras.Input(shape=input_shape)
    x = base(inputs, training=not freeze_base)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs, name='MobileNetV2_Classifier')


def build_densenet121(num_classes, input_shape=(380, 380, 3), freeze_base=True):
    """Build DenseNet121 classifier (8M params — dense connectivity)."""
    base = keras.applications.DenseNet121(
        weights='imagenet', include_top=False, input_shape=input_shape
    )
    base.trainable = not freeze_base

    inputs = keras.Input(shape=input_shape)
    x = base(inputs, training=not freeze_base)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs, name='DenseNet121_Classifier')


def build_resnet101(num_classes, input_shape=(380, 380, 3), freeze_base=True):
    """Build ResNet101 classifier (44.5M params — deep residual)."""
    base = keras.applications.ResNet101(
        weights='imagenet', include_top=False, input_shape=input_shape
    )
    base.trainable = not freeze_base

    inputs = keras.Input(shape=input_shape)
    x = base(inputs, training=not freeze_base)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs, name='ResNet101_Classifier')


# ══════════════════════════════════════════
# Ensemble Methods
# ══════════════════════════════════════════

class SoftVotingEnsemble:
    """
    Soft Voting Ensemble — probability-weighted average.

    Each model contributes its class probability vector,
    and the final prediction is the average (or weighted average).
    """

    def __init__(self, models, weights=None):
        """
        Args:
            models: List of trained Keras models
            weights: Optional list of model weights (default: equal weights)
        """
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)

    def predict(self, images):
        """
        Predict using soft voting ensemble.

        Args:
            images: Input image batch

        Returns:
            Averaged probability vectors
        """
        predictions = []
        for model, weight in zip(self.models, self.weights):
            pred = model.predict(images, verbose=0)
            predictions.append(pred * weight)

        # Weighted average
        ensemble_pred = np.sum(predictions, axis=0)
        ensemble_pred = ensemble_pred / np.sum(self.weights)

        return ensemble_pred

    def predict_with_individual(self, images):
        """Predict and return both individual and ensemble results."""
        individual_preds = {}
        for model in self.models:
            pred = model.predict(images, verbose=0)
            individual_preds[model.name] = pred

        ensemble_pred = self.predict(images)
        return ensemble_pred, individual_preds


class StackingMetaLearner:
    """
    Stacking Meta-Learner — learns optimal linear combination of base model outputs.

    Uses Logistic Regression as the meta-learner to combine predictions
    from MobileNetV2, DenseNet121, and ResNet101.
    """

    def __init__(self, models):
        """
        Args:
            models: List of trained Keras models (base learners)
        """
        self.models = models
        self.meta_learner = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            max_iter=1000,
            C=1.0
        )
        self.is_fitted = False

    def generate_meta_features(self, images):
        """
        Generate meta-features by concatenating predictions from all base models.

        Args:
            images: Input image batch

        Returns:
            Meta-feature matrix (n_samples, n_models × n_classes)
        """
        meta_features = []
        for model in self.models:
            pred = model.predict(images, verbose=0)
            meta_features.append(pred)
        return np.hstack(meta_features)

    def fit(self, train_images, train_labels):
        """
        Train the meta-learner on base model predictions.

        Args:
            train_images: Training image batch
            train_labels: True labels (integer encoded)
        """
        meta_features = self.generate_meta_features(train_images)
        self.meta_learner.fit(meta_features, train_labels)
        self.is_fitted = True

        # Cross-validation score
        scores = cross_val_score(self.meta_learner, meta_features, train_labels, cv=3)
        print(f"[STACKING] Meta-learner CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

    def predict(self, images):
        """Predict using the stacking ensemble."""
        if not self.is_fitted:
            raise ValueError("Meta-learner not fitted. Call fit() first.")

        meta_features = self.generate_meta_features(images)
        return self.meta_learner.predict_proba(meta_features)


class MRLAOptimizer:
    """
    Mutation Rate-Based Lion Algorithm (MRLA) for ensemble hyperparameter optimization.

    Optimizes model weights in the ensemble by simulating lion pride behavior
    with mutation-based exploration.

    Reference: Paper P11 — MRLA Ensemble optimization
    """

    def __init__(self, n_models, population_size=20, max_iterations=50,
                 mutation_rate=0.1):
        self.n_models = n_models
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.mutation_rate = mutation_rate

    def _initialize_population(self):
        """Initialize random weight vectors (each sums to 1)."""
        population = np.random.dirichlet(
            np.ones(self.n_models), self.population_size
        )
        return population

    def _fitness(self, weights, predictions, true_labels):
        """
        Compute fitness (accuracy) for a weight configuration.

        Args:
            weights: Model weight vector
            predictions: List of prediction arrays (one per model)
            true_labels: True class labels

        Returns:
            Accuracy score
        """
        # Weighted ensemble prediction
        ensemble = np.zeros_like(predictions[0])
        for w, pred in zip(weights, predictions):
            ensemble += w * pred

        predicted_classes = np.argmax(ensemble, axis=1)
        accuracy = np.mean(predicted_classes == true_labels)
        return accuracy

    def _mutate(self, individual):
        """Apply mutation to a weight vector."""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if np.random.random() < self.mutation_rate:
                mutated[i] += np.random.normal(0, 0.1)
        # Ensure non-negative and normalize
        mutated = np.clip(mutated, 0.01, None)
        mutated = mutated / np.sum(mutated)
        return mutated

    def optimize(self, predictions, true_labels):
        """
        Run MRLA optimization to find optimal ensemble weights.

        Args:
            predictions: List of prediction arrays from each model
            true_labels: True class labels

        Returns:
            Best weight vector, best fitness score
        """
        population = self._initialize_population()

        best_weights = None
        best_fitness = 0

        for iteration in range(self.max_iterations):
            # Evaluate fitness
            fitness_scores = np.array([
                self._fitness(ind, predictions, true_labels)
                for ind in population
            ])

            # Track best
            idx_best = np.argmax(fitness_scores)
            if fitness_scores[idx_best] > best_fitness:
                best_fitness = fitness_scores[idx_best]
                best_weights = population[idx_best].copy()

            # Selection (tournament)
            new_population = [best_weights.copy()]  # Elitism
            for _ in range(self.population_size - 1):
                i, j = np.random.choice(self.population_size, 2, replace=False)
                winner = population[i] if fitness_scores[i] > fitness_scores[j] else population[j]
                new_population.append(self._mutate(winner))

            population = np.array(new_population)

            # Adaptive mutation rate
            self.mutation_rate *= 0.99  # Decay mutation rate

            if (iteration + 1) % 10 == 0:
                print(f"  [MRLA] Iteration {iteration + 1}/{self.max_iterations} "
                      f"— Best fitness: {best_fitness:.4f}")

        print(f"\n[MRLA] Optimization complete.")
        print(f"[MRLA] Best weights: {best_weights}")
        print(f"[MRLA] Best accuracy: {best_fitness:.4f}")

        return best_weights, best_fitness


def build_ensemble(num_classes, input_shape=(380, 380, 3)):
    """
    Build complete ensemble system with all three models.

    Returns:
        models: List of [MobileNetV2, DenseNet121, ResNet101]
        soft_voter: SoftVotingEnsemble instance
        stacker: StackingMetaLearner instance
        mrla: MRLAOptimizer instance
    """
    print("[ENSEMBLE] Building multi-model ensemble...")

    models = [
        build_mobilenetv2(num_classes, input_shape),
        build_densenet121(num_classes, input_shape),
        build_resnet101(num_classes, input_shape),
    ]

    for m in models:
        print(f"  ✓ {m.name}: {m.count_params():,} parameters")

    soft_voter = SoftVotingEnsemble(models)
    stacker = StackingMetaLearner(models)
    mrla = MRLAOptimizer(n_models=len(models))

    print(f"[ENSEMBLE] {len(models)} models ready for ensemble classification")

    return models, soft_voter, stacker, mrla


if __name__ == "__main__":
    models, soft_voter, stacker, mrla = build_ensemble(num_classes=2)
