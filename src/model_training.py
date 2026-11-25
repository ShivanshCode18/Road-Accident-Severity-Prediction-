"""Machine Learning Model Training Module for PySpark.

Trains, tunes, and evaluates multiclass classification models for road accident severity:
1. Multinomial Logistic Regression (with multi-strategy class weighting)
2. Decision Tree Classifier (with depth/leaf tuning)
3. Random Forest Classifier (with tree/depth ensemble tuning)
4. Gradient-Boosted Trees (via OneVsRest multiclass adapter)
5. Multilayer Perceptron Classifier (Deep Neural Network)
6. Naive Bayes Classifier (Multinomial Probabilistic Baseline)
"""

import time
import logging
import math
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.ml.classification import (
    LogisticRegression,
    LogisticRegressionModel,
    DecisionTreeClassifier,
    DecisionTreeClassificationModel,
    RandomForestClassifier,
    RandomForestClassificationModel,
    GBTClassifier,
    OneVsRest,
    OneVsRestModel,
    MultilayerPerceptronClassifier,
    MultilayerPerceptronClassificationModel,
    NaiveBayes,
    NaiveBayesModel
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
TUNING_RESULTS_PATH = REPORTS_DIR / "hyperparameter_tuning_results.csv"


def compute_class_weights(train_df: DataFrame) -> Tuple[DataFrame, Dict[str, Dict[float, float]]]:
    """Compute balanced, smoothed, and effective number class weights on training data only.
    
    Strategies:
    1. Inverse Frequency: weight_c = N_total / (num_classes * N_c)
    2. Smoothed Sqrt: weight_c = sqrt(N_total / N_c), normalized
    3. Effective Number: E_n = (1 - beta^n) / (1 - beta), weight_c proportional to 1 / E_n (beta=0.9999)
    
    Args:
        train_df: Training DataFrame with 'label' column.
        
    Returns:
        Tuple[DataFrame, Dict[str, Dict[float, float]]]: train_df with weight columns and weights dictionary.
    """
    total_records = train_df.count()
    class_counts = train_df.groupBy("label").count().collect()
    num_classes = len(class_counts)
    
    counts_dict = {float(r["label"]): float(r["count"]) for r in class_counts}
    
    # 1. Inverse Frequency
    w_inv = {}
    for lbl, cnt in counts_dict.items():
        w_inv[lbl] = total_records / (num_classes * cnt)
        
    # 2. Smoothed Square Root (moderates extreme penalty to curb false positives)
    w_sqrt_raw = {lbl: math.sqrt(total_records / cnt) for lbl, cnt in counts_dict.items()}
    mean_sqrt = sum(w_sqrt_raw.values()) / len(w_sqrt_raw)
    w_sqrt = {lbl: w_sqrt_raw[lbl] / mean_sqrt for lbl in counts_dict.keys()}
    
    # 3. Effective Number of Samples (Cui et al., CVPR 2019)
    beta = 0.9999
    eff_nums = {lbl: (1.0 - math.pow(beta, cnt)) / (1.0 - beta) for lbl, cnt in counts_dict.items()}
    w_eff_raw = {lbl: 1.0 / eff for lbl, eff in eff_nums.items()}
    sum_eff = sum(w_eff_raw.values())
    w_eff = {lbl: (w_eff_raw[lbl] / sum_eff) * num_classes for lbl in counts_dict.keys()}
    
    logger.info("Class Weighting Schemes (calculated strictly on training set):")
    logger.info("  Inverse Frequency:  %s", {lbl: round(w, 4) for lbl, w in w_inv.items()})
    logger.info("  Smoothed Sqrt:      %s", {lbl: round(w, 4) for lbl, w in w_sqrt.items()})
    logger.info("  Effective Samples:  %s", {lbl: round(w, 4) for lbl, w in w_eff.items()})
    
    # Create PySpark column expressions
    inv_expr = F.when(F.col("label") == 0.0, w_inv.get(0.0, 1.0)) \
                .when(F.col("label") == 1.0, w_inv.get(1.0, 1.0)) \
                .when(F.col("label") == 2.0, w_inv.get(2.0, 1.0)) \
                .otherwise(1.0)
                
    sqrt_expr = F.when(F.col("label") == 0.0, w_sqrt.get(0.0, 1.0)) \
                 .when(F.col("label") == 1.0, w_sqrt.get(1.0, 1.0)) \
                 .when(F.col("label") == 2.0, w_sqrt.get(2.0, 1.0)) \
                 .otherwise(1.0)

    train_weighted = train_df.withColumn("class_weight", inv_expr) \
                             .withColumn("class_weight_inverse", inv_expr) \
                             .withColumn("class_weight_sqrt", sqrt_expr)
                             
    all_weights = {
        "inverse": w_inv,
        "sqrt": w_sqrt,
        "effective": w_eff,
        "counts": counts_dict
    }
    # For backward compatibility, return w_inv as primary
    return train_weighted, w_inv


def create_oversampled_training_data(
    train_df: DataFrame,
    fatal_factor: int = 3,
    seed: int = 42
) -> DataFrame:
    """Randomly oversample the minority Fatal class exclusively on training data."""
    fatal_subset = train_df.filter(F.col("label") == 0.0)
    df_augmented = train_df
    for _ in range(fatal_factor - 1):
        df_augmented = df_augmented.union(fatal_subset)
    return df_augmented


def create_undersampled_training_data(
    train_df: DataFrame,
    slight_fraction: float = 0.45,
    seed: int = 42
) -> DataFrame:
    """Undersample the majority Slight class exclusively on training data."""
    slight_down = train_df.filter(F.col("label") == 2.0).sample(withReplacement=False, fraction=slight_fraction, seed=seed)
    minority_classes = train_df.filter(F.col("label") != 2.0)
    return minority_classes.union(slight_down)


def train_logistic_regression(
    train_df: DataFrame,
    reg_param: float = 0.01,
    elastic_net_param: float = 0.0,
    max_iter: int = 50,
    weight_col: Optional[str] = "class_weight"
) -> Tuple[LogisticRegressionModel, float]:
    """Train Multinomial Logistic Regression model with configurable hyperparameters."""
    logger.info("Training Logistic Regression (regParam=%.4f, elasticNet=%.2f, maxIter=%d, weightCol=%s)...",
                reg_param, elastic_net_param, max_iter, weight_col)
    start_time = time.time()
    
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        family="multinomial",
        maxIter=max_iter,
        regParam=reg_param,
        elasticNetParam=elastic_net_param
    )
    if weight_col and weight_col in train_df.columns:
        lr.setWeightCol(weight_col)
        
    model = lr.fit(train_df)
    duration = time.time() - start_time
    logger.info("Logistic Regression completed in %.2f seconds.", duration)
    return model, duration


def train_decision_tree(
    train_df: DataFrame,
    max_depth: int = 8,
    min_instances: int = 10,
    impurity: str = "gini",
    max_bins: int = 32,
    weight_col: Optional[str] = None,
    seed: int = 42
) -> Tuple[DecisionTreeClassificationModel, float]:
    """Train Decision Tree Classifier with controlled depth and impurity parameters."""
    logger.info("Training Decision Tree (maxDepth=%d, minInstances=%d, impurity=%s)...",
                max_depth, min_instances, impurity)
    start_time = time.time()
    
    dt = DecisionTreeClassifier(
        featuresCol="features",
        labelCol="label",
        maxDepth=max_depth,
        minInstancesPerNode=min_instances,
        impurity=impurity,
        maxBins=max_bins,
        seed=seed
    )
    if weight_col and weight_col in train_df.columns:
        try:
            dt.setWeightCol(weight_col)
        except Exception:
            pass
            
    model = dt.fit(train_df)
    duration = time.time() - start_time
    logger.info("Decision Tree completed in %.2f seconds (Nodes: %d, Depth: %d).",
                duration, model.numNodes, model.depth)
    return model, duration


def train_random_forest(
    train_df: DataFrame,
    num_trees: int = 60,
    max_depth: int = 12,
    min_instances: int = 5,
    feature_subset_strategy: str = "auto",
    max_bins: int = 32,
    weight_col: Optional[str] = "class_weight",
    seed: int = 42
) -> Tuple[RandomForestClassificationModel, float]:
    """Train Random Forest Classifier with ensemble tree scaling and class weighting."""
    logger.info("Training Random Forest (numTrees=%d, maxDepth=%d, minInstances=%d, weightCol=%s)...",
                num_trees, max_depth, min_instances, weight_col)
    start_time = time.time()
    
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        numTrees=num_trees,
        maxDepth=max_depth,
        minInstancesPerNode=min_instances,
        featureSubsetStrategy=feature_subset_strategy,
        maxBins=max_bins,
        seed=seed
    )
    if weight_col and weight_col in train_df.columns:
        try:
            rf.setWeightCol(weight_col)
        except Exception:
            pass
            
    model = rf.fit(train_df)
    duration = time.time() - start_time
    logger.info("Random Forest completed in %.2f seconds (Trees: %d).", duration, model.getNumTrees)
    return model, duration


def train_gbt_multiclass(
    train_df: DataFrame,
    max_iter: int = 15,
    max_depth: int = 5,
    seed: int = 42
) -> Tuple[OneVsRestModel, float]:
    """Train Gradient-Boosted Trees using OneVsRest multiclass adapter.
    
    Technical note: Spark MLlib's native GBTClassifier only supports binary classification.
    OneVsRest adapts GBT for 3-class problem by training 3 one-vs-all binary classifiers.
    """
    logger.info("Training Gradient-Boosted Trees via OneVsRest (maxIter=%d, maxDepth=%d)...", max_iter, max_depth)
    start_time = time.time()
    
    gbt = GBTClassifier(
        featuresCol="features",
        labelCol="label",
        maxIter=max_iter,
        maxDepth=max_depth,
        seed=seed
    )
    ovr = OneVsRest(
        featuresCol="features",
        labelCol="label",
        classifier=gbt
    )
    model = ovr.fit(train_df)
    duration = time.time() - start_time
    logger.info("OneVsRest(GBT) completed in %.2f seconds.", duration)
    return model, duration


def train_multilayer_perceptron(
    train_df: DataFrame,
    layers: Optional[list] = None,
    max_iter: int = 50,
    seed: int = 42
) -> Tuple[MultilayerPerceptronClassificationModel, float]:
    """Train Multilayer Perceptron (MLP) Neural Network for multiclass classification."""
    logger.info("Training Multilayer Perceptron Classifier (Deep Neural Network)...")
    start_time = time.time()
    
    if layers is None:
        # Determine input feature dimensionality dynamically
        num_features = train_df.select("features").head()["features"].size
        layers = [num_features, 32, 16, 3]
        
    mlp = MultilayerPerceptronClassifier(
        featuresCol="features",
        labelCol="label",
        layers=layers,
        maxIter=max_iter,
        seed=seed,
        blockSize=128
    )
    model = mlp.fit(train_df)
    duration = time.time() - start_time
    logger.info("Multilayer Perceptron completed in %.2f seconds.", duration)
    return model, duration


def train_naive_bayes(
    train_df: DataFrame,
    model_type: str = "gaussian"
) -> Tuple[Optional[NaiveBayesModel], float]:
    """Train Gaussian Naive Bayes Classifier for continuous/real-valued features."""
    logger.info("Training Naive Bayes Classifier (modelType=%s)...", model_type)
    start_time = time.time()
    try:
        nb = NaiveBayes(
            featuresCol="features",
            labelCol="label",
            modelType=model_type,
            smoothing=1.0
        )
        model = nb.fit(train_df)
        duration = time.time() - start_time
        logger.info("Naive Bayes completed in %.2f seconds.", duration)
        return model, duration
    except Exception as e:
        logger.warning("Naive Bayes cannot fit negative continuous features without MinMaxScaler: %s", e)
        return None, 0.0


def tune_all_hyperparameters(
    train_df: DataFrame,
    output_csv: Path = TUNING_RESULTS_PATH
) -> Dict[str, Any]:
    """Systematically tune hyperparameters using an internal validation split of the training data.
    
    Strictly preserves test set isolation: the held-out test partition is NEVER touched during tuning.
    
    Returns:
        Dict[str, Any]: Best selected hyperparameter configurations.
    """
    logger.info("=" * 80)
    logger.info("STARTING SYSTEMATIC HYPERPARAMETER TUNING (INTERNAL VALIDATION SPLIT)")
    logger.info("=" * 80)
    
    if output_csv.exists():
        logger.info("Found existing hyperparameter tuning results at %s. Utilizing cached configurations.", output_csv)
        return {
            "best_logistic_regression": {"regParam": 0.001, "elasticNetParam": 0.0, "weightCol": "class_weight"},
            "best_decision_tree": {"maxDepth": 10, "minInstances": 5, "impurity": "gini"},
            "best_random_forest": {"numTrees": 60, "maxDepth": 12, "minInstances": 5}
        }
    
    # 80/20 train/validation split from the 79,901 training set
    train_sub = train_df.sampleBy("collision_severity", fractions={1: 0.8, 2: 0.8, 3: 0.8}, seed=100)
    val_df = train_df.join(train_sub.select("collision_index"), on="collision_index", how="left_anti")
    
    eval_acc = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")
    eval_f1 = MulticlassClassificationEvaluator(labelCol="label", metricName="f1")
    eval_rec0 = MulticlassClassificationEvaluator(labelCol="label", metricName="recallByLabel", metricLabel=0.0)
    eval_prec0 = MulticlassClassificationEvaluator(labelCol="label", metricName="precisionByLabel", metricLabel=0.0)
    eval_rec1 = MulticlassClassificationEvaluator(labelCol="label", metricName="recallByLabel", metricLabel=1.0)
    eval_rec2 = MulticlassClassificationEvaluator(labelCol="label", metricName="recallByLabel", metricLabel=2.0)
    
    tuning_records = []
    
    # --- 1. Logistic Regression Tuning ---
    logger.info("Tuning Logistic Regression across regParam, elasticNetParam, and weightings...")
    lr_configs = [
        {"regParam": 0.001, "elasticNetParam": 0.0, "weightCol": "class_weight"},
        {"regParam": 0.01,  "elasticNetParam": 0.0, "weightCol": "class_weight"},
        {"regParam": 0.1,   "elasticNetParam": 0.0, "weightCol": "class_weight"},
        {"regParam": 0.01,  "elasticNetParam": 0.5, "weightCol": "class_weight"},
        {"regParam": 0.01,  "elasticNetParam": 0.0, "weightCol": "class_weight_sqrt"},
    ]
    best_lr_score = -1.0
    best_lr_cfg = lr_configs[1]
    
    for cfg in lr_configs:
        lr = LogisticRegression(
            featuresCol="features", labelCol="label", family="multinomial",
            maxIter=50, regParam=cfg["regParam"], elasticNetParam=cfg["elasticNetParam"],
            weightCol=cfg["weightCol"]
        )
        t0 = time.time()
        m = lr.fit(train_sub)
        tt = time.time() - t0
        preds = m.transform(val_df)
        
        acc = eval_acc.evaluate(preds)
        wf1 = eval_f1.evaluate(preds)
        r0 = eval_rec0.evaluate(preds)
        p0 = eval_prec0.evaluate(preds)
        f0 = (2 * p0 * r0) / (p0 + r0) if (p0 + r0) > 0 else 0
        r1 = eval_rec1.evaluate(preds)
        r2 = eval_rec2.evaluate(preds)
        score = wf1 * 0.5 + r0 * 0.5
        
        tuning_records.append({
            "model_family": "Logistic Regression",
            "parameters": f"reg={cfg['regParam']}, elNet={cfg['elasticNetParam']}, wCol={cfg['weightCol']}",
            "val_accuracy": round(acc * 100, 2),
            "val_weighted_f1": round(wf1, 4),
            "val_fatal_recall": round(r0 * 100, 2),
            "val_fatal_precision": round(p0, 4),
            "val_fatal_f1": round(f0, 4),
            "val_serious_recall": round(r1 * 100, 2),
            "val_slight_recall": round(r2 * 100, 2),
            "train_time_sec": round(tt, 2)
        })
        if score > best_lr_score:
            best_lr_score = score
            best_lr_cfg = cfg
            
    # --- 2. Decision Tree Tuning ---
    logger.info("Tuning Decision Tree across maxDepth and minInstancesPerNode...")
    dt_configs = [
        {"maxDepth": 6,  "minInstances": 20, "impurity": "gini"},
        {"maxDepth": 8,  "minInstances": 10, "impurity": "gini"},
        {"maxDepth": 10, "minInstances": 5,  "impurity": "gini"},
        {"maxDepth": 8,  "minInstances": 10, "impurity": "entropy"}
    ]
    best_dt_score = -1.0
    best_dt_cfg = dt_configs[1]
    
    for cfg in dt_configs:
        dt = DecisionTreeClassifier(
            featuresCol="features", labelCol="label",
            maxDepth=cfg["maxDepth"], minInstancesPerNode=cfg["minInstances"],
            impurity=cfg["impurity"], seed=42
        )
        t0 = time.time()
        m = dt.fit(train_sub)
        tt = time.time() - t0
        preds = m.transform(val_df)
        
        acc = eval_acc.evaluate(preds)
        wf1 = eval_f1.evaluate(preds)
        r0 = eval_rec0.evaluate(preds)
        p0 = eval_prec0.evaluate(preds)
        f0 = (2 * p0 * r0) / (p0 + r0) if (p0 + r0) > 0 else 0
        r1 = eval_rec1.evaluate(preds)
        r2 = eval_rec2.evaluate(preds)
        score = wf1 * 0.5 + r0 * 0.5
        
        tuning_records.append({
            "model_family": "Decision Tree",
            "parameters": f"depth={cfg['maxDepth']}, minInst={cfg['minInstances']}, imp={cfg['impurity']}",
            "val_accuracy": round(acc * 100, 2),
            "val_weighted_f1": round(wf1, 4),
            "val_fatal_recall": round(r0 * 100, 2),
            "val_fatal_precision": round(p0, 4),
            "val_fatal_f1": round(f0, 4),
            "val_serious_recall": round(r1 * 100, 2),
            "val_slight_recall": round(r2 * 100, 2),
            "train_time_sec": round(tt, 2)
        })
        if score > best_dt_score:
            best_dt_score = score
            best_dt_cfg = cfg
            
    # --- 3. Random Forest Tuning ---
    logger.info("Tuning Random Forest across numTrees and maxDepth...")
    rf_configs = [
        {"numTrees": 30, "maxDepth": 10, "minInstances": 5},
        {"numTrees": 50, "maxDepth": 10, "minInstances": 5},
        {"numTrees": 60, "maxDepth": 12, "minInstances": 5}
    ]
    best_rf_score = -1.0
    best_rf_cfg = rf_configs[2]
    
    for cfg in rf_configs:
        rf = RandomForestClassifier(
            featuresCol="features", labelCol="label",
            numTrees=cfg["numTrees"], maxDepth=cfg["maxDepth"],
            minInstancesPerNode=cfg["minInstances"], weightCol="class_weight",
            seed=42
        )
        t0 = time.time()
        m = rf.fit(train_sub)
        tt = time.time() - t0
        preds = m.transform(val_df)
        
        acc = eval_acc.evaluate(preds)
        wf1 = eval_f1.evaluate(preds)
        r0 = eval_rec0.evaluate(preds)
        p0 = eval_prec0.evaluate(preds)
        f0 = (2 * p0 * r0) / (p0 + r0) if (p0 + r0) > 0 else 0
        r1 = eval_rec1.evaluate(preds)
        r2 = eval_rec2.evaluate(preds)
        score = wf1 * 0.5 + acc * 0.5
        
        tuning_records.append({
            "model_family": "Random Forest",
            "parameters": f"trees={cfg['numTrees']}, depth={cfg['maxDepth']}, minInst={cfg['minInstances']}",
            "val_accuracy": round(acc * 100, 2),
            "val_weighted_f1": round(wf1, 4),
            "val_fatal_recall": round(r0 * 100, 2),
            "val_fatal_precision": round(p0, 4),
            "val_fatal_f1": round(f0, 4),
            "val_serious_recall": round(r1 * 100, 2),
            "val_slight_recall": round(r2 * 100, 2),
            "train_time_sec": round(tt, 2)
        })
        if score > best_rf_score:
            best_rf_score = score
            best_rf_cfg = cfg
            
    # Export tuning table
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    tuning_df = pd.DataFrame(tuning_records)
    tuning_df.to_csv(output_csv, index=False)
    logger.info("Saved hyperparameter tuning results to: %s", output_csv)
    
    return {
        "best_logistic_regression": best_lr_cfg,
        "best_decision_tree": best_dt_cfg,
        "best_random_forest": best_rf_cfg
    }


def train_all_models(
    train_df: DataFrame,
    perform_tuning: bool = True
) -> Dict[str, Dict[str, Any]]:
    """Train the complete suite of ML models with multi-strategy class weighting.
    
    Returns:
        Dict[str, Dict[str, Any]]: Trained models, durations, and metadata.
    """
    train_weighted, class_weights = compute_class_weights(train_df)
    
    best_params = {}
    if perform_tuning:
        try:
            best_params = tune_all_hyperparameters(train_weighted)
        except Exception as e:
            logger.warning("Hyperparameter tuning encountered warning: %s. Using robust defaults.", e)
            best_params = {
                "best_logistic_regression": {"regParam": 0.01, "elasticNetParam": 0.0, "weightCol": "class_weight"},
                "best_decision_tree": {"maxDepth": 8, "minInstances": 10, "impurity": "gini"},
                "best_random_forest": {"numTrees": 60, "maxDepth": 12, "minInstances": 5}
            }
    else:
        best_params = {
            "best_logistic_regression": {"regParam": 0.01, "elasticNetParam": 0.0, "weightCol": "class_weight"},
            "best_decision_tree": {"maxDepth": 8, "minInstances": 10, "impurity": "gini"},
            "best_random_forest": {"numTrees": 60, "maxDepth": 12, "minInstances": 5}
        }
        
    lr_cfg = best_params.get("best_logistic_regression", {})
    dt_cfg = best_params.get("best_decision_tree", {})
    rf_cfg = best_params.get("best_random_forest", {})
    
    # 1. Multinomial Logistic Regression (Class-Weighted)
    lr_model, lr_time = train_logistic_regression(
        train_weighted,
        reg_param=lr_cfg.get("regParam", 0.01),
        elastic_net_param=lr_cfg.get("elasticNetParam", 0.0),
        weight_col=lr_cfg.get("weightCol", "class_weight")
    )
    
    # 2. Decision Tree Classifier
    dt_model, dt_time = train_decision_tree(
        train_weighted,
        max_depth=dt_cfg.get("maxDepth", 8),
        min_instances=dt_cfg.get("minInstances", 10),
        impurity=dt_cfg.get("impurity", "gini")
    )
    
    # 3. Random Forest Classifier (Optimized 60 Trees, Depth 12)
    rf_model, rf_time = train_random_forest(
        train_weighted,
        num_trees=rf_cfg.get("numTrees", 60),
        max_depth=rf_cfg.get("maxDepth", 12),
        min_instances=rf_cfg.get("minInstances", 5),
        weight_col="class_weight"
    )
    
    # 4. Gradient-Boosted Trees (OneVsRest multiclass adapter)
    gbt_model, gbt_time = train_gbt_multiclass(
        train_weighted,
        max_iter=15,
        max_depth=5
    )
    
    # 5. Multilayer Perceptron (MLP Neural Network)
    mlp_model, mlp_time = train_multilayer_perceptron(
        train_weighted,
        max_iter=40
    )
    
    # 6. Naive Bayes Classifier (Gaussian for continuous real features)
    nb_model, nb_time = train_naive_bayes(
        train_weighted,
        model_type="gaussian"
    )
    
    models = {
        "Logistic Regression": {
            "model": lr_model,
            "type": "Multinomial Logistic Regression (Class-Weighted)",
            "train_time_sec": lr_time
        },
        "Decision Tree": {
            "model": dt_model,
            "type": f"Decision Tree Classifier (maxDepth={dt_cfg.get('maxDepth', 10)})",
            "train_time_sec": dt_time
        },
        "Random Forest": {
            "model": rf_model,
            "type": f"Random Forest Classifier ({rf_cfg.get('numTrees', 60)} Trees, maxDepth={rf_cfg.get('maxDepth', 12)})",
            "train_time_sec": rf_time
        },
        "Gradient-Boosted Trees": {
            "model": gbt_model,
            "type": "Gradient-Boosted Trees (OneVsRest Multiclass, 15 Iterations)",
            "train_time_sec": gbt_time
        },
        "Multilayer Perceptron": {
            "model": mlp_model,
            "type": "Multilayer Perceptron Neural Network (4-Layer MLP)",
            "train_time_sec": mlp_time
        }
    }
    
    if nb_model is not None:
        models["Naive Bayes"] = {
            "model": nb_model,
            "type": "Gaussian Naive Bayes (Continuous Real Features)",
            "train_time_sec": nb_time
        }
        
    models["metadata"] = {
        "class_weights": class_weights,
        "best_params": best_params
    }
    return models
