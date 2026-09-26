"""
Official Challenge Metric: Macro-Averaged Per-Entity F_0.5 Evaluator
Compares predicted matching results against ground truth.
"""
import sys
import numpy as np
import pandas as pd


def compute_per_entity_f05(gt_series: pd.Series, pred_series: pd.Series) -> dict:
    beta_sq = 0.5 ** 2  # 0.25
    scores = []
    
    tp_total = 0
    fp_total = 0
    fn_total = 0
    singletons_correct = 0
    singletons_total = 0

    for g_val, p_val in zip(gt_series, pred_series):
        g_set = set(g_val.split(',')) if g_val else set()
        p_set = set(p_val.split(',')) if p_val else set()

        # True singleton
        if not g_set:
            singletons_total += 1
            if not p_set:
                singletons_correct += 1
                scores.append(1.0)
            else:
                scores.append(0.0)
            continue

        # Missed non-singleton
        if not p_set:
            scores.append(0.0)
            fn_total += len(g_set)
            continue

        tp = len(g_set & p_set)
        fp = len(p_set - g_set)
        fn = len(g_set - p_set)

        tp_total += tp
        fp_total += fp
        fn_total += fn

        if tp == 0:
            scores.append(0.0)
            continue

        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f05 = (1.0 + beta_sq) * prec * rec / (beta_sq * prec + rec)
        scores.append(f05)

    macro_f05 = float(np.mean(scores))
    micro_prec = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    micro_rec = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0

    return {
        "macro_f05": macro_f05,
        "singleton_accuracy": (singletons_correct / singletons_total) if singletons_total > 0 else 0.0,
        "singletons_count": singletons_total,
        "micro_precision": micro_prec,
        "micro_recall": micro_rec,
        "total_evaluated": len(scores)
    }


def evaluate(gt_path: str, pred_path: str):
    print(f"Loading ground truth: {gt_path}")
    gt = pd.read_csv(gt_path, sep='\t', dtype=str, keep_default_na=False)
    
    print(f"Loading predictions:  {pred_path}")
    pred = pd.read_csv(pred_path, sep='\t', dtype=str, keep_default_na=False)

    gt_col = 'source1_entity_id' if 'source1_entity_id' in gt.columns else 'entity_id'
    pred_col = 'source1_entity_id' if 'source1_entity_id' in pred.columns else 'entity_id'

    merged = gt[[gt_col, 'matched_entity_ids']].merge(
        pred[[pred_col, 'matched_entity_ids']],
        left_on=gt_col,
        right_on=pred_col,
        how='left',
        suffixes=('_gt', '_pred')
    )
    merged['matched_entity_ids_pred'] = merged['matched_entity_ids_pred'].fillna('')

    metrics = compute_per_entity_f05(
        merged['matched_entity_ids_gt'],
        merged['matched_entity_ids_pred']
    )

    print("\n" + "="*45)
    print(f"  MACRO F_0.5 SCORE : {metrics['macro_f05']:.5f}")
    print("="*45)
    print(f"  Entities Evaluated : {metrics['total_evaluated']:,}")
    print(f"  Singleton Accuracy : {metrics['singleton_accuracy']*100:.2f}% ({metrics['singletons_count']:,} singletons)")
    print(f"  Pairwise Precision : {metrics['micro_precision']*100:.2f}%")
    print(f"  Pairwise Recall    : {metrics['micro_recall']*100:.2f}%")
    print("="*45)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python utils/eval_macro_f05.py <ground_truth.tsv> <predictions.tsv>")
        sys.exit(1)
    evaluate(sys.argv[1], sys.argv[2])
