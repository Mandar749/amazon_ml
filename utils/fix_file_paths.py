from pathlib import Path

# 1. Update predict.py
predict_path = Path("code/business_entity_resolution/src/pipeline/predict.py")
if predict_path.exists():
    text = predict_path.read_text(encoding="utf-8")
    text = text.replace('default="output/submission.tsv"', 'default="output/matching_results.tsv"')
    text = text.replace('default="output/submission_final.tsv"', 'default="output/matching_results.tsv"')
    predict_path.write_text(text, encoding="utf-8")
    print("[PASS] Updated predict.py output default to output/matching_results.tsv")

# 2. Update verify_submission.py
v_path = Path("utils/verify_submission.py")
if v_path.exists():
    v_text = v_path.read_text(encoding="utf-8")
    v_text = v_text.replace("submission_final.tsv", "matching_results.tsv")
    v_path.write_text(v_text, encoding="utf-8")
    print("[PASS] Updated utils/verify_submission.py target to output/matching_results.tsv")
