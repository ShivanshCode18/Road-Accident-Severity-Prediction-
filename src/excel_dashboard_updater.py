"""Excel Dashboard Synchronization Module.

Updates Microsoft Excel workbook (Road_Accident_Severity_Professional_Dashboard.xlsx)
with the latest PySpark ML evaluation outputs, confusion matrices, and test predictions.
Operates via high-speed OpenXML zip streaming to preserve 100% of charts, shapes, formulas,
and conditional formats without memory bloat.
"""

import zipfile
import shutil
import logging
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
import json
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXCEL_PATH = PROJECT_ROOT / "dataset" / "excel" / "Road_Accident_Severity_Professional_Dashboard.xlsx"
EXCEL_WORKBOOK_PATH = EXCEL_PATH
METRICS_CSV_PATH = PROJECT_ROOT / "dataset" / "processed" / "model_performance_metrics.csv"
PREDICTIONS_CSV_PATH = PROJECT_ROOT / "dataset" / "processed" / "excel_dashboard_predictions.csv"
SUMMARY_JSON_PATH = PROJECT_ROOT / "models" / "evaluation_summary.json"


def update_sheet4_metrics(s4_xml_bytes: bytes, evaluations: dict) -> bytes:
    """Update Sheet 4 (ML Model Performance) source table and confusion matrix cells."""
    ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    ET.register_namespace("", "http://schemas.openxmlformats.org/spreadsheetml/2006/main")
    root = ET.fromstring(s4_xml_bytes)
    
    # 1. Update source table (Rows 105 to 117)
    # Columns: G = Logistic Regression, J = Decision Tree, M = Random Forest
    model_col_map = {
        "Logistic Regression": "G",
        "Decision Tree": "J",
        "Random Forest": "M"
    }
    
    metric_row_map = {
        "accuracy": 105,
        "weighted_precision": 106,
        "weighted_recall": 107,
        "weighted_f1": 108,
        "macro_f1": 109,
        "fatal_recall": 110,
        "fatal_precision": 111,
        "fatal_f1": 112,
        "serious_recall": 113,
        "serious_f1": 114,
        "slight_recall": 115,
        "slight_f1": 116,
        "train_time_sec": 117
    }
    
    for mname, col_letter in model_col_map.items():
        if mname not in evaluations:
            continue
        ev = evaluations[mname]
        values = {
            "accuracy": ev["accuracy"] * 100.0,
            "weighted_precision": ev["weighted_precision"],
            "weighted_recall": ev["weighted_recall"],
            "weighted_f1": ev["weighted_f1"],
            "macro_f1": ev["macro_f1"],
            "fatal_recall": ev["fatal_recall"] * 100.0,
            "fatal_precision": ev["per_class"]["Fatal"]["precision"],
            "fatal_f1": ev["per_class"]["Fatal"]["f1"],
            "serious_recall": ev["per_class"]["Serious"]["recall"] * 100.0,
            "serious_f1": ev["per_class"]["Serious"]["f1"],
            "slight_recall": ev["per_class"]["Slight"]["recall"] * 100.0,
            "slight_f1": ev["per_class"]["Slight"]["f1"],
            "train_time_sec": ev.get("train_time_sec", 0.0)
        }
        
        for metric_name, row_num in metric_row_map.items():
            cell_ref = f"{col_letter}{row_num}"
            cell = root.find(f".//main:row[@r='{row_num}']/main:c[@r='{cell_ref}']", ns)
            if cell is not None:
                val_node = cell.find("main:v", ns)
                if val_node is None:
                    val_node = ET.SubElement(cell, "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v")
                val_node.text = str(round(values[metric_name], 4))
                # Remove string type attribute if present so it renders as numeric
                if "t" in cell.attrib:
                    del cell.attrib["t"]

    # 2. Update Confusion Matrices (Rows 82 to 85)
    # LR: D82:F84 (counts), G82:G84 (recall), D85:F85 (precision)
    # DT: K82:M84 (counts), N82:N84 (recall), K85:M85 (precision)
    # RF: R82:T84 (counts), U82:U84 (recall), R85:T85 (precision)
    cm_map = {
        "Logistic Regression": {"cols": ["D", "E", "F"], "rec_col": "G", "prec_row": 85},
        "Decision Tree":       {"cols": ["K", "L", "M"], "rec_col": "N", "prec_row": 85},
        "Random Forest":       {"cols": ["R", "S", "T"], "rec_col": "U", "prec_row": 85}
    }
    
    row_labels = ["Fatal (True)", "Serious (True)", "Slight (True)"]
    class_names = ["Fatal", "Serious", "Slight"]
    
    for mname, cfg in cm_map.items():
        if mname not in evaluations:
            continue
        ev = evaluations[mname]
        cm = ev["confusion_matrix"]
        per_class = ev["per_class"]
        
        for i, row_lbl in enumerate(row_labels):
            row_num = 82 + i
            r_data = cm[row_lbl]
            
            # Counts
            for j, cname in enumerate(class_names):
                cell_ref = f"{cfg['cols'][j]}{row_num}"
                cell = root.find(f".//main:row[@r='{row_num}']/main:c[@r='{cell_ref}']", ns)
                if cell is not None:
                    val_node = cell.find("main:v", ns)
                    if val_node is None:
                        val_node = ET.SubElement(cell, "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v")
                    val_node.text = str(r_data[cname])
                    if "t" in cell.attrib:
                        del cell.attrib["t"]
                        
            # Row Recall
            cname = class_names[i]
            rec_val = per_class[cname]["recall"]
            rec_cell_ref = f"{cfg['rec_col']}{row_num}"
            rec_cell = root.find(f".//main:row[@r='{row_num}']/main:c[@r='{rec_cell_ref}']", ns)
            if rec_cell is not None:
                v_node = rec_cell.find("main:v", ns)
                if v_node is None:
                    v_node = ET.SubElement(rec_cell, "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v")
                v_node.text = str(round(rec_val, 4))
                if "t" in rec_cell.attrib:
                    del rec_cell.attrib["t"]
                    
        # Precision row (85)
        for j, cname in enumerate(class_names):
            prec_val = per_class[cname]["precision"]
            prec_cell_ref = f"{cfg['cols'][j]}85"
            p_cell = root.find(f".//main:row[@r='85']/main:c[@r='{prec_cell_ref}']", ns)
            if p_cell is not None:
                v_node = p_cell.find("main:v", ns)
                if v_node is None:
                    v_node = ET.SubElement(p_cell, "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v")
                v_node.text = str(round(prec_val, 4))
                if "t" in p_cell.attrib:
                    del p_cell.attrib["t"]

    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def update_sheet5_predictions(s5_xml_bytes: bytes, pred_df: pd.DataFrame) -> bytes:
    """Update Sheet 5 (Prediction Analysis) table with latest test predictions and probabilities."""
    ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    ET.register_namespace("", "http://schemas.openxmlformats.org/spreadsheetml/2006/main")
    root = ET.fromstring(s5_xml_bytes)
    
    # Fast row-by-row update for rows 77 to 77 + len(pred_df) - 1
    # Updating columns: K (predicted_severity), M (prediction_correct),
    # N (prob_fatal), O (prob_serious), P (prob_slight)
    logger.info("Synchronizing %d predictions into Sheet 5 XML...", len(pred_df))
    
    # Pre-extract numpy arrays for fast vectorized access
    p_sev = pred_df["predicted_severity"].values
    p_cor = pred_df["prediction_correct"].values
    p_fat = pred_df["prob_fatal"].values
    p_ser = pred_df["prob_serious"].values
    p_sli = pred_df["prob_slight"].values
    
    row_nodes = root.findall(".//main:row", ns)
    row_dict = {int(r.attrib["r"]): r for r in row_nodes if int(r.attrib.get("r", 0)) >= 77}
    
    for idx in range(len(pred_df)):
        r_num = 77 + idx
        row = row_dict.get(r_num)
        if row is None:
            continue
            
        updates = {
            f"K{r_num}": str(int(p_sev[idx])),
            f"M{r_num}": str(int(p_cor[idx])),
            f"N{r_num}": str(round(float(p_fat[idx]), 4)),
            f"O{r_num}": str(round(float(p_ser[idx]), 4)),
            f"P{r_num}": str(round(float(p_sli[idx]), 4)),
        }
        
        for c in row.findall("main:c", ns):
            ref = c.attrib.get("r")
            if ref in updates:
                v_node = c.find("main:v", ns)
                if v_node is not None:
                    v_node.text = updates[ref]
                if "t" in c.attrib:
                    del c.attrib["t"]

    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def update_excel_workbook(
    excel_path: Path = EXCEL_PATH,
    evaluations: Optional[dict] = None,
    predictions_csv: Path = PREDICTIONS_CSV_PATH
) -> Path:
    """Stream-update Road_Accident_Severity_Professional_Dashboard.xlsx safely."""
    logger.info("Synchronizing Excel Dashboard deliverable at: %s...", excel_path)
    
    if evaluations is None:
        if SUMMARY_JSON_PATH.exists():
            with open(SUMMARY_JSON_PATH, "r", encoding="utf-8") as f:
                raw_summary = json.load(f)
                evaluations = raw_summary.get("models", raw_summary)
        else:
            raise FileNotFoundError(f"Cannot find evaluation summary at {SUMMARY_JSON_PATH}")
            
    df_pred = pd.read_csv(predictions_csv)
    
    temp_excel = excel_path.with_suffix(".tmp.xlsx")
    
    with zipfile.ZipFile(excel_path, "r") as zin:
        with zipfile.ZipFile(temp_excel, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                content = zin.read(item.filename)
                if item.filename == "xl/worksheets/sheet4.xml":
                    logger.info("Updating Sheet 4 (ML Model Performance) in Excel workbook...")
                    content = update_sheet4_metrics(content, evaluations)
                elif item.filename == "xl/worksheets/sheet5.xml":
                    logger.info("Updating Sheet 5 (Prediction Analysis) in Excel workbook...")
                    content = update_sheet5_predictions(content, df_pred)
                zout.writestr(item, content)
                
    # Atomic file replace
    shutil.move(str(temp_excel), str(excel_path))
    logger.info("Successfully updated Excel dashboard deliverable: %s", excel_path)
    return excel_path

