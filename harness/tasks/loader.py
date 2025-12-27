import json
import pandas as pd


def load_dataset(task_name, config):
    """
    Load and normalize task data into a unified list of dictionaries.

    Each entry has:
    - text: input text constructed from configured keys
    - label: ground-truth label (if available)
    """
    records = []
    data_path = config["path"]
    data_format = config["format"]

    if data_format == "jsonl":
        with open(data_path, "r") as f:
            for line in f:
                row = json.loads(line)
                text_parts = [row[key] for key in config["keys"]]
                records.append({
                    "text": " ".join(text_parts),
                    "label": row.get(config.get("label_key")),
                })

    elif data_format == "csv":
        df = pd.read_csv(data_path)
        text_key = config["keys"][0]
        label_key = config["label_key"]

        for _, row in df.iterrows():
            records.append({
                "text": row[text_key],
                "label": row[label_key],
            })

    elif data_format == "tsv_no_header":
        with open(data_path, "r") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue

                records.append({
                    "text": parts[1],
                    "label": parts[0],
                })

    elif data_format == "parallel_text":
        with open(data_path, "r") as src_file, open(config["ref_path"], "r") as ref_file:
            for src_line, ref_line in zip(src_file, ref_file):
                records.append({
                    "text": src_line.strip(),
                    "label": ref_line.strip(),
                })

    else:
        raise ValueError(f"Unsupported dataset format: {data_format}")

    return records
