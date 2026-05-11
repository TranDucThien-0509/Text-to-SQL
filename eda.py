import json
import os

DATA_DIR = "word-level"


def load_json(file):
    with open(os.path.join(DATA_DIR, file), "r", encoding="utf-8") as f:
        return json.load(f)


def load_sql(file):
    with open(os.path.join(DATA_DIR, file), "r", encoding="utf-8") as f:
        return f.readlines()


def load_all():
    data = {}

    data["train"] = load_json("train.json")
    data["dev"] = load_json("dev.json")
    data["test"] = load_json("test.json")
    data["tables"] = load_json("tables.json")
    data["gold_sql"] = load_sql("test_gold.sql")

    return data


if __name__ == "__main__":
    data = load_all()

#    print("Train size:", len(data["train"]))
#    print("Dev size:", len(data["dev"]))
#    print("Test size:", len(data["test"]))
#    print("Tables:", len(data["tables"]))

    # xem thử 1 sample
    print("\n=== SAMPLE ===")
    print(data["gold_sql"][0])