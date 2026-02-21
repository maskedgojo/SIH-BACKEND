import csv, os

class CSVLogger:
    def __init__(self, out_path):
        self.out_path = out_path
        self.fields_written = False

    def log(self, records):
        if not records: return
        fields = list(records[0].keys())
        write_header = not os.path.exists(self.out_path)
        with open(self.out_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if write_header:
                writer.writeheader()
            writer.writerows(records)
