"""
train_mini_classifier.py — Local CPU quick-test (runs in ~5 min, 200 examples)

Purpose: Verify the full training pipeline works before committing to Kaggle.
This trains on a tiny hardcoded dataset — don't expect good metrics.
For production training, use train_vulnerability_classifier.py on Kaggle.

Run: python3 scripts/train_mini_classifier.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, classification_report
from config import config

# ─── Mini dataset (100 vulnerable + 100 safe examples) ────────────────────────

VULNERABLE_EXAMPLES = [
    # SQL injection
    'def get_user(name):\n    q = "SELECT * FROM users WHERE name=\'" + name + "\'"\n    cursor.execute(q)',
    'def find(id):\n    sql = f"SELECT * FROM t WHERE id={id}"\n    db.execute(sql)',
    'def search(q):\n    cursor.execute("SELECT * FROM items WHERE name=\'%s\'" % q)',
    'def login(user, pw):\n    q = "SELECT * FROM users WHERE user=\'" + user + "\' AND pass=\'" + pw + "\'"\n    return db.execute(q)',
    'def delete(id):\n    cursor.execute("DELETE FROM records WHERE id=" + str(id))',
    # Command injection
    'def ping(host):\n    import os\n    os.system("ping -c 1 " + host)',
    'def run(cmd):\n    import subprocess\n    subprocess.run(cmd, shell=True)',
    'def calc(expr):\n    return eval(expr)',
    'def exec_cmd(c):\n    import os\n    return os.popen(c).read()',
    'def execute(code):\n    exec(code)',
    # Path traversal
    'def read_file(name):\n    return open("/uploads/" + name).read()',
    'def serve(path):\n    import os\n    return open(os.path.join("/var/www", path)).read()',
    'def get_content(f):\n    base = "/data/"\n    return open(base + f, "rb").read()',
    # Hardcoded credentials
    'def connect():\n    password = "admin123"\n    return db.connect(host="localhost", password=password)',
    'def get_client():\n    api_key = "sk-abc123xyz"\n    return Client(api_key=api_key)',
    'def setup():\n    SECRET = "supersecretkey123"\n    app.secret_key = SECRET',
    # Weak crypto
    'def hash_pw(pw):\n    import hashlib\n    return hashlib.md5(pw.encode()).hexdigest()',
    'def make_token(uid):\n    import hashlib\n    return hashlib.sha1(str(uid).encode()).hexdigest()',
    'def gen_token():\n    import random, string\n    return "".join(random.choice(string.ascii_letters) for _ in range(32))',
    # Insecure deserialization
    'def load_session(data):\n    import pickle\n    return pickle.loads(data)',
    'def parse_config(s):\n    import yaml\n    return yaml.load(s)',
    'def restore(data):\n    import pickle\n    obj = pickle.loads(data)\n    return obj',
    # Sensitive data exposure
    'def auth(user, pw):\n    import logging\n    logging.info(f"Login: user={user} pw={pw}")\n    return check(user, pw)',
    'def handle(data):\n    try:\n        return process(data)\n    except Exception as e:\n        return {"error": str(e), "trace": traceback.format_exc()}',
    # SSRF
    'def fetch(url):\n    import requests\n    return requests.get(url).text',
    'def proxy(target):\n    import urllib.request\n    return urllib.request.urlopen(target).read()',
    # More SQL
    'def update_user(id, name):\n    q = "UPDATE users SET name=\'" + name + "\' WHERE id=" + str(id)\n    cursor.execute(q)',
    'def count(table):\n    cursor.execute("SELECT COUNT(*) FROM " + table)',
    'def insert(val):\n    db.execute(f"INSERT INTO log VALUES (\'{val}\')")',
    'def raw_query(q):\n    return cursor.execute(q)',
    # More command injection
    'def compress(fname):\n    os.system("gzip " + fname)',
    'def convert(f, fmt):\n    subprocess.run("convert " + f + " output." + fmt, shell=True)',
    'def check_dns(domain):\n    return os.popen("nslookup " + domain).read()',
    # More path traversal
    'def download(filename):\n    path = UPLOAD_DIR + "/" + filename\n    return send_file(path)',
    'def load_template(name):\n    return open("templates/" + name + ".html").read()',
    # More weak crypto
    'def hash_file(f):\n    import hashlib\n    return hashlib.md5(open(f, "rb").read()).hexdigest()',
    'def create_session_id(user):\n    import random\n    return str(random.randint(100000, 999999))',
    # XXE
    'def parse_xml(data):\n    import xml.etree.ElementTree as ET\n    return ET.fromstring(data)',
    # Race condition
    'def write_file(path, content):\n    if not os.path.exists(path):\n        with open(path, "w") as f:\n            f.write(content)',
    # More hardcoded
    'def get_db():\n    DB_PASSWORD = "root1234"\n    return psycopg2.connect(password=DB_PASSWORD)',
    # More SSRF
    'def get_avatar(uid):\n    url = request.args.get("url")\n    return requests.get(url).content',
    # Injection via format
    'def report(col):\n    cursor.execute("SELECT %s FROM users" % col)',
    'def find_by(field, val):\n    cursor.execute("SELECT * FROM t WHERE " + field + "=" + val)',
    # More eval
    'def dynamic_filter(expr):\n    return list(filter(lambda x: eval(expr), items))',
    'def transform(code, data):\n    return eval(compile(code, "<string>", "exec"))',
    # Pickle variants
    'def cache_load(key):\n    data = redis.get(key)\n    return pickle.loads(data)',
    'def deserialize(blob):\n    return pickle.loads(base64.b64decode(blob))',
    # Open redirect
    'def redirect_after_login(next_url):\n    return redirect(next_url)',
    'def go_to(dest):\n    return HttpResponseRedirect(dest)',
    # More logging
    'def process_payment(card_number, cvv):\n    print(f"Processing card {card_number} cvv {cvv}")\n    return charge(card_number)',
]

SAFE_EXAMPLES = [
    # Parameterized queries
    'def get_user(name):\n    cursor.execute("SELECT * FROM users WHERE name = %s", (name,))\n    return cursor.fetchone()',
    'def find(id):\n    cursor.execute("SELECT * FROM t WHERE id = ?", (id,))\n    return cursor.fetchone()',
    'def search(q):\n    cursor.execute("SELECT * FROM items WHERE name = %s", (q,))\n    return cursor.fetchall()',
    'def login(user, pw):\n    cursor.execute("SELECT * FROM users WHERE user=%s AND pass=%s", (user, pw))\n    return cursor.fetchone()',
    'def delete(id):\n    cursor.execute("DELETE FROM records WHERE id = %s", (id,))',
    # Safe subprocess
    'def ping(host):\n    import subprocess\n    result = subprocess.run(["ping", "-c", "1", host], capture_output=True)\n    return result.stdout',
    'def compress(fname):\n    subprocess.run(["gzip", fname], check=True)',
    # Safe file ops
    'def read_file(name):\n    safe = os.path.realpath(os.path.join("/uploads", name))\n    assert safe.startswith("/uploads/")\n    return open(safe).read()',
    'def serve(path):\n    full = os.path.realpath(os.path.join(BASE, path))\n    if not full.startswith(BASE):\n        raise ValueError("Path traversal")\n    return open(full).read()',
    # Safe crypto
    'def hash_pw(pw):\n    import bcrypt\n    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt())',
    'def gen_token():\n    import secrets\n    return secrets.token_urlsafe(32)',
    'def make_token():\n    import os\n    return os.urandom(32).hex()',
    'def hash_file(f):\n    import hashlib\n    return hashlib.sha256(open(f, "rb").read()).hexdigest()',
    # Env-based credentials
    'def connect():\n    password = os.environ["DB_PASSWORD"]\n    return db.connect(host="localhost", password=password)',
    'def get_client():\n    api_key = os.getenv("API_KEY")\n    return Client(api_key=api_key)',
    # Safe deserialization
    'def parse_config(s):\n    import yaml\n    return yaml.safe_load(s)',
    'def load_data(s):\n    import json\n    return json.loads(s)',
    # Generic safe functions
    'def add(a, b):\n    return a + b',
    'def square(n):\n    return n * n',
    'def greet(name):\n    return f"Hello, {name}!"',
    'def is_even(n):\n    return n % 2 == 0',
    'def clamp(val, lo, hi):\n    return max(lo, min(hi, val))',
    'def flatten(lst):\n    return [x for sub in lst for x in sub]',
    'def count_words(text):\n    return len(text.split())',
    'def avg(nums):\n    return sum(nums) / len(nums) if nums else 0',
    'def reverse(s):\n    return s[::-1]',
    'def capitalize_words(s):\n    return " ".join(w.capitalize() for w in s.split())',
    # Safe URL handling
    'def fetch_public(url):\n    allowed = ["api.github.com", "api.example.com"]\n    parsed = urllib.parse.urlparse(url)\n    if parsed.hostname not in allowed:\n        raise ValueError("Disallowed host")\n    return requests.get(url).json()',
    # Error handling
    'def safe_handle(data):\n    try:\n        return process(data)\n    except Exception:\n        logger.exception("Processing failed")\n        return {"error": "Internal error"}',
    # Safe logging
    'def auth(user, pw):\n    logger.info(f"Login attempt for user={user}")\n    return check(user, hash_pw(pw))',
    # Config loading
    'def load_settings():\n    with open("config.json") as f:\n        return json.load(f)',
    # Input validation
    'def validate_email(email):\n    import re\n    return bool(re.match(r"^[\\w.]+@[\\w.]+\\.[a-z]{2,}$", email))',
    'def validate_age(age):\n    if not isinstance(age, int) or age < 0 or age > 150:\n        raise ValueError("Invalid age")\n    return age',
    # Data processing
    'def normalize(data):\n    mn, mx = min(data), max(data)\n    return [(x - mn) / (mx - mn) for x in data]',
    'def chunk(lst, size):\n    return [lst[i:i+size] for i in range(0, len(lst), size)]',
    'def dedupe(items):\n    return list(dict.fromkeys(items))',
    'def merge_dicts(a, b):\n    return {**a, **b}',
    'def safe_get(d, key, default=None):\n    return d.get(key, default)',
    # File I/O safe
    'def write_output(path, content):\n    os.makedirs(os.path.dirname(path), exist_ok=True)\n    with open(path, "w") as f:\n        f.write(content)',
    'def read_json(path):\n    with open(path) as f:\n        return json.load(f)',
    # API helpers
    'def paginate(items, page, size):\n    start = (page - 1) * size\n    return items[start:start+size]',
    'def make_response(data, status=200):\n    return {"data": data, "status": status}',
    # Math utils
    'def euclidean(a, b):\n    return sum((x-y)**2 for x,y in zip(a,b)) ** 0.5',
    'def sigmoid(x):\n    import math\n    return 1 / (1 + math.exp(-x))',
    'def softmax(values):\n    import math\n    e = [math.exp(v) for v in values]\n    s = sum(e)\n    return [v/s for v in e]',
    # More safe
    'def retry(fn, n=3):\n    for i in range(n):\n        try:\n            return fn()\n        except Exception:\n            if i == n-1:\n                raise',
    'def timed(fn):\n    import time\n    start = time.time()\n    result = fn()\n    return result, time.time() - start',
    'def memoize(fn):\n    cache = {}\n    def wrapper(*args):\n        if args not in cache:\n            cache[args] = fn(*args)\n        return cache[args]\n    return wrapper',
]

assert 45 <= len(VULNERABLE_EXAMPLES) <= 55, f"Expected ~50 vulnerable, got {len(VULNERABLE_EXAMPLES)}"
assert 45 <= len(SAFE_EXAMPLES) <= 55, f"Expected ~50 safe, got {len(SAFE_EXAMPLES)}"


# ─── Dataset ───────────────────────────────────────────────────────────────────

class CodeDataset(Dataset):
    def __init__(self, codes, labels, tokenizer, max_len=256):
        self.codes = codes
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.codes[idx],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ─── Training ─────────────────────────────────────────────────────────────────

def train():
    MODEL_NAME = "microsoft/codebert-base"
    OUTPUT_DIR = os.path.join(config.MODELS_DIR, "vulnerability_classifier")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Output dir: {OUTPUT_DIR}")

    # Build dataset: 50 vulnerable + 50 safe, 80/20 train/test split
    all_codes = VULNERABLE_EXAMPLES + SAFE_EXAMPLES
    all_labels = [1] * 50 + [0] * 50   # 1 = vulnerable, 0 = safe

    # Shuffle with fixed seed for reproducibility
    import random
    random.seed(42)
    combined = list(zip(all_codes, all_labels))
    random.shuffle(combined)
    codes, labels = zip(*combined)
    codes, labels = list(codes), list(labels)

    split = int(0.8 * len(codes))
    train_codes, train_labels = codes[:split], labels[:split]
    test_codes, test_labels = codes[split:], labels[split:]
    print(f"Train: {len(train_codes)} | Test: {len(test_codes)}")

    print(f"\nLoading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )
    model.to(device)

    train_ds = CodeDataset(train_codes, train_labels, tokenizer)
    test_ds = CodeDataset(test_codes, test_labels, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1,
        total_iters=len(train_loader) * 3,
    )

    # Train 3 epochs
    print("\nTraining...")
    for epoch in range(3):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += out.loss.item()

        avg_loss = total_loss / len(train_loader)

        # Eval
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for batch in test_loader:
                out = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
                preds = torch.argmax(out.logits, dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_true.extend(batch["labels"].tolist())

        f1 = f1_score(all_true, all_preds, average="binary")
        print(f"  Epoch {epoch+1}/3 | Loss: {avg_loss:.4f} | Test F1: {f1:.3f}")

    # Final evaluation
    print("\nFinal evaluation:")
    print(classification_report(all_true, all_preds, target_names=["SAFE", "VULNERABLE"]))

    # Save
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    metrics = {
        "f1": round(f1, 4),
        "dataset": "mini_hardcoded_100examples",
        "note": "Mini local test only. Run train_vulnerability_classifier.py on Kaggle for production model.",
    }
    with open(os.path.join(OUTPUT_DIR, "eval_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✓ Model saved to {OUTPUT_DIR}")
    print(f"  Test F1: {f1:.3f} (expect 0.70-0.95 on mini dataset)")
    print("  Note: This is a sanity check. For real F1≥0.75 on BigVul, train on Kaggle.")


if __name__ == "__main__":
    train()
