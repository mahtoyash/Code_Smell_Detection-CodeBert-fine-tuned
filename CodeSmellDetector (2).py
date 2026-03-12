#!/usr/bin/env python
# coding: utf-8

# # CodeSmellDetector: 5-Class Code Smell Detection with CodeBERT
# 
# Fine-tunes `microsoft/codebert-base` to detect Python code smells:

# | 0 | Long Method | Function >20 lines or cyclomatic complexity >10 |
# | 1 | Large Parameter List | Function with >5 parameters |
# | 2 | God Class | Class with >10 methods AND >15 attributes |
# | 3 | Feature Envy | Method uses external objects more than self |
# | 4 | Clean Code | No smells detected |

# In[1]:


# CELL 1: Environment Setup
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import subprocess
import platform

print(f"Python: {sys.version}")
print(f"Platform: {platform.system()} {platform.release()}")

# Install required packages
packages = [
    "torch", "transformers", "datasets", "accelerate",
    "radon", "pandas", "matplotlib", "scikit-learn", "tqdm", "seaborn"
]

for pkg in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import torch
print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing device: {DEVICE}")


# In[2]:


# CELL 2: Repository Cloning
from pathlib import Path
from tqdm.auto import tqdm

DEFAULT_REPOS = [
    "https://github.com/django/django.git",
    "https://github.com/pandas-dev/pandas.git",
    "https://github.com/scikit-learn/scikit-learn.git",
    "https://github.com/ansible/ansible.git",
    "https://github.com/apache/airflow.git",
    "https://github.com/scrapy/scrapy.git",
    "https://github.com/keras-team/keras.git",
    "https://github.com/saltstack/salt.git",
    "https://github.com/pallets/flask.git",
    "https://github.com/psf/requests.git",
    "https://github.com/encode/httpx.git",
    "https://github.com/tiangolo/fastapi.git",
    "https://github.com/sqlalchemy/sqlalchemy.git",
    "https://github.com/celery/celery.git",
    "https://github.com/spotify/luigi.git",
    "https://github.com/pallets/click.git",
    "https://github.com/pallets/jinja.git",
    "https://github.com/pallets/werkzeug.git",
    "https://github.com/pytest-dev/pytest.git",
    "https://github.com/pypa/pip.git",
    "https://github.com/psf/black.git",
    "https://github.com/matplotlib/matplotlib.git",
    "https://github.com/networkx/networkx.git",
    "https://github.com/python-pillow/Pillow.git",
    "https://github.com/paramiko/paramiko.git",
    "https://github.com/docker/docker-py.git",
    "https://github.com/redis/redis-py.git",
    "https://github.com/httpie/httpie.git",
    "https://github.com/encode/starlette.git",
    "https://github.com/pydantic/pydantic.git",
    "https://github.com/aio-libs/aiohttp.git",
    "https://github.com/tornadoweb/tornado.git",
    "https://github.com/benoitc/gunicorn.git",
    "https://github.com/geekcomputers/Python.git",
    "https://github.com/TheAlgorithms/Python.git",
    "https://github.com/faif/python-patterns.git",
    "https://github.com/jackfrued/Python-100-Days.git",
    "https://github.com/realpython/python-guide.git",
    "https://github.com/satwikkansal/wtfpython.git",
    "https://github.com/encode/django-rest-framework.git",
    "https://github.com/huge-success/sanic.git",
    "https://github.com/mongodb/mongo-python-driver.git",
    "https://github.com/aws/aws-cli.git"
]

REPOS_DIR = Path("./cloned_repos")
REPOS_DIR.mkdir(exist_ok=True)

def clone_repos(repo_urls=None):
    urls = repo_urls or DEFAULT_REPOS
    cloned_paths = []

    for url in tqdm(urls, desc="Cloning repos"):
        repo_name = url.split("/")[-1].replace(".git", "")
        dest = REPOS_DIR / repo_name

        if dest.exists():
            print(f"  [skip] {repo_name} already exists")
            cloned_paths.append(dest)
            continue

        try:
            result = subprocess.run(
                ["git", "clone", "--depth", "1", url, str(dest)],
                capture_output=True, text=True, timeout=180
            )
            if result.returncode == 0:
                print(f"  [ok] {repo_name}")
                cloned_paths.append(dest)
            else:
                print(f"  [fail] {repo_name}: {result.stderr[:80]}")
        except subprocess.TimeoutExpired:
            print(f"  [timeout] {repo_name} - skipping")
        except Exception as e:
            print(f"  [error] {repo_name}: {e}")

    return cloned_paths

repo_paths = clone_repos()
print(f"\n✓ Cloned {len(repo_paths)} repositories")


# In[3]:


# CELL 3: AST-Based Snippet Extraction
import ast
from dataclasses import dataclass
from typing import Literal

@dataclass
class CodeSnippet:
    code: str
    snippet_type: Literal["function", "class"]
    name: str
    file_path: str
    num_lines: int
    num_params: int
    num_methods: int
    num_attributes: int

def extract_snippets_from_file(file_path):
    snippets = []
    try:
        source = file_path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
    except:
        return snippets

    lines = source.splitlines()

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            try:
                start = node.lineno - 1
                end = node.end_lineno
                code = "\n".join(lines[start:end])
                num_params = len(node.args.args) + len(node.args.posonlyargs) + len(node.args.kwonlyargs)
                if node.args.vararg: num_params += 1
                if node.args.kwarg: num_params += 1

                snippets.append(CodeSnippet(
                    code=code,
                    snippet_type="function",
                    name=node.name,
                    file_path=str(file_path),
                    num_lines=end - start,
                    num_params=num_params,
                    num_methods=0,
                    num_attributes=0
                ))
            except:
                pass

        elif isinstance(node, ast.ClassDef):
            try:
                start = node.lineno - 1
                end = node.end_lineno
                code = "\n".join(lines[start:end])

                num_methods = sum(1 for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
                num_attributes = 0
                for n in node.body:
                    if isinstance(n, ast.Assign):
                        num_attributes += len(n.targets)
                    elif isinstance(n, ast.AnnAssign):
                        num_attributes += 1
                for n in node.body:
                    if isinstance(n, ast.FunctionDef) and n.name == "__init__":
                        for stmt in ast.walk(n):
                            if isinstance(stmt, ast.Assign):
                                for target in stmt.targets:
                                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                                        if target.value.id == "self":
                                            num_attributes += 1

                snippets.append(CodeSnippet(
                    code=code,
                    snippet_type="class",
                    name=node.name,
                    file_path=str(file_path),
                    num_lines=end - start,
                    num_params=0,
                    num_methods=num_methods,
                    num_attributes=num_attributes
                ))
            except:
                pass

    return snippets

def extract_all_snippets(repo_paths):
    all_snippets = []

    for repo_path in tqdm(repo_paths, desc="Extracting snippets"):
        py_files = list(repo_path.rglob("*.py"))
        for py_file in py_files:
            if "test" in str(py_file).lower():
                continue
            snippets = extract_snippets_from_file(py_file)
            all_snippets.extend(snippets)

    return all_snippets

snippets = extract_all_snippets(repo_paths)
print(f"\n✓ Extracted {len(snippets)} snippets")
print(f"  Functions: {sum(1 for s in snippets if s.snippet_type == 'function')}")
print(f"  Classes: {sum(1 for s in snippets if s.snippet_type == 'class')}")


# In[5]:


# CELL 4: FIXED Code Smell Labeling (Relaxed God Class threshold)
from radon.complexity import cc_visit
from collections import Counter
import random

# Smell thresholds -  GOD CLASS!
LONG_METHOD_LINES = 20
LONG_METHOD_CC = 10
GOD_CLASS_METHODS = 7       
GOD_CLASS_ATTRIBUTES = 10   
LARGE_PARAM_COUNT = 5
FEATURE_ENVY_THRESHOLD = 3

# Label mapping
LABEL_NAMES = ["long_method", "large_param_list", "god_class", "feature_envy", "clean_code"]
LABEL_MAP = {name: i for i, name in enumerate(LABEL_NAMES)}

def get_cyclomatic_complexity(code):
    try:
        blocks = cc_visit(code)
        if blocks:
            return max(b.complexity for b in blocks)
    except:
        pass
    return 0

def detect_feature_envy(code, snippet_type):
    if snippet_type != "function":
        return False

    try:
        tree = ast.parse(code)

        self_refs = 0
        external_refs = 0

        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    if node.value.id == "self":
                        self_refs += 1
                    elif node.value.id not in ("cls", "super"):
                        external_refs += 1

        if external_refs > self_refs and external_refs >= FEATURE_ENVY_THRESHOLD:
            return True
        return False
    except:
        return False

@dataclass
class LabeledSnippet:
    code: str
    label: int
    label_name: str

def get_smell_label(snippet):
    # God Class FIRST (since we have so few!)
    if snippet.snippet_type == "class":
        # Relaxed: 7+ methods AND 10+ attributes, OR 12+ methods alone
        if (snippet.num_methods >= GOD_CLASS_METHODS and snippet.num_attributes >= GOD_CLASS_ATTRIBUTES) or \
           (snippet.num_methods >= 12):
            return 2, "god_class"

    # Long Method
    if snippet.snippet_type == "function":
        cc = get_cyclomatic_complexity(snippet.code)
        if snippet.num_lines > LONG_METHOD_LINES or cc > LONG_METHOD_CC:
            return 0, "long_method"

    # Large Parameter List
    if snippet.snippet_type == "function" and snippet.num_params > LARGE_PARAM_COUNT:
        return 1, "large_param_list"

    # Feature Envy
    if snippet.snippet_type == "function":
        if detect_feature_envy(snippet.code, snippet.snippet_type):
            return 3, "feature_envy"

    # Clean code
    return 4, "clean_code"

# Label all snippets
print("Labeling snippets (fast AST-based)...")
labeled_snippets = []

for snippet in tqdm(snippets, desc="Labeling"):
    if len(snippet.code) < 50 or len(snippet.code) > 8000:
        continue

    label, label_name = get_smell_label(snippet)
    labeled_snippets.append(LabeledSnippet(
        code=snippet.code,
        label=label,
        label_name=label_name
    ))

print(f"\n✓ Labeled {len(labeled_snippets)} snippets")

label_counts = Counter(s.label_name for s in labeled_snippets)
print("\nLabel distribution (BEFORE balancing):")
for name in LABEL_NAMES:
    count = label_counts.get(name, 0)
    pct = count / len(labeled_snippets) * 100 if labeled_snippets else 0
    print(f"  {name}: {count} ({pct:.1f}%)")


# In[27]:


#  CELL 4b — REAL GOD CLASS MINER                             

import ast
import os
import subprocess
import textwrap
from pathlib import Path

# projects where god classes are common and real

GOD_CLASS_REPOS = [
    # ERP / Enterprise — notorious for massive model classes
    "https://github.com/odoo/odoo.git",

    # Home automation — large entity/component classes
    "https://github.com/home-assistant/core.git",

    # CMS systems — large model + view classes mixed together
    "https://github.com/django-cms/django-cms.git",
    "https://github.com/wagtail/wagtail.git",

    # Data science — DataFrame itself is a textbook god class
    "https://github.com/pandas-dev/pandas.git",

    # ORM — complex mapper and session classes
    "https://github.com/sqlalchemy/sqlalchemy.git",

    # OpenStack — large service/manager classes
    "https://github.com/openstack/nova.git",

    # Testing framework — large runner/session classes
    "https://github.com/pytest-dev/pytest.git",

    # Network automation — large module/task classes  
    "https://github.com/ansible/ansible.git",

    # E-commerce — large cart/order/product classes
    "https://github.com/saleor/saleor.git",
]

# ── God class thresholds (same as your Cell 4 labeling rules) ──
GOD_CLASS_MIN_METHODS = 7
GOD_CLASS_MIN_ATTRS   = 9

CLONE_DIR = Path("./repos_godclass")
CLONE_DIR.mkdir(exist_ok=True)


def clone_repo(url, target_dir):
    repo_name = url.split("/")[-1].replace(".git", "")
    repo_path = target_dir / repo_name
    if repo_path.exists():
        print(f"  [skip] {repo_name} already cloned")
        return repo_path
    print(f"  [clone] {repo_name}...")
    result = subprocess.run(
        ["git", "clone", "--depth=1", url, str(repo_path)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  [error] failed to clone {repo_name}: {result.stderr[:100]}")
        return None
    print(f"  [done] {repo_name}")
    return repo_path


def count_class_methods_and_attrs(node):
    """
    Count instance methods and self.x attributes in a class node.
    Same logic as your Cell 4 labeling rules.
    """
    methods = 0
    attrs   = set()

    for item in ast.walk(node):
        # Count methods (functions defined directly in class)
        if isinstance(item, ast.FunctionDef):
            # Only count direct methods, not nested functions
            methods += 1

        # Count self.x assignments → instance attributes
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if (isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"):
                    attrs.add(target.attr)

        if isinstance(item, ast.AnnAssign):
            if (isinstance(item.target, ast.Attribute)
                    and isinstance(item.target.value, ast.Name)
                    and item.target.value.id == "self"):
                attrs.add(item.target.attr)

    return methods, len(attrs)


def extract_god_classes_from_file(filepath):
    """
    Parse a Python file and extract classes that qualify as god classes.
    Returns list of (class_source_code, class_name) tuples.
    """
    results = []
    try:
        source = filepath.read_text(encoding="utf-8", errors="ignore")
        tree   = ast.parse(source)
    except Exception:
        return results

    lines = source.splitlines()

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        methods, attrs = count_class_methods_and_attrs(node)

        if methods >= GOD_CLASS_MIN_METHODS and attrs >= GOD_CLASS_MIN_ATTRS:
            # Extract full class source using line numbers
            try:
                start = node.lineno - 1
                end   = node.end_lineno
                class_lines = lines[start:end]
                class_source = "\n".join(class_lines)

                # Skip if too short or too long (noisy)
                if len(class_lines) < 30 or len(class_lines) > 500:
                    continue

                # Skip auto-generated or migration files
                fname = str(filepath)
                if any(skip in fname for skip in
                       ["migration", "generated", "test_", "_test",
                        "conftest", "setup.py", "alembic"]):
                    continue

                results.append((class_source, node.name))

            except Exception:
                continue

    return results


def mine_repo_for_god_classes(repo_path):
    """Walk all .py files in a repo and extract god classes."""
    god_classes = []
    py_files    = list(repo_path.rglob("*.py"))
    print(f"    Scanning {len(py_files)} files in {repo_path.name}...")

    for filepath in py_files:
        found = extract_god_classes_from_file(filepath)
        god_classes.extend(found)

    print(f"    Found {len(god_classes)} god classes in {repo_path.name}")
    return god_classes


# ══════════════════════════════════════════════════════════════
# MINE ALL REPOS
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("  Mining real god classes from production codebases...")
print("=" * 60)

all_real_god_classes = []

for repo_url in GOD_CLASS_REPOS:
    repo_path = clone_repo(repo_url, CLONE_DIR)
    if repo_path is None:
        continue
    found = mine_repo_for_god_classes(repo_path)
    all_real_god_classes.extend(found)

print(f"\n  Total real god classes found: {len(all_real_god_classes)}")


# ══════════════════════════════════════════════════════════════
# CONVERT TO LabeledSnippet AND APPEND TO labeled_snippets
# ══════════════════════════════════════════════════════════════
real_god_snippets = []

for source, class_name in all_real_god_classes:
    # Dedent to remove inconsistent indentation from extraction
    try:
        clean_source = textwrap.dedent(source).strip()
    except Exception:
        clean_source = source.strip()

    real_god_snippets.append(LabeledSnippet(
        code       = clean_source,
        label      = 2,
        label_name = "god_class"
    ))

# Deduplicate by source hash
seen     = set()
unique   = []
for s in real_god_snippets:
    h = hash(s.code[:200])   # hash first 200 chars as proxy
    if h not in seen:
        seen.add(h)
        unique.append(s)

real_god_snippets = unique
print(f"  After deduplication: {len(real_god_snippets)} unique god classes")

# ── Append to labeled_snippets from Cell 4 ────────────────────
labeled_snippets.extend(real_god_snippets)

print(f"\n  labeled_snippets now contains:")
counts = Counter(s.label_name for s in labeled_snippets)
for name in LABEL_NAMES:
    print(f"    {name}: {counts.get(name, 0)}")

print("\n  Cell 4b complete ✓")


# In[28]:


# ╔══════════════════════════════════════════════════════════════╗
# ║         CELL 5 — BRUTAL BALANCED DATASET v2                 ║
# ║  Hard positives + Hard negatives + Edge cases                ║
# ║  Target: 12,000 per class = 60,000 total                     ║
# ╚══════════════════════════════════════════════════════════════╝

import random
from collections import Counter

random.seed(42)

# ══════════════════════════════════════════════════════════════
# VOCABULARY BANKS
# ══════════════════════════════════════════════════════════════
FUNC_NAMES = [
    "process","handle","execute","validate","compute","transform",
    "generate","calculate","initialize","update","fetch","parse",
    "render","build","create","manage","dispatch","resolve",
    "prepare","finalize","check","verify","sync","apply","load"
]
VAR_NAMES = [
    "result","data","value","item","output","response","payload",
    "record","entry","node","token","content","buffer","config",
    "state","index","count","flag","status","temp","total","key",
    "error","message","context","target","source","current","prev"
]
CLASS_NAMES = [
    "Manager","Handler","Controller","Processor","Engine","Service",
    "Registry","Factory","Dispatcher","Coordinator","Validator",
    "Parser","Builder","Executor","Scheduler","Monitor","Tracker",
    "Adapter","Provider","Resolver","Aggregator","Supervisor"
]
ATTR_NAMES = [
    "name","value","data","config","logger","cache","db","conn",
    "session","timeout","retries","status","items","queue","registry",
    "settings","callbacks","listeners","handlers","pool","buffer",
    "lock","token","url","host","port","user","password","key",
    "prefix","limit","offset","cursor","mode","level","scope","tag",
    "active","enabled","debug","version","created","updated","owner"
]
METHOD_NAMES = [
    "initialize","setup","teardown","validate","process","execute",
    "handle","dispatch","register","unregister","connect","disconnect",
    "start","stop","reset","update","delete","create","get","set",
    "load","save","parse","serialize","encode","decode","format",
    "flush","refresh","sync","authenticate","authorize","notify","send"
]
OBJ_NAMES = [
    "user","order","product","account","invoice","customer",
    "payment","report","request","session","profile","transaction",
    "document","record","entry","entity","item","resource"
]
TYPES    = ["int","str","float","bool","List[str]","Dict[str,Any]",
            "Optional[str]","Optional[int]","Any","bytes","tuple"]
DEFAULTS = ["None","0","False","''","[]","{}","True","-1","0.0"]

DOMAIN_GROUPS = {
    "auth":    ["login","logout","register","verify_token",
                "reset_password","check_permissions","revoke_token"],
    "db":      ["save_record","load_record","delete_record",
                "run_query","migrate","rollback","bulk_insert"],
    "email":   ["send_email","format_email","queue_email",
                "verify_address","track_open","render_template"],
    "cache":   ["cache_get","cache_set","cache_delete",
                "invalidate_all","warm_cache","cache_stats"],
    "logging": ["log_info","log_error","log_warning",
                "log_debug","rotate_logs","archive_logs"],
    "file":    ["read_file","write_file","delete_file",
                "compress","move_file","list_files","watch_dir"],
    "http":    ["get","post","put","delete","patch",
                "handle_redirect","parse_response","stream"],
    "notify":  ["push_notification","send_sms","webhook",
                "publish_event","broadcast","subscribe"],
    "report":  ["generate_report","export_csv","export_pdf",
                "aggregate","summarize","schedule_report"],
    "payment": ["charge","refund","validate_card","create_invoice",
                "apply_discount","process_payout","reconcile"],
}

FIELD_TYPES = [
    "CharField(max_length=255)", "IntegerField(default=0)",
    "BooleanField(default=False)", "DateTimeField(auto_now=True)",
    "TextField(blank=True)", "FloatField(default=0.0)",
    "EmailField()", "URLField(blank=True)", "ForeignKey('self', null=True)"
]

REAL_VALUES = {
    "int":  ["0","1","-1","100","None"],
    "str":  ["''","None","'default'","'pending'","'active'"],
    "bool": ["False","True","None"],
    "list": ["[]","None"],
    "dict": ["{}","None"],
}


# ══════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════
def rand_attr_init(attr):
    kind = random.choice(["int","str","bool","list","dict"])
    val  = random.choice(REAL_VALUES[kind])
    return f"self.{attr} = {val}"

def make_method_body(attrs):
    """5 realistic patterns — not just 'return self.x'."""
    a1 = random.choice(attrs)
    a2 = random.choice(attrs)
    a3 = random.choice(attrs)
    pattern = random.choice(["compute","validate","update","pipeline","conditional"])

    if pattern == "compute":
        return (f"    result = self.{a1}\n"
                f"    if result is None:\n"
                f"        result = self.{a2} or 0\n"
                f"    self.{a3} = result\n"
                f"    return result")
    elif pattern == "validate":
        return (f"    if self.{a1} is None:\n"
                f"        raise ValueError('{a1} not set')\n"
                f"    if not self.{a2}:\n"
                f"        self.{a2} = self.{a3}\n"
                f"    return bool(self.{a1})")
    elif pattern == "update":
        return (f"    old = self.{a1}\n"
                f"    self.{a1} = data\n"
                f"    self.{a2} = True\n"
                f"    self.{a3} = old\n"
                f"    return old")
    elif pattern == "pipeline":
        return (f"    step1 = self.{a1}\n"
                f"    step2 = step1 if step1 else self.{a2}\n"
                f"    step3 = str(step2) if step2 else ''\n"
                f"    self.{a3} = step3\n"
                f"    return step3")
    else:
        return (f"    if self.{a1} and self.{a2}:\n"
                f"        return self.{a1}\n"
                f"    elif self.{a2}:\n"
                f"        return self.{a2}\n"
                f"    return self.{a3}")


# ══════════════════════════════════════════════════════════════
# GENERATOR 1 — LONG METHOD (5 hard patterns)
# ══════════════════════════════════════════════════════════════
def generate_long_method(n):
    snippets = []
    patterns = ["multi_step","nested_cond","long_loop","exception_chain","state_machine"]
    per_pat  = n // len(patterns)

    for pat_idx, pattern in enumerate(patterns):
        for i in range(per_pat):
            idx  = pat_idx * per_pat + i
            name = f"{random.choice(FUNC_NAMES)}_{pattern}_{idx}"
            v1   = random.choice(VAR_NAMES)
            v2   = random.choice(VAR_NAMES)
            v3   = random.choice(VAR_NAMES)

            if pattern == "multi_step":
                lines = [
                    f"def {name}(self, data, config):",
                    f"    if data is None:",
                    f"        raise ValueError('data cannot be None')",
                    f"    if not isinstance(data, dict):",
                    f"        raise TypeError('expected dict')",
                    f"    if 'id' not in data:",
                    f"        raise KeyError('id required')",
                    f"    {v1} = data.get('value', 0)",
                    f"    if {v1} < 0:",
                    f"        {v1} = 0",
                    f"    {v2} = self._transform({v1})",
                    f"    {v2} = {v2} * config.get('multiplier', 1)",
                    f"    if config.get('normalize'):",
                    f"        {v2} = {v2} / max(abs({v2}), 1)",
                    f"    if config.get('clip'):",
                    f"        {v2} = min(max({v2}, config['min']), config['max'])",
                    f"    self.db.save({{'{v1}': {v1}, '{v2}': {v2}}})",
                    f"    self.cache.invalidate(data['id'])",
                    f"    self.logger.info(f'saved {{data[\"id\"]}}')",
                    f"    {v3} = self.notifier.send(data['id'], {v2})",
                    f"    if not {v3}:",
                    f"        self.logger.warning('notification failed')",
                    f"    return {v2}",
                ]

            elif pattern == "nested_cond":
                lines = [
                    f"def {name}(self, x, y, z):",
                    f"    {v1} = 0",
                    f"    if x.get('active'):",
                    f"        if y > 0:",
                    f"            {v1} = x.get('score', 0) * y",
                    f"            if {v1} > 100:",
                    f"                {v1} = 100",
                    f"                self.logger.warning('score capped')",
                    f"            elif {v1} < 0:",
                    f"                {v1} = 0",
                    f"        else:",
                    f"            {v1} = 0",
                    f"    elif x.get('pending'):",
                    f"        {v2} = self._lookup(y)",
                    f"        if {v2}:",
                    f"            {v1} = {v2}.get('score', 0)",
                    f"        else:",
                    f"            {v1} = self.default_score",
                    f"    elif y > z:",
                    f"        {v1} = sum(k * z for k in range(y))",
                    f"    else:",
                    f"        {v1} = self._fallback(x, y, z)",
                    f"    self.state = {v1}",
                    f"    return {v1}",
                ]

            elif pattern == "long_loop":
                lines = [
                    f"def {name}(self, items, config):",
                    f"    {v1} = []",
                    f"    {v2} = 0",
                    f"    errors = []",
                    f"    for idx, item in enumerate(items):",
                    f"        if item is None:",
                    f"            errors.append(f'null at {{idx}}')",
                    f"            continue",
                    f"        if not self.validator.check(item):",
                    f"            errors.append(f'invalid at {{idx}}')",
                    f"            continue",
                    f"        {v3} = self._process(item, config)",
                    f"        if {v3} is None:",
                    f"            continue",
                    f"        if config.get('transform'):",
                    f"            {v3} = self._transform({v3})",
                    f"        if {v3}.get('weight', 0) > 0:",
                    f"            {v2} += {v3}['weight']",
                    f"        {v1}.append({v3})",
                    f"    if errors:",
                    f"        self.logger.warning(f'{{len(errors)}} errors occurred')",
                    f"    return {v1}, {v2}",
                ]

            elif pattern == "exception_chain":
                lines = [
                    f"def {name}(self, source, target, options=None):",
                    f"    options = options or {{}}",
                    f"    {v1} = None",
                    f"    try:",
                    f"        self.conn.open(source)",
                    f"        {v1} = self.conn.read()",
                    f"        if {v1} is None:",
                    f"            raise ValueError('empty response')",
                    f"        {v1} = self.parser.parse({v1})",
                    f"        if not self.schema.validate({v1}):",
                    f"            raise ValueError('schema mismatch')",
                    f"    except ConnectionError as e:",
                    f"        self.logger.error(f'connection error: {{e}}')",
                    f"        if options.get('retry'):",
                    f"            return self.{name[:10]}(source, target, options)",
                    f"        raise",
                    f"    except ValueError as e:",
                    f"        self.logger.warning(f'value error: {{e}}')",
                    f"        {v1} = self.default_value",
                    f"    except Exception as e:",
                    f"        self.logger.critical(f'unexpected error: {{e}}')",
                    f"        raise RuntimeError('fatal error') from e",
                    f"    finally:",
                    f"        self.conn.close()",
                    f"    self.db.write(target, {v1})",
                    f"    return {v1}",
                ]

            else:  # state_machine
                lines = [
                    f"def {name}(self, event, payload=None):",
                    f"    current = self.state",
                    f"    if current == 'idle':",
                    f"        if event == 'start':",
                    f"            self.state = 'running'",
                    f"            self._on_start(payload)",
                    f"        elif event == 'reset':",
                    f"            self._on_reset()",
                    f"        else:",
                    f"            raise ValueError(f'invalid event {{event}} in idle')",
                    f"    elif current == 'running':",
                    f"        if event == 'pause':",
                    f"            self.state = 'paused'",
                    f"            self._on_pause()",
                    f"        elif event == 'stop':",
                    f"            self.state = 'stopping'",
                    f"            self._on_stop(payload)",
                    f"        elif event == 'error':",
                    f"            self.state = 'idle'",
                    f"            self.logger.error('runtime error')",
                    f"        else:",
                    f"            raise ValueError(f'invalid event {{event}} in running')",
                    f"    elif current == 'paused':",
                    f"        if event == 'resume':",
                    f"            self.state = 'running'",
                    f"        elif event == 'stop':",
                    f"            self.state = 'stopped'",
                    f"        else:",
                    f"            raise ValueError('invalid event in paused')",
                    f"    return self.state",
                ]

            snippets.append(LabeledSnippet(
                code="\n".join(lines), label=0, label_name="long_method"
            ))

    return snippets


# ══════════════════════════════════════════════════════════════
# GENERATOR 2 — LARGE PARAM LIST (6 hard patterns)
# ══════════════════════════════════════════════════════════════
def generate_large_param_list(n):
    snippets = []

    API_PARAMS  = ["request","response","user_id","session_token","db_conn",
                   "cache_client","logger","config","timeout","retry_count",
                   "auth_token","trace_id","correlation_id"]
    DB_PARAMS   = ["host","port","username","password","database",
                   "schema","timeout","pool_size","charset","ssl_mode",
                   "max_overflow","connect_timeout","application_name"]
    REPORT_SETS = [
        ["start_date","end_date","user_id","group_by","output_format",
         "include_totals","timezone","locale"],
        ["source_path","dest_path","encoding","overwrite",
         "create_dirs","dry_run","verbose","checksum"],
        ["title","content","author","category","tags",
         "published","draft","slug","thumbnail","priority"],
        ["width","height","depth","color","material",
         "weight","quantity","sku","barcode","warehouse_id"],
    ]

    for i in range(n):
        pattern = i % 6
        fname   = f"{random.choice(FUNC_NAMES)}_params_{i}"
        nparams = random.randint(6, 10)

        if pattern == 0:
            # Cryptic single-letter — hardest to understand
            params = [f"p{j}" for j in range(nparams)]

        elif pattern == 1:
            # Mixed typed with defaults
            attrs  = random.sample(ATTR_NAMES, nparams)
            params = [
                f"{a}: {random.choice(TYPES)} = {random.choice(DEFAULTS)}"
                for a in attrs
            ]

        elif pattern == 2:
            # API handler style
            params = random.sample(API_PARAMS, min(nparams, len(API_PARAMS)))

        elif pattern == 3:
            # Database connection style
            params = random.sample(DB_PARAMS, min(nparams, len(DB_PARAMS)))

        elif pattern == 4:
            # Exactly 6 params — boundary edge case
            params = [f"{random.choice(ATTR_NAMES)}_{j}=None" for j in range(6)]

        else:
            # Report / export style
            base   = random.choice(REPORT_SETS)
            params = list(base[:nparams])
            while len(params) < nparams:
                params.append(f"extra_{len(params)}=None")

        params_str  = ", ".join(params)
        first_param = params[0].split(":")[0].split("=")[0].strip()
        code = (f"def {fname}({params_str}):\n"
                f"    result = {first_param}\n"
                f"    return result")
        snippets.append(LabeledSnippet(
            code=code, label=1, label_name="large_param_list"
        ))

    return snippets


# ══════════════════════════════════════════════════════════════
# GENERATOR 3 — GOD CLASS (6 distinct real-world types)
# ══════════════════════════════════════════════════════════════
def generate_god_class(n):
    snippets  = []
    per_type  = n // 6
    SUBSYSTEMS = ["db","cache","logger","notifier","validator",
                  "auth","scheduler","reporter","mailer","indexer"]
    BASE_CLASSES = ["BaseHandler","BaseService","BaseManager",
                    "BaseProcessor","BaseController","object"]

    def build_class(class_name, attrs, methods_list, base=None):
        """Assemble a full class string from parts."""
        init_lines = "\n        ".join([rand_attr_init(a) for a in attrs])
        if base and base != "object":
            init_block = (f"def __init__(self):\n"
                          f"        super().__init__()\n"
                          f"        {init_lines}")
        else:
            init_block = f"def __init__(self):\n        {init_lines}"

        blocks = [init_block] + methods_list
        methods_str = "\n\n    ".join(blocks)
        header = f"class {class_name}({base}):" if base else f"class {class_name}:"
        return f"{header}\n\n    {methods_str}"

    # ── Type 1: The BLOB ──────────────────────────────────────
    # Stores everything: auth state, db, cache, session, config in one class
    for i in range(per_type):
        attrs   = random.sample(ATTR_NAMES, random.randint(12, 18))
        domains = random.sample(list(DOMAIN_GROUPS.keys()), 3)
        methods = []
        for d in domains:
            methods += random.sample(DOMAIN_GROUPS[d], 2)

        method_blocks = []
        for m in methods:
            body = make_method_body(attrs)
            method_blocks.append(f"def {m}_{i}(self, data=None):\n{body}")

        # Add a @property — real god classes always have these
        prop_attr = random.choice(attrs)
        method_blocks.append(
            f"@property\n    def {prop_attr}_value(self):\n"
            f"        return self.{prop_attr}"
        )

        code = build_class(f"BlobClass_{i}", attrs, method_blocks)
        snippets.append(LabeledSnippet(code=code, label=2, label_name="god_class"))

    # ── Type 2: The SWISS ARMY ────────────────────────────────
    # Does every operation itself — 4 unrelated domains in one class
    for i in range(per_type):
        attrs   = random.sample(ATTR_NAMES, random.randint(9, 14))
        domains = random.sample(list(DOMAIN_GROUPS.keys()), 4)
        methods = []
        for d in domains:
            methods += random.sample(DOMAIN_GROUPS[d], random.randint(2, 3))

        method_blocks = []
        for m in methods:
            body = make_method_body(attrs)
            method_blocks.append(f"def {m}_{i}(self, *args, **kwargs):\n{body}")

        code = build_class(f"SwissArmy_{i}", attrs, method_blocks)
        snippets.append(LabeledSnippet(code=code, label=2, label_name="god_class"))

    # ── Type 3: The LEGACY GROWER ─────────────────────────────
    # Clean 3-method core, then 6+ unrelated methods bolted on over time
    for i in range(per_type):
        attrs        = random.sample(ATTR_NAMES, random.randint(9, 13))
        core_domain  = random.choice(list(DOMAIN_GROUPS.keys()))
        core_methods = DOMAIN_GROUPS[core_domain][:3]
        other_domains = [d for d in DOMAIN_GROUPS if d != core_domain]
        bolt_domains  = random.sample(other_domains, 3)
        bolt_methods  = []
        for d in bolt_domains:
            bolt_methods += random.sample(DOMAIN_GROUPS[d], 2)

        method_blocks = []
        for m in core_methods + bolt_methods:
            body = make_method_body(attrs)
            method_blocks.append(f"def {m}_{i}(self, data=None):\n{body}")

        code = (f"class Legacy{random.choice(CLASS_NAMES)}_{i}:\n"
                f"    # Core: {core_domain} — other concerns added over time\n\n"
                f"    def __init__(self):\n"
                f"        " + "\n        ".join([rand_attr_init(a) for a in attrs])
                + "\n\n    " + "\n\n    ".join(method_blocks))
        snippets.append(LabeledSnippet(code=code, label=2, label_name="god_class"))

    # ── Type 4: The UTILITY DUMP ──────────────────────────────
    # Mix of @staticmethod, @classmethod, instance methods — a dumping ground
    for i in range(per_type):
        attrs   = random.sample(ATTR_NAMES, random.randint(9, 12))
        domains = random.sample(list(DOMAIN_GROUPS.keys()), 3)
        methods = []
        for d in domains:
            methods += random.sample(DOMAIN_GROUPS[d], 2)

        method_blocks = []
        for j, m in enumerate(methods):
            if j % 3 == 0:
                v = random.choice(VAR_NAMES)
                method_blocks.append(
                    f"@staticmethod\n    def {m}_{i}(data):\n"
                    f"        {v} = data\n"
                    f"        return {v}"
                )
            elif j % 3 == 1:
                method_blocks.append(
                    f"@classmethod\n    def {m}_{i}(cls, data=None):\n"
                    f"        return cls()"
                )
            else:
                body = make_method_body(attrs)
                method_blocks.append(f"def {m}_{i}(self, data=None):\n{body}")

        code = build_class(f"UtilDump_{i}", attrs, method_blocks)
        snippets.append(LabeledSnippet(code=code, label=2, label_name="god_class"))

    # ── Type 5: The DEEP COORDINATOR ─────────────────────────
    # Every single method touches 4-5 subsystems — maximum coupling
    for i in range(per_type):
        chosen     = random.sample(SUBSYSTEMS, random.randint(5, 7))
        extra_attrs = random.sample(ATTR_NAMES, random.randint(4, 8))
        all_attrs   = chosen + extra_attrs
        domains     = random.sample(list(DOMAIN_GROUPS.keys()), 3)
        methods     = []
        for d in domains:
            methods += random.sample(DOMAIN_GROUPS[d], 2)

        method_blocks = []
        for m in methods:
            touched = random.sample(chosen, min(4, len(chosen)))
            v       = random.choice(VAR_NAMES)
            body_lines = [f"    {v} = None"]
            for sys_name in touched:
                action = random.choice(["process","check","notify","log","update"])
                body_lines.append(f"    self.{sys_name}.{action}({v})")
            body_lines.append(f"    return {v}")
            body = "\n".join(body_lines)
            method_blocks.append(f"def {m}_{i}(self, data=None):\n{body}")

        code = build_class(f"DeepCoord_{i}", all_attrs, method_blocks)
        snippets.append(LabeledSnippet(code=code, label=2, label_name="god_class"))

    # ── Type 6: The INHERITING GOD CLASS ─────────────────────
    # Inherits from a base + piles more unrelated methods on top
    for i in range(per_type):
        attrs   = random.sample(ATTR_NAMES, random.randint(9, 13))
        domains = random.sample(list(DOMAIN_GROUPS.keys()), 3)
        methods = []
        for d in domains:
            methods += random.sample(DOMAIN_GROUPS[d], 2)

        method_blocks = []
        for m in methods:
            body = make_method_body(attrs)
            method_blocks.append(f"def {m}_{i}(self, data=None):\n{body}")

        base = random.choice(BASE_CLASSES)
        code = build_class(f"GrowingClass_{i}", attrs, method_blocks, base=base)
        snippets.append(LabeledSnippet(code=code, label=2, label_name="god_class"))

    return snippets


# ══════════════════════════════════════════════════════════════
# GENERATOR 4 — FEATURE ENVY (5 hard patterns)
# ══════════════════════════════════════════════════════════════
def generate_feature_envy(n):
    snippets = []

    for i in range(n):
        pattern  = i % 5
        fname    = f"{random.choice(FUNC_NAMES)}_envy_{i}"
        obj      = random.choice(OBJ_NAMES)
        ext_attrs = random.sample(ATTR_NAMES, 8)

        if pattern == 0:
            # Sum external fields — clearly belongs on the other object
            lines = [f"def {fname}(self, {obj}):"]
            lines.append(f"    total = 0")
            for a in ext_attrs[:5]:
                lines.append(f"    total += {obj}.{a} or 0")
            lines.append(f"    return total")

        elif pattern == 1:
            # Validate external object's state — should be on that object
            lines = [f"def {fname}(self, {obj}):"]
            for a in ext_attrs[:4]:
                lines.append(f"    if {obj}.{a} is None:")
                lines.append(f"        return False")
            lines.append(f"    if {obj}.{ext_attrs[0]} < 0:")
            lines.append(f"        return False")
            lines.append(f"    return True")

        elif pattern == 2:
            # Format/display external object's fields
            lines = [f"def {fname}(self, {obj}):"]
            lines.append(f"    parts = []")
            for a in ext_attrs[:5]:
                lines.append(f"    if {obj}.{a}:")
                lines.append(f"        parts.append(str({obj}.{a}))")
            lines.append(f"    return ', '.join(parts)")

        elif pattern == 3:
            # Compute price/discount using another object's fields
            lines = [
                f"def {fname}(self, {obj}, rate=0.1):",
                f"    base  = {obj}.{ext_attrs[0]} or 0",
                f"    tax   = {obj}.{ext_attrs[1]} or 0",
                f"    fee   = {obj}.{ext_attrs[2]} or 0",
                f"    disc  = {obj}.{ext_attrs[3]} or 0",
                f"    sub   = (base + tax + fee) * (1 - disc * rate)",
                f"    if {obj}.{ext_attrs[4]}:",
                f"        sub *= 0.95",
                f"    if {obj}.{ext_attrs[5]} and {obj}.{ext_attrs[5]} > 0:",
                f"        sub -= {obj}.{ext_attrs[5]}",
                f"    return round(sub, 2)",
            ]

        else:
            # Update many fields on external object — should be a method on it
            lines = [f"def {fname}(self, {obj}, updates):"]
            for a in ext_attrs[:5]:
                lines.append(f"    if '{a}' in updates:")
                lines.append(f"        {obj}.{a} = updates['{a}']")
            lines.append(f"    {obj}.updated = True")
            lines.append(f"    return {obj}")

        snippets.append(LabeledSnippet(
            code="\n".join(lines), label=3, label_name="feature_envy"
        ))

    return snippets


# ══════════════════════════════════════════════════════════════
# GENERATOR 5 — CLEAN CODE HARD NEGATIVES
# Things that LOOK like smells but genuinely are NOT
# ══════════════════════════════════════════════════════════════
def generate_clean_hard_negatives(n):
    snippets = []
    per_type = n // 7

    # ── A: Django-style Model — many class attrs, only 2-3 methods ──
    # Many fields but NOT a god_class (method count is low)
    for i in range(per_type):
        num_fields = random.randint(10, 18)
        attrs      = random.sample(ATTR_NAMES, min(num_fields, len(ATTR_NAMES)))
        field_lines = "\n    ".join([
            f"{a} = models.{random.choice(FIELD_TYPES)}" for a in attrs
        ])
        model_name = random.choice(
            ["UserProfile","Product","Order","Article","Event","Invoice","Transaction"]
        )
        code = (f"class {model_name}Model_{i}:\n"
                f"    {field_lines}\n\n"
                f"    def __str__(self):\n"
                f"        return str(self.{attrs[0]})\n\n"
                f"    def save(self, *args, **kwargs):\n"
                f"        super().save(*args, **kwargs)")
        snippets.append(LabeledSnippet(code=code, label=4, label_name="clean_code"))

    # ── B: Dataclass with many fields, zero methods ──────────────
    # Pure DTO — not a god_class
    for i in range(per_type):
        num_fields = random.randint(8, 15)
        attrs      = random.sample(ATTR_NAMES, min(num_fields, len(ATTR_NAMES)))
        field_lines = "\n    ".join([
            f"{a}: {random.choice(['int','str','float','bool','Optional[str]'])} = "
            f"{random.choice(DEFAULTS)}"
            for a in attrs
        ])
        code = f"@dataclass\nclass Config_{i}:\n    {field_lines}"
        snippets.append(LabeledSnippet(code=code, label=4, label_name="clean_code"))

    # ── C: Abstract Base Class — many abstract methods, NO attributes ──
    # Many methods but no attributes = not a god_class
    for i in range(per_type):
        num_methods = random.randint(8, 14)
        methods     = random.sample(METHOD_NAMES, min(num_methods, len(METHOD_NAMES)))
        methods_str = "\n\n    ".join([
            f"@abstractmethod\n    def {m}(self, *args, **kwargs):\n        ..."
            for m in methods
        ])
        code = (f"from abc import ABC, abstractmethod\n\n"
                f"class Base{random.choice(CLASS_NAMES)}_{i}(ABC):\n\n"
                f"    {methods_str}")
        snippets.append(LabeledSnippet(code=code, label=4, label_name="clean_code"))

    # ── D: Adapter / delegation — intentional external calls ─────
    # Looks like feature_envy but it's NOT — delegation is the design
    for i in range(per_type):
        backend  = random.choice(["client","backend","driver","store","engine"])
        methods  = random.sample(METHOD_NAMES[:15], 4)
        method_str = "\n\n    ".join([
            f"def {m}(self, *args, **kwargs):\n"
            f"        return self.{backend}.{m}(*args, **kwargs)"
            for m in methods
        ])
        code = (f"class {random.choice(CLASS_NAMES)}Adapter_{i}:\n"
                f"    def __init__(self, {backend}):\n"
                f"        self.{backend} = {backend}\n\n"
                f"    {method_str}")
        snippets.append(LabeledSnippet(code=code, label=4, label_name="clean_code"))

    # ── E: Method using self.logger / self.cache heavily ─────────
    # Looks like feature_envy but self-references dominate
    for i in range(per_type):
        v1    = random.choice(VAR_NAMES)
        fname = f"{random.choice(FUNC_NAMES)}_clean_{i}"
        code  = (f"def {fname}(self, data):\n"
                 f"    self.logger.info('starting {fname}')\n"
                 f"    if not self.config.get('enabled'):\n"
                 f"        self.logger.warning('disabled')\n"
                 f"        return None\n"
                 f"    {v1} = self._prepare(data)\n"
                 f"    self.cache.set('last', {v1})\n"
                 f"    self.metrics.increment('calls')\n"
                 f"    self.logger.info('done')\n"
                 f"    return {v1}")
        snippets.append(LabeledSnippet(code=code, label=4, label_name="clean_code"))

    # ── F: 13-19 line function — just UNDER long_method threshold ──
    for i in range(per_type):
        v1       = random.choice(VAR_NAMES)
        v2       = random.choice(VAR_NAMES)
        n_extra  = random.randint(2, 5)
        extras   = "\n    ".join([f"{v2}_{j} = {v1} + {j}" for j in range(n_extra)])
        fname    = f"{random.choice(FUNC_NAMES)}_short_{i}"
        code     = (f"def {fname}(self, data):\n"
                    f"    if data is None:\n"
                    f"        return None\n"
                    f"    {v1} = data.get('value', 0)\n"
                    f"    {extras}\n"
                    f"    return {v2}_0")
        snippets.append(LabeledSnippet(code=code, label=4, label_name="clean_code"))

    # ── G: Function with 4-5 params — UNDER large_param_list threshold ──
    for i in range(per_type):
        num_p  = random.randint(4, 5)
        params = random.sample(ATTR_NAMES, num_p)
        body   = ", ".join([f"{p}={p}" for p in params])
        code   = (f"def {random.choice(FUNC_NAMES)}_few_{i}"
                  f"({', '.join(params)}):\n"
                  f"    return dict({body})")
        snippets.append(LabeledSnippet(code=code, label=4, label_name="clean_code"))

    return snippets


# ══════════════════════════════════════════════════════════════
# GENERATOR 6 — EDGE CASES (boundary examples for every class)
# ══════════════════════════════════════════════════════════════
def generate_edge_cases():
    snippets = []

    # Edge: exactly 21-line function (just over long_method threshold)
    for i in range(600):
        v1 = random.choice(VAR_NAMES)
        v2 = random.choice(VAR_NAMES)
        lines = [
            f"def edge_long_{i}(self, x, y):",
            f"    if x is None: raise ValueError('x required')",
            f"    if y is None: raise ValueError('y required')",
            f"    {v1} = self._step1(x)",
            f"    {v2} = self._step2(y)",
            f"    if {v1} > 0:",
            f"        {v1} *= 2",
            f"    if {v2} > 0:",
            f"        {v2} *= 2",
            f"    a = {v1} + {v2}",
            f"    b = {v1} - {v2}",
            f"    c = {v1} * {v2}",
            f"    if a > b:",
            f"        result = a",
            f"    elif b > c:",
            f"        result = b",
            f"    else:",
            f"        result = c",
            f"    self.logger.info(f'result={{result}}')",
            f"    self.cache.set('last', result)",
            f"    return result",
        ]
        snippets.append(LabeledSnippet(
            code="\n".join(lines), label=0, label_name="long_method"
        ))

    # Edge: exactly 6 params (boundary of large_param_list)
    for i in range(600):
        params     = random.sample(ATTR_NAMES, 6)
        params_str = ", ".join(params)
        code       = f"def edge_params_{i}({params_str}):\n    return {params[0]}"
        snippets.append(LabeledSnippet(
            code=code, label=1, label_name="large_param_list"
        ))

    # Edge: god class at minimum threshold (7 methods + 9 attrs)
    for i in range(600):
        attrs   = random.sample(ATTR_NAMES, 9)
        domains = random.sample(list(DOMAIN_GROUPS.keys()), 2)
        methods = []
        for d in domains:
            methods += random.sample(DOMAIN_GROUPS[d], 3)
        methods = methods[:6]  # 6 non-init + __init__ = 7 total

        init_lines = "\n        ".join([rand_attr_init(a) for a in attrs])
        blocks = [f"def __init__(self):\n        {init_lines}"]
        for m in methods:
            body = make_method_body(attrs)
            blocks.append(f"def {m}_{i}(self, data=None):\n{body}")

        methods_str = "\n\n    ".join(blocks)
        code = f"class EdgeGod_{i}:\n\n    {methods_str}"
        snippets.append(LabeledSnippet(code=code, label=2, label_name="god_class"))

    # Edge: feature envy just at threshold (4 external, 1 self reference)
    for i in range(600):
        obj       = random.choice(OBJ_NAMES)
        ext_attrs = random.sample(ATTR_NAMES, 4)
        lines     = [
            f"def edge_envy_{i}(self, {obj}):",
            f"    a = {obj}.{ext_attrs[0]}",
            f"    b = {obj}.{ext_attrs[1]}",
            f"    c = {obj}.{ext_attrs[2]}",
            f"    d = {obj}.{ext_attrs[3]}",
            f"    flag = self.enabled",
            f"    return (a + b + c + d) if flag else 0",
        ]
        snippets.append(LabeledSnippet(
            code="\n".join(lines), label=3, label_name="feature_envy"
        ))

    return snippets


# ══════════════════════════════════════════════════════════════
# BUILD THE FULL DATASET
# ══════════════════════════════════════════════════════════════
TARGET_PER_CLASS = 12_000

print("=" * 60)
print("  Generating synthetic training data...")
print("=" * 60)

synth_long_method  = generate_long_method(8000)
synth_large_param  = generate_large_param_list(10000)
synth_god_class    = generate_god_class(11400)   # 6 types × 1900 each
synth_feature_envy = generate_feature_envy(9000)
hard_negatives     = generate_clean_hard_negatives(5600)
edge_cases         = generate_edge_cases()

print(f"\n  Synthetic counts:")
print(f"    long_method:      {len(synth_long_method)}")
print(f"    large_param_list: {len(synth_large_param)}")
print(f"    god_class:        {len(synth_god_class)}")
print(f"    feature_envy:     {len(synth_feature_envy)}")
print(f"    hard negatives:   {len(hard_negatives)}")
print(f"    edge cases:       {len(edge_cases)}")

# ── Combine real (from Cell 4) + synthetic ────────────────────
all_snippets = (
      labeled_snippets       # ← real data from Cell 4
    + synth_long_method
    + synth_large_param
    + synth_god_class
    + synth_feature_envy
    + hard_negatives
    + edge_cases
)

print(f"\n  Total before balancing: {len(all_snippets)}")
raw_counts = Counter(s.label_name for s in all_snippets)
print("  Raw counts per class:")
for name in LABEL_NAMES:
    print(f"    {name}: {raw_counts.get(name, 0)}")


# ══════════════════════════════════════════════════════════════
# BALANCE TO TARGET_PER_CLASS
# ══════════════════════════════════════════════════════════════
def balance_dataset(snippets, target):
    grouped = {}
    for s in snippets:
        grouped.setdefault(s.label_name, []).append(s)

    balanced = []
    for name in LABEL_NAMES:
        items = grouped.get(name, [])
        random.shuffle(items)
        if len(items) >= target:
            balanced.extend(items[:target])                       # undersample
        else:
            extra = random.choices(items, k=target - len(items)) # oversample
            balanced.extend(items + extra)

    random.shuffle(balanced)
    return balanced

balanced_snippets = balance_dataset(all_snippets, TARGET_PER_CLASS)

print(f"\n{'=' * 60}")
print(f"  FINAL BALANCED DATASET")
print(f"{'=' * 60}")
print(f"  Total snippets : {len(balanced_snippets)}")
print(f"  Target/class   : {TARGET_PER_CLASS}")
final_counts = Counter(s.label_name for s in balanced_snippets)
for name in LABEL_NAMES:
    count = final_counts.get(name, 0)
    bar   = "█" * (count // 400)
    print(f"  {name:<20} {count:>6}  {bar}")


# ══════════════════════════════════════════════════════════════
# TRAIN / TEST SPLIT  (80 / 20)
# ══════════════════════════════════════════════════════════════
all_indices = list(range(len(balanced_snippets)))
random.shuffle(all_indices)
split       = int(0.8 * len(all_indices))
train_idx   = all_indices[:split]
test_idx    = all_indices[split:]

train_snippets = [balanced_snippets[i] for i in train_idx]
test_snippets  = [balanced_snippets[i] for i in test_idx]

print(f"\n  Train : {len(train_snippets)}")
print(f"  Test  : {len(test_snippets)}")
print(f"\n  Cell 5 complete ✓")


# In[31]:


# CELL 6: Build HuggingFace Dataset
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

data_dict = {
    "code": [s.code for s in balanced_snippets],
    "label": [s.label for s in balanced_snippets]
}

labels = [s.label for s in balanced_snippets]

# First split: 80% train, 20% temp
train_idx, temp_idx = train_test_split(
    range(len(balanced_snippets)), 
    test_size=0.2, 
    random_state=42,
    stratify=labels
)

# Second split: split temp into 50/50 → 10% val, 10% test of total
temp_labels = [labels[i] for i in temp_idx]
val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.5,
    random_state=42,
    stratify=temp_labels
)

train_data = {
    "code": [data_dict["code"][i] for i in train_idx],
    "label": [data_dict["label"][i] for i in train_idx]
}
val_data = {
    "code": [data_dict["code"][i] for i in val_idx],
    "label": [data_dict["label"][i] for i in val_idx]
}
test_data = {
    "code": [data_dict["code"][i] for i in test_idx],
    "label": [data_dict["label"][i] for i in test_idx]
}

dataset = DatasetDict({
    "train": Dataset.from_dict(train_data),
    "validation": Dataset.from_dict(val_data),
    "test": Dataset.from_dict(test_data)
})

print(dataset)
print(f"\n✓ Train: {len(dataset['train'])} | Val: {len(dataset['validation'])} | Test: {len(dataset['test'])}")


# In[32]:


# CELL 7: Tokenization
from transformers import AutoTokenizer

MODEL_NAME = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(
        examples["code"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["code"])
tokenized_dataset = tokenized_dataset.with_format("torch")

print("✓ Tokenization complete")
print(tokenized_dataset)


# In[33]:


# CELL 8: Model Setup
from transformers import AutoModelForSequenceClassification

NUM_LABELS = 5

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    problem_type="single_label_classification"
)
model.to(DEVICE)

print(f"✓ Model loaded on {DEVICE}")
print(f"  Parameters: {model.num_parameters():,}")
print(f"  Num labels: {NUM_LABELS}")


# In[34]:


# CELL 9: Training Configuration
import shutil
import os
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np

# Clear old checkpoints to keep things clean
if os.path.exists("./codesmell_model"):
    shutil.rmtree("./codesmell_model")
    print("✓ Cleared old checkpoints")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy         = accuracy_score(labels, predictions)
    macro_f1         = f1_score(labels, predictions, average="macro")
    weighted_f1      = f1_score(labels, predictions, average="weighted")
    macro_precision  = precision_score(labels, predictions, average="macro", zero_division=0)
    macro_recall     = recall_score(labels, predictions, average="macro", zero_division=0)
    per_class_f1     = f1_score(labels, predictions, average=None, zero_division=0)

    metrics = {
        "accuracy":        accuracy,
        "macro_f1":        macro_f1,
        "weighted_f1":     weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall":    macro_recall,
    }
    for i, name in enumerate(LABEL_NAMES):
        metrics[f"{name}_f1"] = per_class_f1[i]

    return metrics

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir                  = "./codesmell_model",
    num_train_epochs            = 5,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size  = 8,
    gradient_accumulation_steps = 2,
    warmup_steps                = 500,          # ← changed (3x more data now)
    learning_rate               = 2e-5,
    weight_decay                = 0.01,
    logging_steps               = 100,
    eval_strategy               = "epoch",
    save_strategy               = "epoch",
    load_best_model_at_end      = True,
    metric_for_best_model       = "macro_f1",   # ← changed (was accuracy)
    greater_is_better           = True,
    fp16                        = torch.cuda.is_available(),
    report_to                   = "none",
    dataloader_num_workers      = 0,
    save_total_limit            = 2,
)

trainer = Trainer(
    model         = model,
    args          = training_args,
    train_dataset = tokenized_dataset["train"],
    eval_dataset  = tokenized_dataset["test"],
    compute_metrics = compute_metrics,
    data_collator = data_collator,
)

print("=" * 60)
print("TRAINING CONFIGURATION")
print("=" * 60)
print(f"  Training samples : {len(tokenized_dataset['train'])}")
print(f"  Eval samples     : {len(tokenized_dataset['test'])}")
print(f"  Epochs           : 5")
print(f"  Batch size       : 8  (effective: 16)")
print(f"  Warmup steps     : 500")
print(f"  Learning rate    : 2e-5")
print(f"  Best model by    : macro_f1  ← key change")
print("=" * 60)


# In[35]:


# CELL 10: Training
import time
import os

print("=" * 60)
print("  TRAINING START")
print("=" * 60)
print(f"  Train samples  : {len(tokenized_dataset['train'])}")
print(f"  Eval  samples  : {len(tokenized_dataset['test'])}")
print(f"  Epochs         : 5")
print(f"  Effective batch: 16  (8 x 2 accumulation)")
total_steps = (len(tokenized_dataset['train']) // 16) * 5
print(f"  Total steps    : ~{total_steps:,}")
print(f"  Best model by  : macro_f1")
print("=" * 60)

# -- Train -------------------------------------------------------
start_time = time.time()

train_result = trainer.train()

elapsed  = time.time() - start_time
hours    = int(elapsed // 3600)
minutes  = int((elapsed % 3600) // 60)
seconds  = int(elapsed % 60)

print(f"\nTraining complete in {hours}h {minutes}m {seconds}s")

# -- Save best model to a clean known location ------------------
BEST_MODEL_PATH = "./best_codesmell_model"
trainer.save_model(BEST_MODEL_PATH)
tokenizer.save_pretrained(BEST_MODEL_PATH)
print(f"Best model saved -> {BEST_MODEL_PATH}")

# -- Print training summary -------------------------------------
print("\n" + "=" * 60)
print("  TRAINING SUMMARY")
print("=" * 60)
metrics = train_result.metrics
print(f"  Train loss     : {metrics.get('train_loss', 'N/A'):.4f}")
print(f"  Samples/sec    : {metrics.get('train_samples_per_second', 'N/A'):.1f}")
print(f"  Time taken     : {hours}h {minutes}m {seconds}s")

# -- Final evaluation on test set -------------------------------
print("\n" + "=" * 60)
print("  FINAL EVALUATION (best model)")
print("=" * 60)
final_metrics = trainer.evaluate()

print(f"\n  Overall:")
print(f"    accuracy     : {final_metrics.get('eval_accuracy',    0):.4f}")
print(f"    macro_f1     : {final_metrics.get('eval_macro_f1',    0):.4f}")
print(f"    weighted_f1  : {final_metrics.get('eval_weighted_f1', 0):.4f}")

print(f"\n  Per-class F1:")
for name in LABEL_NAMES:
    f1  = final_metrics.get(f'eval_{name}_f1', 0)
    bar = "#" * int(f1 * 20)
    gap = "-" * (20 - int(f1 * 20))
    print(f"    {name:<20} {f1:.3f}  [{bar}{gap}]")

# -- Sanity check -----------------------------------------------
print("\n" + "=" * 60)
macro = final_metrics.get('eval_macro_f1',    0)
god   = final_metrics.get('eval_god_class_f1', 0)

if macro >= 0.85 and god >= 0.68:
    print("  RESULT: TARGET MET — model is ready")
elif macro >= 0.80 and god >= 0.55:
    print("  RESULT: DECENT — consider 1-2 more epochs or lower LR")
else:
    print("  RESULT: BELOW TARGET — check Cell 4b output (real god classes)")
print("=" * 60)


# In[36]:


# CELL 11: Save Model
import json

MODEL_SAVE_PATH = "./codesmell_model_final"
trainer.save_model(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)

label_config = {
    "label_names": LABEL_NAMES,
    "label_map": LABEL_MAP,
    "num_labels": NUM_LABELS
}
with open(f"{MODEL_SAVE_PATH}/label_config.json", "w") as f:
    json.dump(label_config, f, indent=2)

print(f"✓ Model saved to {MODEL_SAVE_PATH}")


# In[37]:


# CELL 12: Detailed Evaluation
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("Running evaluation...")
predictions = trainer.predict(tokenized_dataset["test"])
logits  = predictions.predictions
preds   = np.argmax(logits, axis=-1)
labels  = predictions.label_ids

accuracy = accuracy_score(labels, preds)
macro_f1 = f1_score(labels, preds, average="macro")

print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)
print(f"  Accuracy : {accuracy * 100:.2f}%")
print(f"  Macro F1 : {macro_f1:.4f}")
print("-" * 60)
print("\nClassification Report:")
print(classification_report(labels, preds, target_names=LABEL_NAMES, digits=3))

cm = confusion_matrix(labels, preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix (Accuracy: {accuracy*100:.1f}%)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("Confusion matrix saved -> confusion_matrix.png")


# In[38]:


# CELL 12b: TRUE EVALUATION — Real Code Only
import re

def is_synthetic(code):
    """
    Detect synthetic snippets by their generated naming patterns.
    UPDATE these patterns to match whatever Cell 5 uses.
    Check Cell 5 output to confirm all synthetic naming conventions are covered.
    """
    code = code.strip()
    patterns = [
        # Cell 5 original synthetic patterns
        r'^def long_function_\d+\(',
        r'^def many_params_\d+\(',
        r'^class GodClass_\d+:',
        r'^def feature_envy_\d+\(',
        r'^def clean_func_\d+\(',
        # Cell 4b/5 improved synthetic god class patterns
        # --- VERIFY THESE MATCH YOUR CELL 5 OUTPUT ---
        r'^class BlobClass',
        r'^class SwissArmy',
        r'^class Legacy',
        r'^class UtilDump',
        r'^class DeepCoord',
        r'^class GodService',
        r'^class Inheriting',
    ]
    return any(re.match(p, code) for p in patterns)

# Recover test snippets — requires test_idx and balanced_snippets in memory from Cell 6
test_snippets = [balanced_snippets[i] for i in test_idx]

real_mask  = np.array([not is_synthetic(s.code) for s in test_snippets])
synth_mask = ~real_mask

print("=" * 60)
print("TEST SET COMPOSITION")
print("=" * 60)
print(f"  Total     : {len(test_snippets)}")
print(f"  Real      : {real_mask.sum()}  ({real_mask.mean()*100:.1f}%)")
print(f"  Synthetic : {synth_mask.sum()} ({synth_mask.mean()*100:.1f}%)")

print("\nReal samples per class in test set:")
real_test_snippets = [s for s, keep in zip(test_snippets, real_mask) if keep]
real_counts = Counter(s.label_name for s in real_test_snippets)
for name in LABEL_NAMES:
    count = real_counts.get(name, 0)
    warn  = "  <- LOW, treat with caution" if count < 50 else ""
    print(f"  {name}: {count}{warn}")

# Filter predictions from Cell 12 — requires preds and labels in memory
real_preds   = preds[real_mask]
real_labels  = labels[real_mask]
synth_preds  = preds[synth_mask]
synth_labels = labels[synth_mask]

real_accuracy  = accuracy_score(real_labels,  real_preds)
real_macro_f1  = f1_score(real_labels,  real_preds,  average="macro", zero_division=0)
synth_accuracy = accuracy_score(synth_labels, synth_preds)
full_accuracy  = accuracy_score(labels, preds)

print("\n" + "=" * 60)
print("ACCURACY COMPARISON")
print("=" * 60)
print(f"  Reported (real + synthetic) : {full_accuracy*100:.2f}%  <- inflated")
print(f"  Synthetic-only              : {synth_accuracy*100:.2f}%  <- model memorized patterns")
print(f"  TRUE (real code only)       : {real_accuracy*100:.2f}%  <- what actually matters")
print(f"  Macro F1 (real only)        : {real_macro_f1:.4f}")

print("\nClassification Report — Real Code Only:")
print(classification_report(real_labels, real_preds,
                             target_names=LABEL_NAMES,
                             digits=3,
                             zero_division=0))

cm_real = confusion_matrix(real_labels, real_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_real, annot=True, fmt="d", cmap="Oranges",
            xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix — Real Code Only\n"
          f"True Accuracy: {real_accuracy*100:.1f}%   vs   Reported: {full_accuracy*100:.1f}%")
plt.tight_layout()
plt.savefig("confusion_matrix_real_only.png", dpi=150)
plt.show()
print("Saved -> confusion_matrix_real_only.png")


# In[40]:


# CELL 13: Inference Functions

def load_smell_detector(model_path=MODEL_SAVE_PATH):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    with open(f"{model_path}/label_config.json") as f:
        config = json.load(f)

    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    return model, tokenizer, config

def predict_smell(code, model, tokenizer, config):
    inputs = tokenizer(
        code,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )

    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        pred_label = int(np.argmax(probs))

    return {
        "predicted_label": pred_label,
        "predicted_smell": config["label_names"][pred_label],
        "confidence": float(probs[pred_label]),
        "all_probabilities": {
            name: float(probs[i]) 
            for i, name in enumerate(config["label_names"])
        }
    }

print("✓ Inference functions ready!")


# In[41]:


# CELL 14: Demo
test_cases = [
    # Long method
    """def very_long_function(x):
    a = x + 1
    b = a + 2
    c = b + 3
    d = c + 4
    e = d + 5
    f = e + 6
    g = f + 7
    h = g + 8
    i = h + 9
    j = i + 10
    k = j + 11
    l = k + 12
    m = l + 13
    n = m + 14
    o = n + 15
    p = o + 16
    q = p + 17
    r = q + 18
    s = r + 19
    t = s + 20
    u = t + 21
    return u""",

    # Large parameter list
    """def many_params(a, b, c, d, e, f, g, h):
    return a + b + c + d + e + f + g + h""",

    # Feature envy
    """def process_order(self, order):
    total = order.get_items()
    tax = order.calculate_tax()
    discount = order.apply_discount()
    shipping = order.get_shipping()
    return total + tax - discount + shipping""",

    # Clean code
    """def add(x, y):
    return x + y"""
]

print("\n" + "=" * 60)
print("DEMO: Testing on sample code")
print("=" * 60)

model, tokenizer, config = load_smell_detector()

for i, code in enumerate(test_cases, 1):
    result = predict_smell(code, model, tokenizer, config)
    print(f"\n--- Test {i} ---")
    print(f"Code preview: {code[:50]}...")
    print(f"Predicted: {result['predicted_smell']} ({result['confidence']:.1%})")


# In[45]:


# CELL 15a: Save label config 
import json

BEST_MODEL_PATH = "./best_codesmell_model"

label_config = {
    "label_names": LABEL_NAMES,
    "id2label":    {str(i): name for i, name in enumerate(LABEL_NAMES)},
    "label2id":    {name: i      for i, name in enumerate(LABEL_NAMES)}
}

with open(f"{BEST_MODEL_PATH}/label_config.json", "w") as f:
    json.dump(label_config, f, indent=2)

print(f"Saved -> {BEST_MODEL_PATH}/label_config.json")
print(f"Classes: {LABEL_NAMES}")


# In[7]:


# CELL 15b: Flask Server with God Class Heuristic Fallback

import threading
import time
import ast
import re
import requests
from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import json

app = Flask(__name__)

MODEL_PATH = "./best_codesmell_model"

print("Loading model...")
try:
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    with open(f"{MODEL_PATH}/label_config.json") as f:
        config = json.load(f)
    LABEL_NAMES = config["label_names"]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(DEVICE)
    model.eval()
    print(f"Model loaded on {DEVICE}")
    print(f"Classes: {LABEL_NAMES}")
except Exception as e:
    print(f"ERROR loading model: {e}")
    raise

# ── Heuristic helpers ────────────────────────────────────────────
GOD_CLASS_METHOD_THRESHOLD = 5
GOD_CLASS_LOC_THRESHOLD    = 15

def count_methods(code: str) -> int:
    try:
        tree = ast.parse(code)
        return sum(
            1 for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
    except:
        return len(re.findall(r'^\s+def ', code, re.MULTILINE))

def count_classes(code: str) -> int:
    try:
        tree = ast.parse(code)
        return sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
    except:
        return len(re.findall(r'^class ', code, re.MULTILINE))

def lines_of_code(code: str) -> int:
    return len([l for l in code.splitlines() if l.strip()])

def apply_heuristics(code: str, predicted: str, probs: np.ndarray) -> tuple:
    """
    Returns (final_label, confidence, was_overridden)
    Currently handles god_class fallback only.
    """
    if predicted == "clean_code":
        n_methods = count_methods(code)
        n_classes = count_classes(code)
        loc       = lines_of_code(code)

        if (n_classes >= 1
                and n_methods >= GOD_CLASS_METHOD_THRESHOLD
                and loc       >= GOD_CLASS_LOC_THRESHOLD):
            idx = LABEL_NAMES.index("god_class")
            return "god_class", round(float(probs[idx]), 4), True

    return predicted, round(float(probs[LABEL_NAMES.index(predicted)]), 4), False

# ── Routes ───────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if not data or not data.get("code", "").strip():
        return jsonify({"error": "No code provided"}), 400

    code   = data["code"]
    inputs = tokenizer(
        code,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(DEVICE)

    with torch.no_grad():
        outputs    = model(**inputs)
        probs      = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        pred_index = int(np.argmax(probs))

    raw_label  = LABEL_NAMES[pred_index]
    final_label, confidence, overridden = apply_heuristics(code, raw_label, probs)

    return jsonify({
        "smell":              final_label,
        "confidence":         confidence,
        "heuristic_override": overridden,
        "raw_model_label":    raw_label,
        "all_scores": {
            name: round(float(probs[i]), 4)
            for i, name in enumerate(LABEL_NAMES)
        }
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":  "ok",
        "device":  DEVICE,
        "classes": LABEL_NAMES
    })

# ── Start server ─────────────────────────────────────────────────
flask_thread = threading.Thread(
    target=lambda: app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False),
    daemon=True
)
flask_thread.start()
print("\nServer starting...")
time.sleep(3)

try:
    r = requests.get("http://localhost:5000/health")
    print("Server is up:", r.json())
except Exception as e:
    print("Not responding:", e)


# In[8]:


# CELL 15c: Smoke test all 5 classes
import requests

tests = {
    "large_param_list": "def process(a, b, c, d, e, f, g):\n    return a + b",
    "long_method": "\n".join(["def do_everything():"] + [f"    x{i} = {i} * 2" for i in range(25)] + ["    return x0"]),
    "god_class": """
class Manager:
    def handle_auth(self, user, password):
        hashed = hashlib.sha256(password.encode()).hexdigest()
        return self.db.query(f"SELECT * FROM users WHERE hash='{hashed}'")

    def send_email(self, to, subject, body):
        smtp = smtplib.SMTP('smtp.gmail.com', 587)
        smtp.sendmail("app@company.com", to, f"Subject:{subject}\n{body}")
        smtp.quit()

    def write_to_db(self, table, data):
        cols = ', '.join(data.keys())
        vals = ', '.join([f"'{v}'" for v in data.values()])
        self.db.execute(f"INSERT INTO {table} ({cols}) VALUES ({vals})")

    def parse_csv(self, filepath):
        rows = []
        with open(filepath) as f:
            for line in f.readlines()[1:]:
                rows.append(line.strip().split(','))
        return rows

    def calculate_tax(self, amount, region):
        rates = {'US': 0.07, 'EU': 0.20, 'UK': 0.15}
        return amount * rates.get(region, 0.10)
""",
    "feature_envy": """
def calculate_discount(order):
    base   = order.customer.account.tier.discount_rate
    region = order.customer.address.region.tax_rate
    return base - region
""",
    "clean_code": "def add(a, b):\n    return a + b"
}

print(f"{'Expected':<20} {'Predicted':<20} {'Confidence':<12} {'Override':<10} {'Pass'}")
print("-" * 75)

all_pass = True
for expected, code in tests.items():
    r      = requests.post("http://localhost:5001/predict", json={"code": code})
    res    = r.json()
    pred   = res["smell"]
    conf   = res["confidence"]
    ov     = "YES" if res["heuristic_override"] else "no"
    status = "YES" if pred == expected else "NO  <---"
    if pred != expected:
        all_pass = False
    print(f"{expected:<20} {pred:<20} {conf:<12} {ov:<10} {status}")

print("-" * 75)
print("All tests passed!" if all_pass else "Some tests failed — check above.")


# In[57]:


# CELL DEBUG: Check what the heuristic sees not for running everytimeeee
import ast
import re

god_class_code = """
class Manager:
    def handle_auth(self, user, password):
        hashed = hashlib.sha256(password.encode()).hexdigest()
        return self.db.query(f"SELECT * FROM users WHERE hash='{hashed}'")

    def send_email(self, to, subject, body):
        smtp = smtplib.SMTP('smtp.gmail.com', 587)
        smtp.sendmail("app@company.com", to, f"Subject:{subject}\\n{body}")
        smtp.quit()

    def write_to_db(self, table, data):
        cols = ', '.join(data.keys())
        vals = ', '.join([f"'{v}'" for v in data.values()])
        self.db.execute(f"INSERT INTO {table} ({cols}) VALUES ({vals})")

    def parse_csv(self, filepath):
        rows = []
        with open(filepath) as f:
            for line in f.readlines()[1:]:
                rows.append(line.strip().split(','))
        return rows

    def calculate_tax(self, amount, region):
        rates = {'US': 0.07, 'EU': 0.20, 'UK': 0.15}
        return amount * rates.get(region, 0.10)
"""

def count_methods(code):
    try:
        tree = ast.parse(code)
        return sum(1 for node in ast.walk(tree)
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))
    except Exception as e:
        print(f"  ast.parse failed: {e}")
        return len(re.findall(r'^\s+def ', code, re.MULTILINE))

def count_classes(code):
    try:
        tree = ast.parse(code)
        return sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
    except:
        return len(re.findall(r'^class ', code, re.MULTILINE))

def lines_of_code(code):
    return len([l for l in code.splitlines() if l.strip()])

n_methods = count_methods(god_class_code)
n_classes = count_classes(god_class_code)
loc       = lines_of_code(god_class_code)

print(f"n_methods : {n_methods}  (threshold >= 5)")
print(f"n_classes : {n_classes}  (threshold >= 1)")
print(f"loc       : {loc}  (threshold >= 30)")
print()
print(f"Would trigger: {n_classes >= 1 and n_methods >= 5 and loc >= 30}")

