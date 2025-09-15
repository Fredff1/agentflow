# agentflow/utils/math_answer.py
from __future__ import annotations
from typing import Optional, List, Dict, Any
import re
import sympy
from sympy.parsing import sympy_parser
from pylatexenc import latex2text


BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = [r"\^[0-9]+\^", r"\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"

def mathd_normalize_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    answer = answer.strip()
    try:
        m = re.search(r"^\\text{(?P<text>.+?)}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return _strip_string(answer)
    except Exception:
        return answer

def _strip_string(string: str) -> str:
    def _fix_fracs(s: str) -> str:
        substrs = s.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += "\\frac"
                if substr and substr[0] == "{":
                    new_str += substr
                else:
                    if len(substr) < 2:
                        return s
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        new_str += "{" + a + "}{" + b + "}" + substr[2:]
                    else:
                        new_str += "{" + a + "}" + b + substr[2:]
        return new_str

    def _fix_a_slash_b(s: str) -> str:
        parts = s.split("/")
        if len(parts) != 2:
            return s
        a, b = parts
        try:
            a = int(a); b = int(b)
            if s == f"{a}/{b}":
                return f"\\frac{{{a}}}{{{b}}}"
            return s
        except Exception:
            return s

    def _remove_right_units(s: str) -> str:
        if "\\text{ " in s:
            splits = s.split("\\text{ ")
            if len(splits) == 2:
                return splits[0]
        return s

    def _fix_sqrt(s: str) -> str:
        if "\\sqrt" not in s:
            return s
        splits = s.split("\\sqrt")
        new_s = splits[0]
        for split in splits[1:]:
            if split and split[0] != "{":
                a = split[0]
                new_substr = "\\sqrt{" + a + "}" + split[1:]
            else:
                new_substr = "\\sqrt" + split
            new_s += new_substr
        return new_s

    s = string.replace("\n", "")
    s = s.replace("\\!", "")
    s = s.replace("\\\\", "\\")
    s = s.replace("tfrac", "frac").replace("dfrac", "frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = s.replace("\\$", "")
    s = _remove_right_units(s)
    s = s.replace("\\%", "").replace("%", "")
    s = s.replace(" .", " 0.").replace("{.", "{0.")
    if s and s[0] == ".":
        s = "0" + s
    if len(s.split("=")) == 2 and len(s.split("=")[0]) <= 2:
        s = s.split("=")[1]
    s = _fix_sqrt(s)
    s = s.replace(" ", "")
    s = _fix_fracs(s)
    if s == "0.5":
        s = "\\frac{1}{2}"
    s = _fix_a_slash_b(s)
    return s

def _sympy_parse(expr: str):
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        ),
    )

def _parse_latex(expr: str) -> str:
    expr = expr.replace("\\tfrac", "\\frac").replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)
    expr = (expr
            .replace("√", "sqrt")
            .replace("π", "pi")
            .replace("∞", "inf")
            .replace("∪", "U")
            .replace("·", "*")
            .replace("×", "*")
            .strip())
    return expr

def _is_float(num: str) -> bool:
    try:
        float(num); return True
    except ValueError:
        return False

def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False

def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))

def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False

def _str_to_int(x: str) -> int:
    x = x.replace(",", "")
    x = float(x)
    return int(x)

def _inject_implicit_mixed_number(step: str) -> str:
    p1 = re.compile(r"([0-9]) +([0-9])")
    step = p1.sub(r"\1+\2", step)
    return step

def _strip_properly_formatted_commas(expr: str) -> str:
    p1 = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    next_expr = expr
    while True:
        nx = p1.sub(r"\1\3\4", next_expr)
        if nx == next_expr:
            break
        next_expr = nx
    return next_expr

def _normalize(expr: Optional[str]) -> Optional[str]:
    if expr is None:
        return None

    m = re.search(r"^\\text{(?P<text>.+?)}$", expr)
    if m is not None:
        expr = m.group("text")

    expr = (expr.replace("\\%", "%").replace("\\$", "$")
                 .replace("$", "").replace("%", "")
                 .replace(" or ", " , ").replace(" and ", " , "))
    expr = (expr.replace("million", "*10^6")
                 .replace("billion", "*10^9")
                 .replace("trillion", "*10^12"))

    for unit in [
        "degree","cm","centimeter","meter","mile","second","minute","hour","day",
        "week","month","year","foot","feet","inch","yard",
    ]:
        expr = re.sub(rf"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub(r"\^ *\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(r",\\! *", "", expr)

    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))

    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except Exception:
            pass

    expr = re.sub(r"- *", "-", expr)
    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")
    expr = expr.replace("{", "").replace("}", "")
    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr

def count_unknown_letters_in_expr(expr: str) -> int:
    expr = expr.replace("sqrt", "").replace("frac", "")
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)

def should_allow_eval(expr: str) -> bool:
    if count_unknown_letters_in_expr(expr) > 2:
        return False
    for bad in BAD_SUBSTRINGS:
        if bad in expr:
            return False
    for rgx in BAD_REGEXES:
        if re.search(rgx, expr) is not None:
            return False
    return True

def are_equal_under_sympy(gt_norm: str, pred_norm: str) -> bool:
    try:
        expr = f"({gt_norm})-({pred_norm})"
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            return bool(simplified == 0)
    except Exception:
        pass
    return False

def split_tuple(expr: str) -> List[str]:
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
    ):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems

def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx:right_brace_idx + 1]

def remove_boxed(s: str) -> Optional[str]:
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except Exception:
        return None

def extract_boxed_answer(solution: str) -> Optional[str]:
    boxed = last_boxed_only_string(solution)
    return remove_boxed(boxed) if boxed else None

def extract_answer(passage: str) -> Optional[str]:
    return extract_boxed_answer(passage) if "\\boxed" in passage or "\\fbox" in passage else None

def grade_answer_mathd(given_answer: str, ground_truth: str) -> bool:
    gt_norm = mathd_normalize_answer(ground_truth)
    pred_norm = mathd_normalize_answer(given_answer)
    return gt_norm == pred_norm

def grade_answer_sympy(given_answer: str, ground_truth: str) -> bool:
    gt_norm = _normalize(ground_truth)
    pred_norm = _normalize(given_answer)
    if gt_norm is None or pred_norm is None:
        return False
    if gt_norm == pred_norm:
        return True
    if len(pred_norm) == 0:
        return False

    gt_elems = split_tuple(gt_norm)
    pred_elems = split_tuple(pred_norm)
    if len(gt_elems) > 1 and (gt_norm[0] != pred_norm[0] or gt_norm[-1] != pred_norm[-1]):
        return False
    if len(gt_elems) != len(pred_elems):
        return False

    for g, p in zip(gt_elems, pred_elems):
        if _is_frac(g) and _is_frac(p):
            if g != p:
                return False
        elif (_str_is_int(g) != _str_is_int(p)):
            return False
        else:
            if not are_equal_under_sympy(g, p):
                return False
    return True

def grade_answer_verl(solution_str: str, ground_truth: str) -> Dict[str, Any]:
    """
    综合判分：先从 rollout/gt 中抽取 boxed，随后用 mathd 或 sympy 比对。
    返回包含解析与判分细节的字典。
    """
    if not ground_truth:
        return {"parsed_gt": None, "parsed_pred": None, "correct": False,
                "mathd_equal": False, "sympy_equal": False}

    gt_parsed = extract_answer(ground_truth) if "\\boxed" in ground_truth or "\\fbox" in ground_truth else ground_truth
    pred_parsed = extract_answer(solution_str)

    if pred_parsed is None:
        return {"parsed_gt": gt_parsed, "parsed_pred": None, "correct": False,
                "mathd_equal": False, "sympy_equal": False}

    # 两路判定
    mathd_ok = grade_answer_mathd(pred_parsed, gt_parsed)
    sympy_ok = False if mathd_ok else grade_answer_sympy(pred_parsed, gt_parsed)

    return {
        "parsed_gt": gt_parsed,
        "parsed_pred": pred_parsed,
        "correct": bool(mathd_ok or sympy_ok),
        "mathd_equal": bool(mathd_ok),
        "sympy_equal": bool(sympy_ok),
    }

def evaluate_samples(samples: List[Any], ground_truth: str) -> List[Dict[str, Any]]:
    """对一条记录的多个 sample 逐个评估，返回 evaluations 列表。"""
    outs: List[Dict[str, Any]] = []
    for s in samples:
        text = s if isinstance(s, str) else str(s)
        res = grade_answer_verl(text, ground_truth)
        outs.append(res)
    return outs
