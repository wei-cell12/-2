"""Conservative provisional extraction; final answer+explanation labels belong to Luna."""
import re

def judge(answer,gold):
    text=(answer or '').strip()
    matches=list(re.finditer(r'(?im)^\s*(?:\*\*)?(yes|no)\b',text))
    value=matches[-1].group(1).lower() if len({m.group(1).lower() for m in matches})==1 else None
    return dict(value=value,correct=(value==gold.lower()) if value is not None else None,
                provisional_only=True,semantic_negation_and_explanation_review='same Luna',
                unresolved_reason=None if value is not None else 'No unambiguous explicit yes/no answer; retain for Luna')

