"""Conservative content extraction; quoted/hypothetical claims are not predictions.

Rule-time accuracy is secondary to joint answer/explanation judgment. Ambiguous
or missing content stays unresolved, and source outputs are never modified.
"""
import re
TIME=r'(?<!\d)(\d{1,2}):(\d{2})(?::(\d{2}))?(?!\d)'
ASSERTION=re.compile(r'(?:\b(?:the\s+)?time(?:\s+actually)?(?:\s+(?:shown|displayed))?(?:\s+on\s+the\s+clock)?(?:\s+is)?|\bclock\s+(?:actually\s+)?(?:shows|reads)|\b(?:correct|actual)\s+(?:reading|answer)(?:\s+is)?|\bfinal\s+answer(?:\s+is)?)\s*[:：]?\s*(?:(?:approximately|actually|exactly|roughly|indeed)\s+)*\**\s*'+TIME,re.I)
def literal(text):
    m=re.fullmatch(r'\s*(\d{1,2}):(\d{2})(?::(\d{2}))?\s*',text or '')
    if not m:return None
    h,minute,sec=int(m[1]),int(m[2]),int(m[3] or 0)
    if not (0<=h<=23 and minute<60 and sec<60):return None
    return f'{h%12:02d}:{minute:02d}:{sec:02d}'
def extract(text):
    if not text:return dict(value=None,status='unresolved',reason='No answer text',evidence=[])
    candidates=[]
    for line in text.splitlines():
        for m in ASSERTION.finditer(line):
            # Scope exclusions to the current sentence, so rejecting a claim
            # does not suppress a later explicit correction on the same line.
            prefix=re.split(r'[.!?;]',line[:m.start()])[-1].lower()
            if re.search(r'\bif(?:\s+the)?\s*$',prefix):continue
            if any(x in prefix for x in ['provided answer','suggested answer','prompt states','prompt says','if the','if it','would be','not ','isn\'t','is not','hypothesis']):continue
            if re.search(r'\b(?:statement|claim|assertion|suggestion)\s+(?:that\s+)?$',prefix.strip()):continue
            following=line[m.end():]
            if re.match(r'\s*\**\s*(?:is|was|would\s+be)\s+(?:not\s+(?:correct|right|accurate|true)|incorrect|wrong|false)',following,re.I):continue
            if re.match(r'\s*\**\s*(?:or|/)\s*\**\s*\d',following,re.I):continue
            token=':'.join(x for x in m.groups() if x is not None)
            value=literal(token)
            precision='seconds' if m.group(3) is not None else 'minutes'
            span=m.group(0)
            # Preserve explicitly supplied seconds in an immediately adjacent
            # parenthetical restatement, e.g. 1:10 (1:10:30), without inference.
            detail=re.match(r'\s*\**\s*\(\s*(?:about\s+)?'+TIME,following,re.I)
            if value is not None and precision=='minutes' and detail and detail.group(3) is not None:
                detailed=literal(':'.join(detail.groups()))
                if detailed is not None and detailed[:5]==value[:5]:value=detailed;precision='seconds';span+=detail.group(0)
            if value is not None:candidates.append(dict(value=value,precision=precision,quote=line.strip(),span=span))
    # A bare final time is accepted; a list of possible/hypothetical times is not.
    if not candidates:
        for line in text.splitlines():
            t=line.strip().strip('*').strip().rstrip('.')
            value=literal(t)
            if value is not None:candidates.append(dict(value=value,precision='seconds' if t.count(':')==2 else 'minutes',quote=line.strip(),span=t))
    if not candidates:return dict(value=None,status='unresolved',reason='No unambiguous own time assertion; manual content review required',evidence=[])
    last=candidates[-1]
    return dict(value=last['value'],status='resolved',reason='Last complete explicit own assertion; no inference from truncated text',precision=last['precision'],evidence=candidates)
def judge(text,gold):
    e=extract(text);g=literal(gold)
    assert g is not None,'Invalid publisher gold time'
    correct=None
    if e['value'] is not None:
        if e.get('precision')=='minutes' and not g.endswith(':00') and e['value'][:5]==g[:5]:
            e.update(status='unresolved',reason='Correct hour/minute but seconds not specified; precision unresolved')
        else:correct=e['value']==g
    return dict(**e,gold=g,correct=correct,judgment='time-content rule only; joint semantic/visual explanation pending')
def selftest():
    assert judge('Therefore, the clock shows exactly **9:00**.','9:00:00')['correct'] is True
    assert judge('The time shown is actually **4:00**, not 5:00:00.','4:00:00')['correct'] is True
    assert judge('The provided answer is 6:12:47.','6:02:06')['correct'] is None
    assert judge('If the time is 6:12:47, the hands should differ.','6:02:06')['correct'] is None
    assert judge('The time is 6:01 or 6:02.','6:02:06')['correct'] is None
    assert judge('The time is **6:02**.','6:02:06')['correct'] is None
    assert judge('Time: 6:12:07.\nProvided Answer: 6:12:47.','6:02:06')['value']=='06:12:07'
    assert judge('The time is **1:10:30**.','1:10:30')['correct'] is True
    assert judge('The clock actually shows **4:00:00**, not 5:00:00.','4:00:00')['correct'] is True
    assert judge('The provided answer: the clock actually shows **5:00:00**.','4:00:00')['correct'] is None
    assert judge('The clock actually shows **7:00:00**.\nNote: The statement that the time is 12:35:00 is not correct. If it were 12:35, the hands should differ. The correct reading is **7:00**.','7:00:00')['correct'] is True
    assert judge('The time is 12:35:00 is not correct.','7:00:00')['correct'] is None
    assert judge('The claim that the time is 12:35:00 is wrong.','7:00:00')['correct'] is None
    assert judge('**Time shown: 1:10 (1:10:30).**','1:10:30')['correct'] is True
    assert judge('The time is 1:10 (2:10:30).','1:10:30')['correct'] is None
    assert judge('So the time actually displayed is **1:10:30**.','1:10:30')['correct'] is True
    assert judge('So the time displayed is **5:20** (5:20:45 if you include the seconds).','5:20:45')['correct'] is True
    assert literal('10:60:00') is None
    assert literal('12:00')=='00:00:00'
    return dict(passed=True,quoted_claim_not_prediction=True,ambiguous_and_missing_precision_unresolved=True,last_own_assertion=True)
