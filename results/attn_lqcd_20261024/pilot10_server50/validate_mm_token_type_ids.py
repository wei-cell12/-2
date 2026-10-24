"""CPU-only validation of processor modality labels on unchanged pilot token IDs."""
import hashlib, inspect, json
from pathlib import Path
import numpy as np
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

ROOT = Path(__file__).resolve().parent
MODEL = '/mnt/ssd/gy/models/Qwen3.8-27B'
processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=False, min_pixels=65536, max_pixels=1048576)
assert hasattr(processor, 'create_mm_token_type_ids')
assert 'mm_token_type_ids' in inspect.signature(Qwen3_5ForConditionalGeneration.forward).parameters
config = json.loads((Path(MODEL) / 'config.json').read_text())
image_id = config['image_token_id']
rows = [json.loads(s) for s in (ROOT / 'manifest_pilot10.jsonl').read_text().splitlines()]
results = []
for row in rows:
    ids = row['input_ids'] + row['output_ids']
    original = list(ids)
    types = np.asarray(processor.create_mm_token_type_ids([ids]), dtype=np.int64)
    assert types.shape == (1, len(ids))
    expected_image = np.asarray(ids) == image_id
    assert np.array_equal(types[0] == 1, expected_image)
    assert set(np.unique(types)).issubset({0, 1})
    assert expected_image.sum() > 0
    assert not types[0, len(row['input_ids']):].any()
    assert ids == original
    results.append(dict(id=row['id'], condition=row['condition'], tokens=len(ids),
                        image_tokens=int(expected_image.sum()),
                        token_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(), ok=True))
assert len(results) == 20
receipt = dict(ok=True, sequences=20, retokenized=False,
               processor=type(processor).__name__, checks=results)
(ROOT / 'mm_token_type_ids_validation.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(dict(ok=True, sequences=20, retokenized=False, processor=receipt['processor'])))
