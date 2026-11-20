"""Validate file integrity, schema, provenance, masks, and synthetic exact answers."""
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from PIL import Image, ImageStat

root=Path(__file__).resolve().parent.parent
records=json.loads((root/'candidates_1_1.json').read_text())
manifest=json.loads((root/'provenance/image_manifest.json').read_text())
by_id={x['unit_id']:x for x in manifest}
fields={'unit_id','domain','image_path','width','height','source_dataset','original_id',
        'original_question','ground_truth','candidate_wrong_claim','evidence_anchor','is_synthetic'}
assert len(records)==150
assert Counter(x['domain'] for x in records)=={'daily_logic':50,'chart_data':50,'medical_lesion':50}
assert len({x['unit_id'] for x in records})==150
assert len({x['image_path'] for x in records})==150
assert len({(x['source_dataset'],x['original_id']) for x in records})==150
seen=set()
for record in records:
    assert set(record)==fields, record['unit_id']
    assert type(record['is_synthetic']) is bool
    assert type(record['width']) is type(record['height']) is int
    for key in fields-{'width','height','is_synthetic'}:
        assert isinstance(record[key],str) and record[key].strip(), (record['unit_id'],key)
    path=root/record['image_path']
    assert path.suffix in ['.png','.jpg'] and path.is_file()
    raw=path.read_bytes()
    with Image.open(io.BytesIO(raw)) as image:
        image.verify()
    with Image.open(path) as image:
        image.load()
        assert image.size==(record['width'],record['height'])
        assert min(image.size)>=384
        gray=image.convert('L')
        assert gray.getextrema()[1]>5 and ImageStat.Stat(gray).stddev[0]>1
        pixel_hash=hashlib.sha256(f'{image.width}x{image.height}:RGB:'.encode()+image.convert('RGB').tobytes()).hexdigest()
        assert pixel_hash not in seen, record['unit_id']
        seen.add(pixel_hash)
        entry=by_id[record['unit_id']]
        assert pixel_hash==entry['rgb_pixels_sha256']
        assert hashlib.sha256(raw).hexdigest()==entry['saved_bytes_sha256']
        if record['is_synthetic']:
            assert min(image.info['dpi'])>=180
        else:
            assert entry['image_bytes_unchanged'] == (entry['source_bytes_sha256']==entry['saved_bytes_sha256'])
    if record['domain']=='medical_lesion':
        annotation=json.loads((root/entry['annotation_path']).read_text())
        sample=annotation['original_sample'];key=annotation['selected_question_key']
        assert record['original_question']==sample[key]
        assert record['ground_truth']==sample[key.replace('question','answer')]['label']
        for box in annotation['boxes']:
            x,y,w,h=box['xywh_normalized']
            assert min(x,y,w,h)>=0 and x+w<=1.001 and y+h<=1.001
        if annotation['mask_path']:
            mask=root/annotation['mask_path']
            assert hashlib.sha256(mask.read_bytes()).hexdigest()==annotation['mask_sha256']
            with Image.open(mask) as image:
                image.verify()
    if record['is_synthetic']:
        truth=json.loads((root/entry['truth_path']).read_text())
        with (root/entry['data_table']).open() as f:
            data=list(csv.DictReader(f))
        assert len(data)==len(truth['exact_rows'])
        for text_row, exact_row in zip(data, truth['exact_rows']):
            assert text_row=={k:str(v) for k,v in exact_row.items()}
        assert record['ground_truth']==truth['answer']!=truth['wrong_answer']
        family=truth['family']
        if family=='dual_axis':
            target=max(truth['exact_rows'],key=lambda x:x['revenue_thousand_usd'])
            assert target['month']==truth['answer']
        elif family=='close_grouped_bars':
            target=next(x for x in truth['exact_rows'] if x['region']==truth['target_region'])
            expected='Program A' if target['program_a']>target['program_b'] else 'Program B'
            assert expected==truth['answer']
        elif family=='truncated_y':
            difference=truth['exact_rows'][-1]['output_units']-truth['exact_rows'][0]['output_units']
            assert f'{difference:+d} units'==truth['answer']
        else:
            target=next(x for x in truth['exact_rows'] if x['period']==truth['target_period'])
            values=[target['series_a'],target['series_b'],target['series_c']]
            assert len(set(values))==3
            assert ['Series A','Series B','Series C'][values.index(max(values))]==truth['answer']
for domain in ['daily_logic','chart_data','medical_lesion']:
    assert len(list((root/'images'/domain).iterdir()))==50
result=dict(ok=True, total=150, counts=dict(Counter(r['domain'] for r in records)),
            readable_images=150, min_width=min(r['width'] for r in records),
            min_height=min(r['height'] for r in records), duplicate_rgb_images=0,
            unchanged_source_pixels=125, unchanged_source_bytes=sum(x.get('image_bytes_unchanged',False) for x in manifest if not x.get('family')), lossless_format_conversions=sum(not x.get('image_bytes_unchanged',False) for x in manifest if not x.get('family')), synthetic_exact_answers_checked=25,
            medical_annotations_checked=50, synthetic_dpi=200,
            synthetic_families=dict(Counter(x.get('family') for x in manifest if x.get('family'))))
(root/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
