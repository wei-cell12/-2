"""Build the requested candidate pool from pinned, cached public source files."""
import csv
import hashlib
import io
import json
import re
import shutil
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageStat

from selections import DAILY, CHART

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--cache', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
BUILD = args.cache.resolve()
ROOT = args.out.resolve()
for directory in ['images/daily_logic', 'images/chart_data', 'images/medical_lesion',
                  'synthetic_data', 'provenance', 'medical_annotations/masks', 'previews']:
    (ROOT / directory).mkdir(parents=True, exist_ok=True)

records = []
provenance = []
seen_pixels = {}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def check_image(raw):
    with Image.open(io.BytesIO(raw)) as image:
        image.verify()
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        rgb = image.convert('RGB')
        width, height = image.size
        assert width >= 384 and height >= 384, (width, height)
        gray = rgb.convert('L')
        stat = ImageStat.Stat(gray)
        assert gray.getextrema()[1] > 5 and stat.stddev[0] > 1, 'blank/near-constant image'
        pixel_hash = sha(f'{width}x{height}:RGB:'.encode() + rgb.tobytes())
        return width, height, image.format, pixel_hash, stat.stddev[0]


def add_image(domain, number, raw, source, original_id, question, answer, claim, anchor, details):
    width, height, fmt, pixel_hash, stddev = check_image(raw)
    assert pixel_hash not in seen_pixels, ('duplicate image', original_id, seen_pixels.get(pixel_hash))
    seen_pixels[pixel_hash] = original_id
    extension = '.jpg' if fmt == 'JPEG' else '.png'
    relative = f'images/{domain}/{domain}_{number:03d}{extension}'
    target = ROOT / relative
    if fmt in ['JPEG', 'PNG']:
        target.write_bytes(raw)
    else:
        Image.open(io.BytesIO(raw)).save(target, format='PNG')
    _, _, _, saved_pixels, _ = check_image(target.read_bytes())
    assert pixel_hash == saved_pixels, 'pixel modification'
    short = {'daily_logic': 'logic', 'chart_data': 'chart', 'medical_lesion': 'medical'}[domain]
    unit_id = f's1_1_{short}_{number:03d}'
    records.append(dict(unit_id=unit_id, domain=domain, image_path=relative, width=width,
                        height=height, source_dataset=source, original_id=original_id,
                        original_question=question, ground_truth=answer,
                        candidate_wrong_claim=claim, evidence_anchor=anchor,
                        is_synthetic=source == 'local/matplotlib_tier1_1_v1'))
    details['image_bytes_unchanged'] = sha(raw) == sha(target.read_bytes())
    details['source_image_format'] = fmt
    details['saved_image_format'] = 'JPEG' if extension == '.jpg' else 'PNG'
    provenance.append(dict(unit_id=unit_id, source_dataset=source, original_id=original_id,
                           source_bytes_sha256=sha(raw), saved_bytes_sha256=sha(target.read_bytes()),
                           rgb_pixels_sha256=pixel_hash, grayscale_stddev=stddev, **details))
    return unit_id


def source_info(name):
    return json.loads((BUILD / (name.replace('/', '_') + '_info.json')).read_text())


def build_daily():
    rows = pq.read_table(BUILD / 'realworld.parquet').to_pylist()
    info = source_info('xai-org/RealWorldQA')
    for number, (index, claim, anchor) in enumerate(DAILY, 1):
        row = rows[index]
        raw = row['image']['bytes']
        add_image('daily_logic', number, raw, info['id'],
                  f'test/data/test-00000-of-00002.parquet/row_{index}',
                  row['question'], row['answer'], claim,
                  anchor + ' 本来源未提供 bbox，位置描述由初筛整理。',
                  dict(revision=info['sha'], split='test', shard='data/test-00000-of-00002.parquet',
                       row_index=index, original_image_path=row['image']['path'],
                       source_answer=row['answer'], source_question=row['question'],
                       license='cc-by-nd-4.0', image_bytes_unchanged=True,
                       review='source_QA_plus_visual_prescreen'))
    return rows


def build_real_charts():
    rows = pq.read_table(BUILD / 'chartqa.parquet').to_pylist()
    info = source_info('ahmed-masry/ChartQAPro')
    for number, (index, claim, anchor) in enumerate(CHART, 1):
        row = rows[index]
        assert row['Question Type'] == 'Factoid' and len(row['Question']) == len(row['Answer']) == 1
        assert row['Answer'][0].lower() != 'unanswerable'
        add_image('chart_data', number, row['image'], info['id'],
                  f'test/data/test-00000-of-00001.parquet/row_{index}',
                  row['Question'][0], row['Answer'][0], claim,
                  anchor + ' 本来源未提供 bbox，位置描述由初筛整理。',
                  dict(revision=info['sha'], split='test', row_index=index,
                       shard='data/test-00000-of-00001.parquet',
                       source_question=row['Question'], source_answer=row['Answer'],
                       question_type=row['Question Type'], source_year=row['Year'],
                       source_paragraph=row['Paragraph'], license='mit (dataset card)',
                       image_bytes_unchanged=True, review='source_QA_plus_visual_prescreen'))
    return rows


def build_medical():
    info = source_info('Voxel51/SLAKE')
    plan = json.loads((BUILD / 'medical_plan.json').read_text())
    for number, entry in enumerate(plan, 1):
        sample = entry['sample']
        key = entry['question_key']
        answer = sample[key.replace('question', 'answer')]['label']
        category = entry['category']
        opposite = {'Left': 'Right', 'Right': 'Left', 'Heart': 'Lung', 'Lung': 'Heart', 'Liver': 'Kidney',
                    'Yes': 'No', 'No': 'Yes'}[answer]
        if category == 'laterality':
            claim = f'异常位于患者的{"左" if opposite == "Left" else "右"}肺。'
            anchor = '胸片中标注的肺部异常区域和 L/R 方位标记；左右沿用原 QA 的患者解剖侧，区别于照片左右。'
        elif category == 'lesion_organ':
            claim = '原题询问的异常位于肾脏。'
            anchor = '腹部 CT 中肝脏及其局部异常；原问答将异常器官标注为 Liver。'
        elif category == 'organ_binary':
            claim = f'在心脏与肺的二选一中，异常器官是{"肺" if opposite == "Lung" else "心脏"}。'
            anchor = f'胸片中原问答指定的异常器官区域；原 QA 的二选一答案为 {answer}。'
        else:
            scope = '患者左肺' if 'left lung' in sample[key].lower() else '患者右肺' if 'right lung' in sample[key].lower() else '肾脏' if 'kidney' in sample[key].lower() else '肺部' if 'lung' in sample[key].lower() else '图像中'
            claim = scope + ('没有异常。' if answer == 'Yes' else '存在异常。')
            anchor = '原题指定的图像范围或肺部；必须保持原题限定的器官、解剖侧和否定范围。'
        detections = sample.get('detections', {}).get('detections', [])
        boxes = [dict(label=d['label'], xywh_normalized=d['bounding_box']) for d in detections]
        mask_path = sample.get('segmentation', {}).get('mask_path')
        annotation_path = f'medical_annotations/medical_lesion_{number:03d}.json'
        unit_id = add_image('medical_lesion', number,
                            (BUILD / 'medical_cache' / Path(sample['filepath']).name).read_bytes(),
                            info['id'], f'xmlab{entry["id"]}/{key}', sample[key], answer, claim,
                            anchor + ' 原始区域框 ' + json.dumps(boxes, ensure_ascii=False) +
                            f'。完整标注 {annotation_path}。框标签与问答所指区域应分别核对。',
                            dict(revision=info['sha'], original_image_id=f'xmlab{entry["id"]}',
                                 source_image=sample['filepath'], source_question_key=key,
                                 source_question=sample[key], source_answer=answer,
                                 category=category, license='cc-by-4.0 (dataset card)',
                                 annotation_path=annotation_path, image_bytes_unchanged=True,
                                 review='source_QA_plus_image_quality_screen; clinical_recheck_pending'))
        local_mask = None
        if mask_path:
            local_mask = 'medical_annotations/masks/' + Path(mask_path).name
            shutil.copyfile(BUILD / 'medical_cache' / Path(mask_path).name, ROOT / local_mask)
        write_json(ROOT / annotation_path, dict(unit_id=unit_id, original_sample=sample,
                    selected_question_key=key, boxes=boxes,
                    bbox_convention='normalized [x_left, y_top, width, height] in source image coordinates',
                    mask_path=local_mask, mask_semantics='original SLAKE segmentation labels, not automatically lesion masks',
                    mask_sha256=sha((ROOT / local_mask).read_bytes()) if local_mask else None,
                    source_mask_path=mask_path))


def build_synthetic():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.spines.top': False})
    rng = np.random.default_rng(20261120)
    families = ['dual_axis'] * 7 + ['close_grouped_bars'] * 6 + ['truncated_y'] * 6 + ['crossing_lines'] * 6
    for index, family in enumerate(families, 1):
        number = 25 + index
        figure, ax = plt.subplots(figsize=(10, 6), layout='constrained')
        table = []
        truth = dict(generator='numpy.random.default_rng', seed=20261120, index=index,
                     family=family, dpi=200, language='en')
        if family == 'dual_axis':
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May']
            revenue = rng.choice(np.arange(90, 181), size=5, replace=False)
            costs = rng.choice(np.arange(22, 62), size=5, replace=False)
            winner = int(revenue.argmax())
            if int(costs.argmax()) == winner:
                other = (winner + 2) % 5
                costs[winner], costs[other] = costs[other], costs[winner]
            x = np.arange(5)
            bars = ax.bar(x, revenue, color='#3275a8', width=.55, label='Revenue')
            ax.bar_label(bars, padding=4, fontsize=11)
            ax.set(xticks=x, xticklabels=months, ylabel='Revenue (thousand USD)', ylim=(0, 220))
            right = ax.twinx()
            right.plot(x, costs, color='#c25925', marker='o', linewidth=2.5, label='Operating cost')
            right.set(ylabel='Operating cost (thousand USD)', ylim=(0, 80))
            for xx, value in zip(x, costs):
                right.annotate(str(value), (xx, value), xytext=(12, -18), textcoords='offset points', ha='center', color='#923b17', bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=1))
            figure.legend([bars, right.lines[0]], ['Revenue (left axis)', 'Operating cost (right axis)'], loc='outside upper center', ncol=2)
            question = 'Which month has the highest revenue (thousand USD)?'
            answer = months[winner]
            wrong = months[int(costs.argmax())]
            claim = f'The month with the highest revenue is {wrong}.'
            anchor = f'Revenue bars and left axis; the maximum bar is {answer} at {int(revenue[winner])} thousand USD. The line uses the right axis.'
            table = [dict(month=m, revenue_thousand_usd=int(r), operating_cost_thousand_usd=int(c)) for m, r, c in zip(months, revenue, costs)]
            truth.update(answer=answer, wrong_answer=wrong, maximum_revenue=int(revenue[winner]), left_axis_limits=[0, 220], right_axis_limits=[0, 80])
        elif family == 'close_grouped_bars':
            regions = ['North', 'East', 'South', 'West']
            a = rng.integers(94, 110, size=4)
            differences = rng.choice([-3, -2, -1, 1, 2, 3], size=4)
            target = (index - 8) % 4
            differences[target] = abs(differences[target]) * (1 if index % 2 == 0 else -1)
            b = a + differences
            x = np.arange(4)
            first = ax.bar(x - .19, a, .36, color='#3275a8', label='Program A')
            second = ax.bar(x + .19, b, .36, color='#d58b3a', label='Program B')
            ax.bar_label(first, padding=4)
            ax.bar_label(second, padding=4)
            ax.set(xticks=x, xticklabels=regions, ylabel='Completed tasks (count)', ylim=(0, 130))
            ax.legend(loc='upper left')
            target = (index - 8) % 4
            answer = 'Program A' if a[target] > b[target] else 'Program B'
            wrong = 'Program B' if answer == 'Program A' else 'Program A'
            question = f'In {regions[target]}, which program completed more tasks, Program A or Program B?'
            claim = f'In {regions[target]}, {wrong} completed more tasks.'
            anchor = f'{regions[target]} group and exact labels: Program A = {int(a[target])}, Program B = {int(b[target])}.'
            table = [dict(region=r, program_a=int(aa), program_b=int(bb)) for r, aa, bb in zip(regions, a, b)]
            truth.update(answer=answer, wrong_answer=wrong, target_region=regions[target], difference=int(a[target]-b[target]))
        elif family == 'truncated_y':
            periods = ['P1', 'P2', 'P3', 'P4', 'P5']
            start = int(rng.integers(950, 990))
            values = [start]
            for step in range(4):
                delta = int(rng.integers(2, 7)) * (1 if (index + step) % 3 else -1)
                values.append(values[-1] + delta)
            if values[-1] == values[0]:
                values[-1] += 1
            if (values[-1] > values[0]) != (index % 2 == 0):
                values = [2 * values[0] - v for v in values]
            ax.plot(periods, values, color='#3275a8', marker='o', linewidth=2.5)
            for period, value in zip(periods, values):
                ax.annotate(str(value), (period, value), xytext=(0, 10), textcoords='offset points', ha='center')
            lower, upper = min(values)-6, max(values)+10
            ax.set(ylabel='Output (units)', ylim=(lower, upper))
            ax.text(.02, .05, 'Y-axis starts above zero', transform=ax.transAxes, color='#6b4930')
            difference = int(values[-1]-values[0])
            assert difference != 0
            question = 'What is the signed change in output from P1 to P5 (P5 minus P1), in units?'
            answer = f'{difference:+d} units'
            wrong_value = difference * 10
            claim = f'The signed change from P1 to P5 is {wrong_value:+d} units.'
            anchor = f'P1 label {values[0]}, P5 label {values[-1]}, and truncated Y-axis; exact difference = {difference:+d} units.'
            table = [dict(period=p, output_units=v) for p,v in zip(periods, values)]
            truth.update(answer=answer, wrong_answer=f'{wrong_value:+d} units', signed_change_units=difference, y_limits=[lower,upper])
        else:
            periods = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6']
            base = np.array([[18, 39, 24, 45, 28, 53], [46, 25, 42, 27, 49, 31], [29, 48, 32, 50, 37, 44]])
            values = base + rng.integers(-3, 4, size=(3,6)) + (index-20)
            target = (index - 20) % 6
            desired_winner = (index - 20) % 3
            shift = desired_winner - int(values[:, target].argmax())
            values = np.roll(values, shift, axis=0)
            names = ['Series A', 'Series B', 'Series C']
            for name, series, color, marker in zip(names, values, ['#3275a8','#c25925','#58854a'], ['o','s','^']):
                ax.plot(periods, series, label=name, color=color, marker=marker, linewidth=2)
            ax.set(ylabel='Measurement (units)', ylim=(5,70))
            ax.legend(loc='upper left', ncol=3)
            target = (index - 20) % 6
            vals = values[:,target]
            assert len(set(map(int,vals))) == 3
            winner = int(vals.argmax()); loser = int(vals.argmin())
            answer = names[winner]; wrong = names[loser]
            for name, series in zip(names, values):
                ax.annotate(str(int(series[target])), (periods[target], series[target]), xytext=(9, 0), textcoords='offset points', fontsize=11)
            question = f'At {periods[target]}, which series has the highest measurement?'
            claim = f'At {periods[target]}, {wrong} has the highest measurement.'
            anchor = f'{periods[target]} positions of all three lines and legend; ' + ', '.join(f'{n} = {int(v)}' for n,v in zip(names,vals)) + '.'
            table = [dict(period=p, series_a=int(values[0,j]), series_b=int(values[1,j]), series_c=int(values[2,j])) for j,p in enumerate(periods)]
            truth.update(answer=answer, wrong_answer=wrong, target_period=periods[target])
        ax.set_title(f'Simulated study {index:02d} — {family.replace("_", " ").title()}', pad=14)
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
        buffer=io.BytesIO()
        figure.savefig(buffer, format='png', dpi=200, metadata={'Description':'Fresh synthetic data; not an observation of a real population'})
        plt.close(figure)
        csv_path = ROOT / 'synthetic_data' / f'chart_data_{number:03d}.csv'
        with csv_path.open('w', newline='') as f:
            writer=csv.DictWriter(f, fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
        truth.update(question=question, candidate_wrong_claim=claim, exact_rows=table)
        write_json(ROOT/'synthetic_data'/f'chart_data_{number:03d}_truth.json', truth)
        add_image('chart_data', number, buffer.getvalue(), 'local/matplotlib_tier1_1_v1',
                  f'seed_20261120/chart_{index:02d}/{family}', question, answer, claim, anchor,
                  dict(seed=20261120, synthetic_index=index, family=family, dpi=200,
                       data_table=f'synthetic_data/chart_data_{number:03d}.csv',
                       truth_path=f'synthetic_data/chart_data_{number:03d}_truth.json',
                       license='CC-BY-4.0 (new synthetic charts and local annotations)',
                       review='exact_data_assertions_plus_visual_layout_review'))


def main():
    daily_rows=build_daily()
    chart_rows=build_real_charts()
    build_medical()
    build_synthetic()
    assert len(records)==150
    write_json(ROOT/'candidates_1_1.json',records)
    write_json(ROOT/'provenance'/'image_manifest.json',provenance)
    for name in ['xai-org/RealWorldQA','ahmed-masry/ChartQAPro','Voxel51/SLAKE']:
        info=source_info(name)
        write_json(ROOT/'provenance'/(name.replace('/','_')+'.json'),
                   dict(source_dataset=name, source_url='https://huggingface.co/datasets/'+name,
                        revision=info['sha'], license=info.get('cardData',{}).get('license'),
                        authors=info.get('cardData',{}).get('authors'),
                        downloaded_files=info.get('siblings'), dataset_card=info.get('cardData')))
    slake=json.loads((BUILD/'slake_samples.json').read_text())['samples']
    stats=dict(daily_source_shard_rows=len(daily_rows), chart_source_rows=len(chart_rows),
               medical_source_rows=len(slake), selected=Counter(r['source_dataset'] for r in records),
               medical_below_resolution=sum(min(x['metadata']['width'],x['metadata']['height'])<384 for x in slake),
               daily_below_resolution=sum(min(Image.open(io.BytesIO(r['image']['bytes'])).size)<384 for r in daily_rows),
               chart_below_resolution=sum(min(Image.open(io.BytesIO(r['image'])).size)<384 for r in chart_rows),
               chart_unanswerable=sum(any(a.lower()=='unanswerable' for a in r['Answer']) for r in chart_rows))
    write_json(ROOT/'provenance'/'screening_stats.json',stats)
    print(json.dumps(stats,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
