"""Download only the pinned source artifacts needed to rebuild this pool."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import requests

parser = argparse.ArgumentParser()
parser.add_argument('--cache', type=Path, required=True)
args = parser.parse_args()
cache = args.cache.resolve()
cache.mkdir(parents=True, exist_ok=True)
pool = Path(__file__).resolve().parent.parent
manifest = json.loads((pool/'provenance/image_manifest.json').read_text())


def download(url, destination):
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.part')
    with requests.get(url, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        with temporary.open('wb') as output:
            for block in response.iter_content(4*1024*1024):
                output.write(block)
    temporary.replace(destination)


sources = ['xai-org_RealWorldQA', 'ahmed-masry_ChartQAPro', 'Voxel51_SLAKE']
metadata = {}
for filename in sources:
    item = json.loads((pool/'provenance'/f'{filename}.json').read_text())
    name = item['source_dataset']
    # The requested RealWorldQA alias resolves to the canonical RealworldQA name.
    cache_name = 'xai-org_RealWorldQA' if name.startswith('xai-org/') else filename
    info = dict(id=name, sha=item['revision'], cardData=item['dataset_card'], siblings=item['downloaded_files'])
    (cache/f'{cache_name}_info.json').write_text(json.dumps(info))
    metadata[filename] = item

for filename, shard, target in [
    ('xai-org_RealWorldQA','data/test-00000-of-00002.parquet','realworld.parquet'),
    ('ahmed-masry_ChartQAPro','data/test-00000-of-00001.parquet','chartqa.parquet'),
]:
    item=metadata[filename]
    download(f'https://huggingface.co/datasets/{item["source_dataset"]}/resolve/{item["revision"]}/{shard}', cache/target)

item=metadata['Voxel51_SLAKE']
base=f'https://huggingface.co/datasets/{item["source_dataset"]}/resolve/{item["revision"]}/'
for filename in ['samples.json','metadata.json']:
    download(base+filename, cache/('slake_'+filename))

medical_plan=[]
tasks=[]
for entry in manifest:
    if entry['source_dataset']!='Voxel51/SLAKE':
        continue
    annotation=json.loads((pool/entry['annotation_path']).read_text())
    sample=annotation['original_sample']
    medical_plan.append(dict(id=int(entry['original_image_id'].replace('xmlab','')),
                             question_key=annotation['selected_question_key'],
                             category=entry['category'], sample=sample))
    for source in [sample['filepath'], sample.get('segmentation',{}).get('mask_path')]:
        if source:
            tasks.append((base+source, cache/'medical_cache'/Path(source).name))
(cache/'medical_plan.json').write_text(json.dumps(medical_plan,ensure_ascii=False,indent=2))
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
    futures=[executor.submit(download,url,target) for url,target in tasks]
    for future in futures:
        future.result()
print('Pinned source cache ready')
