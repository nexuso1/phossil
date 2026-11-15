"""Literature references for the canonical sites of demo_evaluation.py, from UniProt.

The canonical sites in demo_evaluation.DEMO only name rows of its output; the labels it scores come
from dbPTM. This script looks every canonical site up in the "Modified residue" features of its
UniProtKB entry and collects the evidence behind it:

    experimental     ECO:0000269, a paper characterising the site (PubMed)
    large-scale      ECO:0007744, a phosphoproteomics study (PubMed)
    by similarity    ECO:0000250, inherited from another entry (usually the mouse or rat ortholog);
                     that entry is fetched, the site is mapped onto it by a global alignment, and
                     its evidence is followed (two levels at most)
    curator / model  ECO:0000305 / ECO:0000255 etc., inferred, no paper

A site without a Modified residue feature is reported as 'not annotated'.

    cd model
    python demo_site_references.py            # fetches what is not cached yet
    python demo_site_references.py --refresh  # fetches everything again

Downloaded entries and citations are cached in ../data/demo_eval/uniprot/, so a rerun works offline
and reproduces the same table; UniProt changes over time, so the retrieval date is recorded.
Writes, into ../data/demo_eval/:
    canonical_references.csv   one row per canonical site: annotation, evidence class, PubMed IDs
    references.csv             one row per cited paper, with the sites it is cited for
demo_evaluation.py merges canonical_references.csv into its canonical.csv when it exists.
"""
import argparse
import datetime
import json
import os
import time
import urllib.request

import pandas as pd
from Bio import Align
from Bio.Align import substitution_matrices

from demo_evaluation import DEMO, resolve_sites

UNIPROT = 'https://rest.uniprot.org'
EVIDENCE_CLASS = {'ECO:0000269': 'experimental', 'ECO:0007744': 'large-scale', 'ECO:0000250': 'by similarity',
                  'ECO:0000305': 'curator inference', 'ECO:0000255': 'sequence model'}
# Order in which a site's strongest evidence is reported
STRENGTH = ['experimental', 'large-scale', 'by similarity', 'curator inference', 'sequence model', 'other']


class Cache:
    def __init__(self, directory, refresh):
        self.directory = directory
        self.refresh = refresh
        os.makedirs(directory, exist_ok=True)

    def get(self, kind, key, url):
        path = os.path.join(self.directory, f'{kind}_{key}.json')
        if self.refresh or not os.path.exists(path):
            request = urllib.request.Request(url, headers={'User-Agent': 'python-urllib', 'Accept': 'application/json'})
            with urllib.request.urlopen(request, timeout=60) as response:
                data = json.load(response)
            with open(path, 'w') as f:
                json.dump(data, f)
            time.sleep(0.2)
        with open(path) as f:
            return json.load(f)

    def entry(self, accession):
        return self.get('entry', accession, f'{UNIPROT}/uniprotkb/{accession}.json')

    def citation(self, pubmed):
        return self.get('citation', pubmed, f'{UNIPROT}/citations/{pubmed}.json')


def map_position(source_seq, target_seq, position):
    """0-based position of `source_seq[position]` in `target_seq` under a global alignment, or None."""
    aligner = Align.PairwiseAligner(mode='global', substitution_matrix=substitution_matrices.load('BLOSUM62'),
                                    open_gap_score=-10, extend_gap_score=-0.5)
    alignment = aligner.align(source_seq, target_seq)[0]
    for (s0, s1), (t0, t1) in zip(*alignment.aligned):
        if s0 <= position < s1:
            return t0 + position - s0
    return None


def site_evidence(cache, accession, position, depth=0):
    """[(class, pubmed or None, via)] of the Modified residue features of `accession` at the 0-based
    `position`, following by-similarity evidence into the source entry. None if not annotated."""
    entry = cache.entry(accession)
    features = [f for f in entry.get('features', []) if f['type'] == 'Modified residue'
                and f['location']['start']['value'] == position + 1]
    if not features:
        return None, None
    description = '; '.join(f.get('description', '') for f in features)
    out = []
    for feature in features:
        for ev in feature.get('evidences', []):
            cls = EVIDENCE_CLASS.get(ev.get('evidenceCode'), 'other')
            if ev.get('source') == 'PubMed':
                out.append((cls, ev['id'], ''))
            elif cls == 'by similarity' and ev.get('source') == 'UniProtKB' and depth < 2:
                source = ev['id']
                source_seq = cache.entry(source)['sequence']['value']
                mapped = map_position(entry['sequence']['value'], source_seq, position)
                if mapped is None:
                    out.append((cls, None, f'{source} (unaligned)'))
                    continue
                via = f'{source} {source_seq[mapped]}{mapped + 1}'
                _, inherited = site_evidence(cache, source, mapped, depth + 1)
                if not inherited:
                    out.append((cls, None, via + ' (not annotated there)'))
                for inner_cls, pubmed, inner_via in inherited or []:
                    out.append((f'{cls} -> {inner_cls}', pubmed, via + (f' -> {inner_via}' if inner_via else '')))
            else:
                out.append((cls, None, ev.get('source', '')))
    return description, out


def strongest(classes):
    final = [c.split(' -> ')[-1] for c in classes]
    for name in STRENGTH:
        for full, last in zip(classes, final):
            if last == name:
                return full
    return classes[0] if classes else 'not annotated'


def citation_record(cache, entries, pubmed):
    """Bibliographic record of a PubMed ID, from a fetched entry's references or UniProt's citations."""
    for accession in entries:
        for ref in cache.entry(accession).get('references', []):
            citation = ref['citation']
            if any(x.get('database') == 'PubMed' and x.get('id') == pubmed
                   for x in citation.get('citationCrossReferences', [])):
                return citation
    return cache.citation(pubmed).get('citation', {})


def main():
    parser = argparse.ArgumentParser(description='UniProt literature evidence for the canonical demo sites.')
    parser.add_argument('--info', default='../data/dbptm/dbptm_info.json', help='Unchunked info file (sequences).')
    parser.add_argument('--out_dir', default='../data/demo_eval')
    parser.add_argument('--refresh', action='store_true', help='Fetch every entry and citation again.')
    cli = parser.parse_args()

    cache = Cache(os.path.join(cli.out_dir, 'uniprot'), cli.refresh)
    info = pd.read_json(cli.info).set_index('id')

    site_rows, cited = [], {}
    for accession, (name, specs) in DEMO.items():
        sequence = info.loc[accession, 'sequence']
        if cache.entry(accession)['sequence']['value'] != sequence:
            raise SystemExit(f'{accession}: the UniProt sequence differs from the info file, positions would not match')
        for position, label in resolve_sites(name, sequence, specs):
            description, evidence = site_evidence(cache, accession, position)
            evidence = evidence or []
            # Strongest final evidence first; at equal strength direct evidence before inherited
            classes = sorted({cls for cls, _, _ in evidence}, key=lambda c: (
                STRENGTH.index(c.split(' -> ')[-1]) if c.split(' -> ')[-1] in STRENGTH else len(STRENGTH),
                c.count(' -> '), c))
            pubmeds = list(dict.fromkeys(p for _, p, _ in evidence if p))
            vias = list(dict.fromkeys(v for _, _, v in evidence if v))
            site_rows.append({'protein': name, 'uniprot': accession, 'site': label, 'position': position + 1,
                              'uniprot_description': description or '', 'evidence': strongest(classes) if description else 'not annotated',
                              'evidence_classes': ' | '.join(classes), 'via': ' | '.join(vias),
                              'pubmed': ';'.join(pubmeds), 'n_papers': len(pubmeds)})
            for cls, pubmed, via in evidence:
                if pubmed:
                    record = cited.setdefault(pubmed, {'classes': set(), 'sites': [], 'entries': {accession}})
                    record['classes'].add(cls.split(' -> ')[-1])
                    record['sites'].append(f'{name} {label}')
                    record['entries'].add(via.split(' ')[0] if via else accession)

    ref_rows = []
    for pubmed, record in cited.items():
        citation = citation_record(cache, sorted(record['entries']), pubmed)
        authors = citation.get('authors') or [citation.get('authoringGroup', [''])[0] if citation.get('authoringGroup') else '']
        doi = next((x['id'] for x in citation.get('citationCrossReferences', []) if x.get('database') == 'DOI'), '')
        ref_rows.append({'pubmed': pubmed, 'doi': doi,
                         'first_author': authors[0] if authors else '', 'n_authors': len(citation.get('authors') or []),
                         'year': str(citation.get('publicationDate', ''))[:4], 'title': citation.get('title', ''),
                         'journal': citation.get('journal', ''), 'volume': citation.get('volume', ''),
                         'pages': '-'.join(p for p in (citation.get('firstPage', ''), citation.get('lastPage', '')) if p),
                         'evidence': ' | '.join(c for c in STRENGTH if c in record['classes']),
                         'n_sites': len(set(record['sites'])), 'sites': '; '.join(dict.fromkeys(record['sites']))})

    sites = pd.DataFrame(site_rows)
    refs = pd.DataFrame(ref_rows).sort_values(['year', 'first_author'])
    date = datetime.date.today().isoformat()
    sites['retrieved'] = refs['retrieved'] = date
    os.makedirs(cli.out_dir, exist_ok=True)
    sites.to_csv(os.path.join(cli.out_dir, 'canonical_references.csv'), index=False)
    refs.to_csv(os.path.join(cli.out_dir, 'references.csv'), index=False)

    pd.set_option('display.width', 250)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_colwidth', 60)
    print(sites[['protein', 'site', 'evidence', 'via', 'n_papers']].to_string(index=False))
    print(f'\n{len(refs)} papers; sites by strongest evidence:')
    print(sites['evidence'].value_counts().to_string())
    print(f'\nWritten to {cli.out_dir}/canonical_references.csv and references.csv (UniProt, retrieved {date})')


if __name__ == '__main__':
    main()
