"""UniProt Mutagenesis features at the phosphosites of the demo proteins, the source of their canonical sites.

For every protein of demo_evaluation.DEMO this lists the single-residue Mutagenesis features of its
UniProtKB entry that fall on an annotated phosphosite (a "Phospho..." Modified residue), one row per
mutant. Features spanning several residues (truncations, motif swaps) are left out.

The canonical sites of the demo are the sites labelled functional, by hand, in mutagenesis_labels.csv
(protein, site, functional). A site is functional (1) when a mutant of it, alone or together with
other sites, changes something beyond that site's own phosphorylation: activity, binding,
localisation, stability, transcription, the phosphorylation or modification of other residues. It is
0 when every description is "no effect" / "no change" or only says the mutant loses (or reduces) its
own phosphorylation, e.g. "Abolishes phosphorylation by HCK". Mind descriptions of what the mutant
retains: "No decrease in microtubule-binding after in vitro phosphorylation of mutant protein" (tau
S214A) means the site causes the decrease, so it is 1. Sites without a mutant are not labelled.

    cd model
    python demo_mutagenesis.py            # uses the cached entries of demo_site_references.py
    python demo_mutagenesis.py --refresh  # fetches the entries again

Writes ../data/demo_eval/mutagenesis_descriptions.csv and reports sites that have a description but
no label (to be labelled) and labels without a description (to be removed).
"""
import argparse
import datetime
import os

import pandas as pd

from demo_evaluation import CANONICAL_LABELS, DEMO, demo_sequences
from demo_site_references import Cache


def mutagenesis_rows(entry, name, aliases):
    sequence = entry['sequence']['value']
    phospho = {f['location']['start']['value'] for f in entry.get('features', [])
               if f['type'] == 'Modified residue' and f.get('description', '').startswith('Phospho')}
    rows = []
    for f in entry.get('features', []):
        start, end = f['location']['start']['value'], f['location']['end']['value']
        if f['type'] != 'Mutagenesis' or start != end or start not in phospho:
            continue
        residue = sequence[start - 1]
        alternative = f.get('alternativeSequence', {})
        rows.append({'protein': name, 'uniprot': entry['primaryAccession'], 'site': f'{residue}{start}',
                     'literature_name': f'{residue}{aliases[start]}' if start in aliases else '',
                     'mutation': f"{alternative.get('originalSequence', residue)}{start}"
                                 f"{'/'.join(alternative.get('alternativeSequences', [])) or 'del'}",
                     'description': f.get('description', ''),
                     'pubmed': ';'.join(e['id'] for e in f.get('evidences', []) if e.get('source') == 'PubMed'),
                     'entry_version': entry.get('entryAudit', {}).get('entryVersion', '')})
    return rows


def main():
    parser = argparse.ArgumentParser(description='UniProt mutagenesis descriptions of the demo phosphosites.')
    parser.add_argument('--info', default='../data/dbptm/dbptm_info.json', help='Unchunked info file (sequences).')
    parser.add_argument('--out_dir', default='../data/demo_eval')
    parser.add_argument('--labels', default=CANONICAL_LABELS)
    parser.add_argument('--refresh', action='store_true', help='Fetch every entry again.')
    cli = parser.parse_args()

    cache = Cache(os.path.join(cli.out_dir, 'uniprot'), cli.refresh)
    sequences = demo_sequences(cli.info)
    rows = []
    for accession, (name, aliases) in DEMO.items():
        entry = cache.entry(accession)
        if entry['sequence']['value'] != sequences[accession]:
            raise SystemExit(f'{accession}: the UniProt sequence differs from the info file, positions would not match')
        cached = os.path.join(cache.directory, f'entry_{accession}.json')
        retrieved = datetime.date.fromtimestamp(os.path.getmtime(cached)).isoformat()
        rows += [{**row, 'retrieved': retrieved} for row in mutagenesis_rows(entry, name, aliases)]
    table = pd.DataFrame(rows)
    path = os.path.join(cli.out_dir, 'mutagenesis_descriptions.csv')
    table.to_csv(path, index=False)
    sites = table[['protein', 'site']].drop_duplicates()
    print(f'{len(table)} mutants of {len(sites)} phosphosites of {table["protein"].nunique()} of the '
          f'{len(DEMO)} demo proteins written to {path}')

    if not os.path.exists(cli.labels):
        print(f'no {cli.labels} yet')
        return
    labels = pd.read_csv(cli.labels)
    described = set(map(tuple, sites.values))
    labelled = set(map(tuple, labels[['protein', 'site']].values))
    if described - labelled:
        print('Described but not labelled:', ', '.join(f'{p} {s}' for p, s in sorted(described - labelled)))
    if labelled - described:
        print('Labelled but not described:', ', '.join(f'{p} {s}' for p, s in sorted(labelled - described)))
    if described == labelled:
        print(f'{cli.labels}: every site labelled, {int(labels["functional"].sum())} of {len(labels)} functional')


if __name__ == '__main__':
    main()
