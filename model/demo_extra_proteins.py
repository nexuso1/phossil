"""Adds demo proteins that are missing from the dbPTM info file, with their dbPTM sites and clusters.

Some of the best-studied human phosphoproteins (STAT1, STAT3, SP1, TFEB, ...) are not in
data/dbptm/dbptm_info.json, although dbPTM lists them; only rodent orthologs or paralogs are. For every
protein of demo_evaluation.DEMO missing from the info file this script

  * takes the canonical sequence of its UniProtKB entry (cached in ../data/demo_eval/uniprot/),
  * takes its phosphorylation sites from its current dbPTM entry page (the S/T/Y positions marked
    "Phosphorylation" on the canonical sequence), the database the other proteins' labels come from;
    the online version is newer than the info file, which for TP53 holds 38 of its 41 S/T/Y sites,
  * places it in the existing clusters of every identity level the way the connected-component
    clustering would have: an MMseqs2 search against the info file with the clustering's alignment
    settings (--alignment-mode 3 -s 6 -c 0, e-value 1e-3), and at each level the protein joins the
    cluster of every hit at or above that identity. A protein whose hits fall in several clusters
    belongs to all of them, so demo_evaluation picks a fold that trained on none. The search keeps up
    to --max_seqs (1000) prefilter hits, where the clustering kept 20, so it can only find more
    homologs than the clustering would have, which errs on the side of excluding training folds.

    cd model
    python demo_extra_proteins.py           # needs mmseqs (--mmseqs, or $MMSEQS)

Writes into ../data/demo_eval/:
    extra_proteins.json   info-file rows (id, sequence, sites 1-based, parent_id, offset, ...) plus the
                          dbPTM entry and retrieval date; demo_evaluation appends them to the info file
    extra_clusters.tsv    identity, protein, rep, hit, fident: one row per (level, cluster) joined
    extra_hits.tsv        every MMseqs2 hit, for reference
"""
import argparse
import datetime
import os
import re
import subprocess
import tempfile
import urllib.request

import pandas as pd

from demo_evaluation import CLUSTER_FILES, DEMO, EXTRA_CLUSTERS, EXTRA_PROTEINS
from demo_site_references import Cache

DBPTM_ENTRY = 'https://biomics.lab.nycu.edu.tw/dbPTM/info.php?id={}'
MAX_LENGTH = 1022  # longer proteins would need chunking (scripts/chunk_proteins.py)


def dbptm_phospho_sites(entry_name, sequence, cache_dir):
    """1-based Phosphorylation sites of the canonical sequence in the "Experimental PTM Sites" table of
    the dbPTM entry page, and the rows left out: isoform-specific ones ("715 (in isoform 2)"), and
    those whose highlighted peptide residue is not an S/T/Y matching `sequence` (dbPTM lists a few
    phosphorylations on other residues)."""
    path = os.path.join(cache_dir, f'dbptm_{entry_name}.html')
    if not os.path.exists(path):
        with urllib.request.urlopen(DBPTM_ENTRY.format(entry_name), timeout=120) as response:
            with open(path, 'wb') as f:
                f.write(response.read())
    with open(path, encoding='utf-8', errors='replace') as f:
        html = f.read()
    table = html[html.find('id="exp"'):html.find('id="upstream"')]
    kept, dropped = set(), []
    for row in re.findall(r"<tr align='center'>(.*?)</tr>", table, flags=re.S):
        cells = re.findall(r'<td[^>]*>(.*?)</td>', row, flags=re.S)
        if len(cells) < 3 or not re.sub(r'<[^>]+>', '', cells[1]).strip().startswith('Phosphorylation'):
            continue
        location = re.sub(r'<[^>]+>', '', cells[0]).strip()
        bold = re.search(r'<b>([A-Z])</b>', cells[2])
        if not location.isdigit():
            dropped.append(location)
            continue
        position = int(location)
        residue = sequence[position - 1] if position <= len(sequence) else '?'
        if residue in 'STY' and (bold is None or bold.group(1) == residue):
            kept.add(position)
        else:
            dropped.append(f'{location} ({bold.group(1) if bold else "?"})')
    if not kept:
        raise SystemExit(f'no phosphorylation site parsed from the dbPTM page of {entry_name} ({path})')
    return sorted(kept), dropped, path


def search(mmseqs, sequences, target_fasta, max_seqs, threads):
    with tempfile.TemporaryDirectory() as tmp:
        query = os.path.join(tmp, 'query.fasta')
        with open(query, 'w') as f:
            for pid, seq in sequences.items():
                f.write(f'>{pid}\n{seq}\n')
        out = os.path.join(tmp, 'hits.m8')
        subprocess.run([mmseqs, 'easy-search', query, target_fasta, out, os.path.join(tmp, 'tmp'),
                        '--alignment-mode', '3', '-s', '6', '-c', '0', '-e', '1e-3', '--max-seqs', str(max_seqs),
                        '--threads', str(threads),
                        '--format-output', 'query,target,fident,alnlen,qcov,tcov,evalue,bits'],
                       check=True, stdout=subprocess.DEVNULL)
        return pd.read_csv(out, sep='\t', names=['protein', 'hit', 'fident', 'alnlen', 'qcov', 'tcov', 'evalue', 'bits'])


def main():
    parser = argparse.ArgumentParser(description='Demo proteins missing from the dbPTM info file.')
    parser.add_argument('--info', default='../data/dbptm/dbptm_info.json', help='Unchunked info file.')
    parser.add_argument('--fasta', default='../data/dbptm/dbptm_info.fasta', help='The info file as FASTA (search target).')
    parser.add_argument('--out_dir', default='../data/demo_eval')
    parser.add_argument('--mmseqs', default=os.environ.get('MMSEQS', os.path.expanduser('~/miniforge3/envs/mmseqs2/bin/mmseqs')))
    parser.add_argument('--max_seqs', type=int, default=1000)
    parser.add_argument('--threads', type=int, default=os.cpu_count())
    cli = parser.parse_args()

    present = set(pd.read_json(cli.info)['id'])
    missing = [pid for pid in DEMO if pid not in present]
    if not missing:
        print('every demo protein is in the info file, nothing to add')
        return
    cache = Cache(os.path.join(cli.out_dir, 'uniprot'), refresh=False)
    today = datetime.date.today().isoformat()
    rows, sequences = [], {}
    for pid in missing:
        entry = cache.entry(pid)
        sequence = entry['sequence']['value']
        if len(sequence) > MAX_LENGTH:
            raise SystemExit(f'{pid} has {len(sequence)} residues; chunk it like chunk_proteins.py first')
        sites, dropped, page = dbptm_phospho_sites(entry['uniProtkbId'], sequence, cache.directory)
        sequences[pid] = sequence
        rows.append({'id': pid, 'sequence': sequence, 'sites': sites, 'parent_id': pid, 'chunk_index': 0,
                     'offset': 0, 'parent_length': len(sequence), 'dbptm_entry': entry['uniProtkbId'],
                     'sequence_version': entry.get('entryAudit', {}).get('sequenceVersion'),
                     'retrieved': datetime.date.fromtimestamp(os.path.getmtime(page)).isoformat()})
        note = f'; left out (isoform-specific or not a matching S/T/Y): {", ".join(dropped)}' if dropped else ''
        print(f'{DEMO[pid][0]:7} {pid}: {len(sequence)} residues, {len(sites)} dbPTM phosphosites{note}')

    hits = search(cli.mmseqs, sequences, cli.fasta, cli.max_seqs, cli.threads)
    clusters = []
    for identity, path in CLUSTER_FILES.items():
        rep_of = pd.read_csv(path, sep='\t', header=None, names=['rep', 'mem']).drop_duplicates('mem').set_index('mem')['rep']
        close = hits[hits['fident'] >= identity / 100].copy()
        close['rep'] = close['hit'].map(rep_of)
        if close['rep'].isna().any():
            raise SystemExit(f'hits without a cluster in {path}: {sorted(close.loc[close["rep"].isna(), "hit"])}')
        best = close.sort_values('bits', ascending=False).drop_duplicates(['protein', 'rep'])
        clusters.append(best.assign(identity=identity)[['identity', 'protein', 'rep', 'hit', 'fident']])
    clusters = pd.concat(clusters, ignore_index=True)

    os.makedirs(cli.out_dir, exist_ok=True)
    pd.DataFrame(rows).to_json(os.path.join(cli.out_dir, os.path.basename(EXTRA_PROTEINS)), orient='records', indent=1)
    clusters.to_csv(os.path.join(cli.out_dir, os.path.basename(EXTRA_CLUSTERS)), sep='\t', index=False)
    hits.to_csv(os.path.join(cli.out_dir, 'extra_hits.tsv'), sep='\t', index=False)

    for pid in missing:
        line = []
        for identity in CLUSTER_FILES:
            c = clusters[(clusters['protein'] == pid) & (clusters['identity'] == identity)]
            line.append(f'{identity} %: ' + (', '.join(f'{r.rep} (via {r.hit} {r.fident:.2f})' for r in c.itertuples())
                                            if len(c) else 'no hit, own cluster'))
        print(f'{DEMO[pid][0]:7} ' + ' | '.join(line))
    print(f'Written to {cli.out_dir}/: {os.path.basename(EXTRA_PROTEINS)}, {os.path.basename(EXTRA_CLUSTERS)}, extra_hits.tsv '
          f'({today})')


if __name__ == '__main__':
    main()
