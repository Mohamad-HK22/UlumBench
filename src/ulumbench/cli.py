import argparse
import json
from pathlib import Path
from .data import Dataset, download
from .protocol import available, load_config, make_plan
from .util import canonical


def main(argv=None):
    parser = argparse.ArgumentParser(description='UlumBench evaluation for new models')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('list', help='List installed experiments')
    fetch = commands.add_parser('download', help='Download the pinned dataset snapshot')
    fetch.add_argument('--output', type=Path)
    fetch.add_argument('--revision', help='Explicit HF revision override; recorded through data checksums')
    check = commands.add_parser('validate', help='Verify dataset checksums and counts')
    check.add_argument('--data', type=Path, required=True)
    summary = commands.add_parser('summarize', help='Summarize a complete or interrupted run')
    summary.add_argument('output', type=Path)
    for command in ['plan', 'run']:
        sub = commands.add_parser(command, help='Build a call plan' if command == 'plan' else 'Run a live evaluation')
        sub.add_argument('--config', type=Path, required=True)
        sub.add_argument('--data', type=Path, required=True)
        sub.add_argument('--output', type=Path, required=True)
        sub.add_argument('--limit', type=int, help='Use only the first N calls of the fixed schedule (smoke test)')
        if command == 'run':
            sub.add_argument('--model', required=True, help='Exact OpenRouter model ID')
            sub.add_argument('--judge-model', help='Exact OpenRouter judge ID for open-ended answers')
            sub.add_argument('--provider', help='Optional exact candidate provider endpoint slug')
            sub.add_argument('--retry-errors', action='store_true', help='Retry unresolved provider attempts; may incur another charge')
    args = parser.parse_args(argv)
    if args.command == 'list':
        print('\n'.join(available())); return
    if args.command == 'download':
        print(download(args.revision, args.output)); return
    if args.command == 'summarize':
        from .reporting import summarize
        print(json.dumps(summarize(args.output), indent=2)); return
    data = Dataset(args.data)
    if args.command == 'validate':
        print(json.dumps({'dataset_sha256': data.fingerprint, 'academic_records': len(data.records),
                          'evaluation_records': sum(r['evaluation_eligible'] for r in data.records.values()),
                          'diagnostic_families': len(data.families)}, indent=2)); return
    config = load_config(args.config)
    requests, plan = make_plan(data, config, args.limit)
    if args.command == 'plan':
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / 'requests.jsonl').write_text(''.join(canonical(r) + '\n' for r in requests), encoding='utf-8')
        (args.output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(plan, indent=2)); return
    if plan['maximum_judge_calls'] and not args.judge_model:
        parser.error('--judge-model is required for this selection')
    from .provider import OpenRouter
    from .runner import run
    report = run(data, config, requests, plan, args.output, args.model, args.judge_model,
                 OpenRouter(), args.provider, args.retry_errors)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
