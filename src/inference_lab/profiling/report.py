"""Render an offline, reproducible DFlash latency report from raw measurements.

No model or MLX runtime is imported. Nested timings are never added twice and
instrumented measurements remain distinct from production throughput.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import statistics

PHASES = ('draft', 'verify', 'acceptance_commit', 'rollback')
LABELS = {'draft': 'Draft', 'verify': 'Target verify', 'acceptance_commit': 'Host acceptance', 'rollback': 'Rollback'}
ROOT = Path(__file__).resolve().parents[3]


def stats(values):
    values = list(values)
    return {'n': len(values), 'mean': statistics.mean(values),
            'stdev': statistics.stdev(values) if len(values) > 1 else None,
            'min': min(values), 'max': max(values)} if values else None


def f(value, digits=2):
    return '—' if value is None else f'{value:.{digits}f}'


def pm(value, digits=2):
    if not value:
        return '—'
    return f"{f(value['mean'], digits)} ± {f(value['stdev'], digits)}"


def tensor_leaves(value):
    if isinstance(value, dict):
        if 'shape' in value and 'dtype' in value:
            yield value
        else:
            for child in value.values():
                yield from tensor_leaves(child)
    elif isinstance(value, list):
        for child in value:
            yield from tensor_leaves(child)


def shapes(value, limit=4):
    leaves = list(tensor_leaves(value))
    result = '; '.join('[' + ','.join(str(d) for d in x['shape']) + '] ' + x['dtype'].removeprefix('mlx.core.') for x in leaves[:limit])
    return result + ('; …' if len(leaves) > limit else '')


def events(row, kind=None, steady=None):
    for event in row.get('profile', {}).get('spans', []):
        meta = event['metadata']
        if kind is not None and meta.get('kind') != kind:
            continue
        if steady is not None and (meta.get('round', 1) > 1) != steady:
            continue
        yield event


def _operator_label(event):
    meta = event['metadata']
    phase = meta['phase']
    path = meta.get('path', event['name'])
    actor = 'draft' if phase == 'draft' else 'target'
    if meta.get('kind') == 'operation':
        return f"{phase}.{meta['operation']}"
    if meta.get('class') != 'QuantizedLinear':
        return None
    if path.endswith('.lm_head'):
        return f'{actor}.lm_head'
    if path.endswith('.fc'):
        return f'{actor}.context_fc'
    if '.mlp.' in path:
        return f"{actor}.MLP.{path.rsplit('.', 1)[-1]}"
    return f"{actor}.{path.rsplit('.', 1)[-1]}"


def build_summary(raw):
    runs = raw['runs']
    repetitions = raw['protocol']['repeats']
    primary = [r for r in runs if r['mode'] in ('ar', 'stock') and r['repeat'] < repetitions]
    ks = sorted({r['block_size'] for r in runs if r['mode'] == 'stock'})
    summary = {'schema_version': 1, 'protocol': raw['protocol'], 'backend': raw['backend'],
               'started': raw['started'], 'finished': raw.get('finished'),
               'upstream_source_sha256': raw['upstream_source_sha256'],
               'all_token_ids_match': all(r['matches_baseline'] for r in runs),
               'run_count': len(runs), 'primary': {}, 'paired_by_prompt': [],
               'bracket': [], 'phases': {}, 'instrumentation': [], 'layers': {}, 'operators': {}, 'input_materialization': {},
               'batching_probe': raw.get('batching_probe')}
    for mode, k in [('ar', 1)] + [('stock', k) for k in ks]:
        rows = [r for r in primary if r['mode'] == mode and r['block_size'] == k]
        summary['primary'][f'{mode}_k{k}'] = {
            'decode_ms': stats(r['measurement']['decode_seconds'] * 1000 for r in rows),
            'ms_per_token': stats(r['measurement']['decode_seconds'] * 1000 / r['measurement']['decode_tokens'] for r in rows),
            'tok_s': stats(r['measurement']['decode_tokens'] / r['measurement']['decode_seconds'] for r in rows),
            'prefill_ms': stats(r['measurement']['prefill_seconds'] * 1000 for r in rows)}
        if mode == 'stock':
            summary['primary'][f'{mode}_k{k}'].update({
                'acceptance': sum(r['measurement']['accepted_draft_tokens'] for r in rows) / sum(r['measurement']['drafted_tokens'] for r in rows),
                'useful_per_round': sum(r['measurement']['decode_tokens'] for r in rows) / sum(r['measurement']['speculative_rounds'] for r in rows),
                'ms_per_round': sum(r['measurement']['decode_seconds'] * 1000 for r in rows) / sum(r['measurement']['speculative_rounds'] for r in rows)})
    for p in sorted({r['prompt_index'] for r in primary}):
        baseline = {r['repeat']: r for r in primary if r['mode'] == 'ar' and r['prompt_index'] == p}
        for k in ks:
            rows = [r for r in primary if r['mode'] == 'stock' and r['block_size'] == k and r['prompt_index'] == p]
            ratios = [baseline[r['repeat']]['measurement']['decode_seconds'] / r['measurement']['decode_seconds'] for r in rows]
            summary['paired_by_prompt'].append({'prompt_index': p, 'block_size': k, 'speedup': stats(ratios)})
    for row in runs:
        m = row['measurement']
        if row['mode'] in ('ar', 'stock') and row['repeat'] >= repetitions:
            previous = [r for r in primary if r['mode'] == row['mode'] and r['block_size'] == row['block_size'] and r['prompt_index'] == row['prompt_index']]
            previous_ms = statistics.mean(r['measurement']['decode_seconds'] * 1000 for r in previous)
            summary['bracket'].append({'mode': row['mode'], 'block_size': row['block_size'], 'prompt_index': row['prompt_index'],
                                      'before_ms': previous_ms, 'after_ms': m['decode_seconds'] * 1000,
                                      'after_over_before': m['decode_seconds'] * 1000 / previous_ms})
        if row['mode'] in ('phases', 'layers', 'operators'):
            paired = [r for r in primary if r['mode'] == 'stock' and r['block_size'] == row['block_size'] and r['prompt_index'] == row['prompt_index']]
            stock_ms = statistics.mean(r['measurement']['decode_seconds'] * 1000 for r in paired)
            phase_row = next(r for r in runs if r['mode'] == 'phases' and r['block_size'] == row['block_size'] and r['prompt_index'] == row['prompt_index'])
            summary['instrumentation'].append({'mode': row['mode'], 'block_size': row['block_size'], 'prompt_index': row['prompt_index'],
                'decode_ms': m['decode_seconds'] * 1000, 'stock_ms': stock_ms,
                'over_stock': m['decode_seconds'] * 1000 / stock_ms,
                'over_phases': m['decode_seconds'] / phase_row['measurement']['decode_seconds'],
                'unattributed_ms': row['profile']['unattributed_decode_ms'],
                'spans': len(row['profile']['spans'])})
    for k in ks:
        rows = [r for r in runs if r['mode'] == 'phases' and r['block_size'] == k]
        es = [e for r in rows for e in events(r, 'phase')]
        phase_summary = {category: {phase: stats(e['inclusive_ms'] for e in es if e['name'] == phase and
                          (category == 'all' or (e['metadata']['round'] == 1) == (category == 'first')))
                          for phase in PHASES} for category in ('all', 'first', 'steady')}
        steady_commits = [e for e in es if e['name'] == 'acceptance_commit' and e['metadata']['round'] > 1]
        rollback_events = [e for e in es if e['name'] == 'rollback']
        rejected = [e for e in rollback_events if e['metadata']['trimmed'] > 0]
        phase_summary.update({'useful_per_steady_round': statistics.mean(e['metadata']['emitted'] for e in steady_commits),
                              'rollback_all': stats(e['inclusive_ms'] for e in rollback_events),
                              'rollback_reject_only': stats(e['inclusive_ms'] for e in rejected),
                              'rejection_round_fraction': len(rejected) / len(rollback_events),
                              'unattributed_ms': stats(r['profile']['unattributed_decode_ms'] for r in rows)})
        summary['phases'][str(k)] = phase_summary
        layer_groups = defaultdict(list)
        for row in [r for r in runs if r['mode'] == 'layers' and r['block_size'] == k]:
            for event in events(row, 'module', steady=True):
                layer_groups[event['name']].append(event)
        summary['layers'][str(k)] = [{'path': path, 'time_ms': stats(e['inclusive_ms'] for e in ev),
                                      'inputs': ev[0]['metadata']['inputs'], 'output': ev[0]['metadata'].get('output')}
                                     for path, ev in sorted(layer_groups.items(), key=lambda pair: pair[0])]
        preparation = defaultdict(lambda: defaultdict(float))
        for row in [r for r in runs if r['mode'] == 'operators' and r['block_size'] == k]:
            for event in events(row, steady=True):
                meta = event['metadata']
                if meta.get('kind') == 'input_materialization':
                    preparation[meta['phase']][meta['round']] += event['exclusive_ms']
        summary['input_materialization'][str(k)] = {phase: stats(costs.values()) for phase, costs in preparation.items()}
        op_groups = defaultdict(list)
        op_rounds = defaultdict(set)
        for row in [r for r in runs if r['mode'] == 'operators' and r['block_size'] == k]:
            for event in events(row, steady=True):
                phase = event['metadata'].get('phase')
                if event['metadata'].get('kind') == 'phase' and event['metadata'].get('detail_enabled'):
                    op_rounds[phase].add((row['prompt_index'], event['metadata']['round']))
                label = _operator_label(event) if event['metadata'].get('kind') in ('module', 'operation') else None
                if label:
                    op_groups[label].append(event)
        op_rows = []
        for label, ev in op_groups.items():
            # Only quantized-linear leaves and directly measured kernels enter
            # this table. Their inclusive costs do not contain each other.
            by_round = defaultdict(float)
            for e in ev:
                by_round[e['metadata']['round']] += e['inclusive_ms']
            phase = ev[0]['metadata']['phase']
            rounds = {r for _, r in op_rounds[phase]}
            costs = [by_round.get(r, 0.0) for r in sorted(rounds)]
            sample = max(ev, key=lambda e: e['inclusive_ms'])
            op_rows.append({'group': label, 'phase': phase, 'calls': len(ev), 'time_per_call_ms': stats(e['inclusive_ms'] for e in ev),
                'time_per_detailed_round_ms': stats(costs), 'example': sample['metadata']})
        summary['operators'][str(k)] = sorted(op_rows, key=lambda x: x['time_per_detailed_round_ms']['mean'], reverse=True)
    return summary


def _table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |'] +
                     ['| ' + ' | '.join(str(v).replace('|', '\\|').replace('\n', ' ') for v in row) + ' |' for row in rows])


def markdown(raw, s, markdown_path, figure_path, data_path, raw_path):
    def link(path):
        return Path(os.path.relpath(path, markdown_path.parent)).as_posix()
    ks = sorted(int(k) for k in s['phases'])
    protocol = raw['protocol']
    baseline = s['primary']['ar_k1']['ms_per_token']['mean']
    lines = ['# Куда уходит время DFlash на Apple Silicon', '',
        'Короткий контролируемый профиль Qwen3.5-9B-4bit и 4-bit DFlash. Реальная скорость взята из прогонов без профилировщика; этапы, слои и операторы измерены отдельными прогонами.', '',
        f"Устройство: **{raw['backend']['device_info']['device_name']}**. {len(protocol['prompts'])} промпта, {protocol['tokens']} сгенерированных токенов на прогон, {protocol['repeats']} повтора основной пары AR/DFlash. Первый токен относится к prefill; decode содержит {protocol['tokens'] - 1} токенов. Детальные раунды: {', '.join(map(str, protocol['detail_rounds']))}.", '',
        f"Все token IDs совпали с AR во всех {len(raw['runs'])} прогонах: **{'да' if s['all_token_ids_match'] else 'НЕТ'}**. EOS игнорируется, sampling greedy, кэш свежий на каждый запрос. Время декодирования текста в измерения не входит.", '',
        f'![DFlash latency profile]({link(figure_path)})', '',
        '## Проверка гипотезы о быстрой верификации', '',
        'Ускорение получается, если `T_draft + T_verify + T_rollback + T_host < (accepted + 1) × T_AR`. Здесь `K` — длина проверяемого блока: один уже известный токен и до `K−1` предложений draft. Это позиции одной последовательности при batch size 1, а не K независимых запросов. При отказе оплачивается проверка всего блока, а полезен только принятый префикс и исправляющий токен target.', '',
        '## Скорость без инструментирования', '',
        'Основная таблица исключает заключительную контрольную пару. ± — выборочное стандартное отклонение между прогонами, а не доверительный интервал; повторения одного промпта не являются новыми независимыми задачами.', '']
    rows = []
    for mode, k in [('ar', 1)] + [('stock', k) for k in ks]:
        item = s['primary'][f'{mode}_k{k}']
        rows.append(['AR' if mode == 'ar' else f'DFlash K={k}', item['decode_ms']['n'], pm(item['tok_s']), pm(item['ms_per_token']),
                     '—' if mode == 'ar' else f"{100 * item['acceptance']:.1f}%", '—' if mode == 'ar' else f(item['useful_per_round'])])
    lines += [_table(['Режим', 'n', 'tok/s', 'мс / токен', 'Принято draft', 'Полезных / раунд'], rows), '',
              'Парное ускорение — отношение времени AR к DFlash на том же промпте в том же повторе; >1 означает ускорение.', '',
              _table(['Промпт (индекс)', 'K', 'AR / DFlash, среднее ± SD'], [[r['prompt_index'], r['block_size'], pm(r['speedup'], 3)] for r in s['paired_by_prompt']]), '',
              '## Этапы одного раунда', '',
              'Только режим `phases`: границы принудительно материализуют lazy MLX-граф и синхронизируют поток. Это диагностическая декомпозиция с дополнительными барьерами. Первый draft отдельно проецирует признаки всего промпта и наполняет свой KV-кэш; эту разовую работу исходный генератор выполняет уже после первого target-токена, внутри decode.', '']
    lines += ['Сравнение без профилировщика: суммарное decode-время делится на суммарное число раундов, а полезные токены — на то же число раундов. Это объясняет реальную разницу без смешивания с синхронизированным профилем:', '', _table(['K', 'Stock DFlash, мс / раунд', 'Полезных / раунд', 'AR для того же числа токенов, мс'], [[k, f(s['primary'][f'stock_k{k}']['ms_per_round']), f(s['primary'][f'stock_k{k}']['useful_per_round']), f(baseline*s['primary'][f'stock_k{k}']['useful_per_round'])] for k in ks]), '', 'Чёрная отметка AR на графике фаз — ориентир из другого протокола. Разницу с инструментированным столбцом нельзя выдавать за точный прогноз production-ускорения.', '']
    rows = []
    for k in ks:
        for category, label in [('first', 'Первый'), ('steady', 'После первого')]:
            p = s['phases'][str(k)][category]
            rows.append([k, label] + [pm(p[name]) for name in PHASES] + [f(sum(p[name]['mean'] for name in PHASES))])
    lines += [_table(['K', 'Раунд', 'Draft, мс', 'Verify, мс', 'Host, мс', 'Rollback, мс', 'Сумма средних, мс'], rows), '',
              'Средние по раундам, поэтому промпты с большим числом раундов получают больший вес. Последний неполный блок также включён. Остаток между decode wall time и суммой корневых этапов — управление генератором и профилировщиком вне границ, а не спрятанный GPU-оператор.', '',
              _table(['K', 'Rollback все раунды, мс', 'Только при trim>0, мс', 'Доля trim>0', 'Остаток / прогон, мс'], [[k, pm(s['phases'][str(k)]['rollback_all']), pm(s['phases'][str(k)]['rollback_reject_only']), f"{s['phases'][str(k)]['rejection_round_fraction']*100:.1f}%", pm(s['phases'][str(k)]['unattributed_ms'])] for k in ks]), '',
              'Rollback повторяет только recurrent `gated_delta_update` для принятого префикса в 24 GDN-слоях и обрезает attention-кэши. Он не запускает заново полные target-слои с MLP. Даже при trim=0 остаётся цена границы профиля и проверки состояния.', '',
              '## Сколько даёт сама проверка блока', '']
    probe = s['batching_probe']
    if probe:
        lines += [f"Один фиксированный префикс длиной {probe['prefix_tokens']} токенов; {probe['repeats']} парных повторов для каждой ширины. Обе ветки получают одни и те же следующие токены реальной AR-траектории. Кэш клонируется до таймера; его исходный снимок проверяется на неизменность. Block включает production GDN capture, hidden concat и host transfer. Sequential выполняет K законченных однотокенных шагов с обычным GDN; feature hooks остаются, concat не выполняется.", '',
            _table(['K', 'K последовательных, мс', 'Проверка блока, мс', 'Отношение seq / block', 'Идеальный остаток для draft + остального, мс', 'Argmax совпал', 'Δ cache max'], [[w['width'], pm({'mean': w['sequential_ms']['mean'], 'stdev': w['sequential_ms']['stdev']}), pm({'mean': w['block_ms']['mean'], 'stdev': w['block_ms']['stdev']}), f(w['speedup']['mean'], 3), f(w['sequential_ms']['mean']-w['block_ms']['mean']), 'да' if w['all_argmax_equal'] else 'НЕТ', f(w['max_cache_abs_error'], 6)] for w in probe['widths']]), '',
            f"Конечные кэши сравнены отдельно с абсолютным допуском {probe['cache_comparison_atol']}; все в допуске: **{'да' if all(w['all_cache_within_atol'] for w in probe['widths']) else 'нет'}**. Равенство argmax не означает побитового равенства внутренних recurrent состояний.", '',
            'Sequential в этом probe завершает и материализует каждый шаг, включая кэш; production AR использует другой граф выполнения и pipeline. Поэтому его цена одного токена может отличаться от основной AR-таблицы. Нельзя подставлять этот sequential вместо production AR при предсказании end-to-end ускорения. Это верхняя оценка выгоды batching на известном продолжении: здесь нет draft, отказов и rollback. Разность `K sequential − block` — бюджет только при принятии всего блока. Реальный бюджет меньше и зависит от числа полезных токенов. Нельзя выдавать это отношение за итоговое ускорение DFlash.', '']
    else:
        lines += ['В данном файле эксперимент фиксированного префикса отсутствует.', '']
    lines += ['## Где находятся дорогие слои', '',
              'Следующая таблица использует отдельный режим `layers`, только детальные раунды после первого. Режимы `layers` и `operators` в этом протоколе измеряют промпт с индексом 0; phase-профиль охватывает все промпты. У каждого decoder layer материализуется выход; `fc` и vocabulary head измеряются отдельно. Некоторые lazy-побочные записи кэша, например contiguous-копия convolution state GDN, могут завершиться лишь в общей границе фазы: там отдельно материализуются все кэши. Полное время фазы их включает, но таблица слоёв не распределяет каждую cache-write операцию по её источнику. Inclusive-время слоя включает его внутренние операции. Эти числа нельзя складывать с вложенными operator-временами из следующего раздела.', '']
    paths = sorted({row['path'] for rows in s['layers'].values() for row in rows}, key=lambda x: (x.split('.')[0], int(re.search(r'\.layers\.(\d+)', x).group(1)) if re.search(r'\.layers\.(\d+)', x) else -1, x))
    layer_maps = {k: {r['path']: r for r in s['layers'][str(k)]} for k in ks}
    lines += [_table(['Слой / модуль'] + [f'K={k}, мс ± SD' for k in ks], [[f'`{path}`'] + [pm(layer_maps[k].get(path, {}).get('time_ms')) for k in ks] for path in paths]), '',
              '## Операторы и формы тензоров', '',
              'Режим `operators` материализует входы до каждой измеряемой операции, затем выход и синхронизирует поток. В таблице приведены только QuantizedLinear и отдельно обёрнутые attention/GDN-операции: их inclusive-времена не вложены друг в друга. «Сумма / раунд» складывает одинаковые операции по всем слоям внутри измеренных раундов; это стоимость в сильно инструментированном исполнении, **не процент времени stock**. SD относится к доступным детальным раундам, которых мало.', '']
    for k in ks:
        rows = s['operators'][str(k)]
        lines += [f'### K={k}: измеренные операции после первого раунда', '',
            _table(['Группа', 'Вызовов', 'Один вызов, мс ± SD', 'Сумма / раунд, мс ± SD', 'Пример входа → выхода', 'Логическая W'], [[f"`{r['group']}`", r['calls'], pm(r['time_per_call_ms'], 3), pm(r['time_per_detailed_round_ms']), f"`{shapes(r['example']['inputs'], 3)} → {shapes(r['example'].get('output'), 2)}`", str(r['example'].get('logical_weight_shape', '—'))] for r in rows]), '']
    first_fcs = [(r['block_size'], e) for r in raw['runs'] if r['mode'] == 'layers' for e in events(r, 'module', steady=False) if e['name'] == 'draft.fc']
    lines += ['Отдельно записанные `.inputs`-интервалы материализуют ожидающие входные выражения: активации, residual/reshape и другую upstream lazy-работу, а также синхронизацию. Это не только копирование входов и не чистые накладные расходы профилировщика. Ниже сумма exclusive-времён этих непересекающихся spans на детальный раунд после первого.', '', _table(['K', 'Фаза', 'Подготовка входов, мс / раунд ± SD'], [[k, phase, pm(value)] for k in ks for phase, value in s['input_materialization'][str(k)].items()]), '']
    lines += ['### Почему эти тензоры важны', '',
        '- Draft использует шесть transformer-слоёв. Признаки восьми target-слоёв конкатенируются по hidden dimension: 8 × 4096 = 32768. `draft.fc` проецирует 32768 → 4096. В первом раунде длина контекста равна всему промпту; после него — только принятому префиксу последнего блока.',
        '- Общая с target vocabulary head проецирует hidden 4096 → 248320. В draft она обрабатывает до K−1 позиций, в verifier — K. Q4 уменьшает хранение весов, но не устраняет работу над большим словарём. Полный логитный тензор: `[1, K−1, 248320]` или `[1, K, 248320]`.',
        '- Target состоит из 32 decoder-слоёв: 24 recurrent GDN и 8 full-attention. GDN state имеет форму `[1,32,128,128]` и dtype float32. В установленном `mlx_lm/models/gated_delta.py` Metal-kernel выполняет цикл `for (int t = 0; t < T; ++t)`: recurrent update внутри блока остаётся последовательным по токенам; линейные проекции и MLP при этом могут обрабатывать позиции вместе.',
        '- Draft MLP имеет промежуточную размерность 12288: gate/up `[1,K,4096] → [1,K,12288]`, down возвращает 4096. В таблицах фактические формы и packed Q4-веса сохранены в JSON, поэтому большие vocabulary-head и MLP-проекции отделены от самого attention.', '',
        _table(['Первый draft.fc', 'Вход', 'Выход', 'мс'], [[f"K={k}, раунд {e['metadata']['round']}", f"`{shapes(e['metadata']['inputs'])}`", f"`{shapes(e['metadata'].get('output'))}`", f(e['inclusive_ms'])] for k, e in first_fcs]), '',
        '## Насколько профилировщик меняет исполнение', '',
        'Это synchronized wall time: Python, создание lazy-графа, его выполнение, ожидание Metal и барьеры включены. Это **не Metal GPU kernel timestamps**. В частности, `entry_sync_ms`, `exit_sync_ms` и `materialize_ms` уже входят в длительность соответствующего span; их нельзя прибавлять ещё раз или целиком называть накладными расходами профилировщика. Корректная оценка вмешательства — отдельное сравнение с stock ниже.', '',
        _table(['Режим', 'K', 'Промпт', 'Decode, мс', 'К stock того же промпта', 'К phases', 'Spans'], [[r['mode'], r['block_size'], r['prompt_index'], f(r['decode_ms']), f(r['over_stock'], 3)+'×', f(r['over_phases'], 3)+'×', r['spans']] for r in s['instrumentation']]), '',
        'Заключительная контрольная пара показывает дрейф после диагностики. Она сохранена, но не подмешана в основные средние:', '',
        _table(['Режим', 'K', 'До, мс', 'После, мс', 'После / до'], [[r['mode'], r['block_size'], f(r['before_ms']), f(r['after_ms']), f(r['after_over_before'], 3)+'×'] for r in s['bracket']]), '',
        '## Ограничения и воспроизведение', '',
        'Это короткий диагностический пилот на выбранных промптах, а не оценка длинных цепочек или всего датасета. Небольшое преимущество нельзя уверенно переносить на 2048+ токенов, другую температуру Mac или другое питание. Длинные нагрузочные прогоны остаются отложенными по просьбе пользователя. Состояния питания и thermal pressure до/после, версии библиотек, SHA исходника, промпты, все token IDs и полное дерево timings сохранены.', '',
        f"- [Компактные измерения и агрегаты]({link(data_path)})\n- [Полный сырой профиль, JSON gzip]({link(raw_path)})\n- Upstream DFlash: [`{raw['backend']['upstream_commit'][:12]}`](https://github.com/z-lab/dflash/tree/{raw['backend']['upstream_commit']})\n- SHA256 model_mlx.py: `{raw['upstream_source_sha256']}`", '',
        '```bash\nbash scripts/setup/speculative.sh\nbash scripts/download/draft.sh\nbash scripts/setup/analysis.sh\nbash scripts/benchmark/dflash_profile.sh --output artifacts/profiling/my-dflash-profile\nbash scripts/analysis/dflash_profile_report.sh --input artifacts/profiling/my-dflash-profile --output docs/results/my-dflash-profile\n```', '',
        'Скрипт отчёта работает офлайн, не загружает MLX и не запускает модель. Полные исходные span сохраняют parent_id; сумма всех inclusive-времён дерева двойным счётом завышала бы стоимость. Для непересекающейся суммы используются корневые этапы или exclusive-времена.', '']
    return '\n'.join(lines)


def plot(s, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    ks = sorted(int(k) for k in s['phases'])
    colors = ['#6887ed', '#32ae98', '#f1bd55', '#df7195']
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), constrained_layout=True)
    fig.suptitle('DFlash on Apple Silicon: where the decode time goes', fontsize=18, fontweight='bold')
    ax = axes[0, 0]
    mode_keys = ['ar_k1'] + [f'stock_k{k}' for k in ks]
    values = [s['primary'][key]['ms_per_token'] for key in mode_keys]
    ax.bar(range(len(values)), [v['mean'] for v in values], yerr=[v['stdev'] or 0 for v in values], capsize=5, color=['#354456']+colors[:len(ks)])
    ax.set(xticks=range(len(values)), xticklabels=['AR']+[f'DFlash K={k}' for k in ks], ylabel='Milliseconds / generated decode token', title='Production runs: no profiling barriers')
    for i, v in enumerate(values):
        ax.text(i, v['mean'] + (v['stdev'] or 0) + 0.5, f"{v['mean']:.2f}", ha='center')
    ax.set_ylim(0, max(v['mean']+(v['stdev'] or 0) for v in values)*1.2)
    ax.text(0.01, 0.96, 'Error bars: sample SD across paired runs', transform=ax.transAxes, va='top', fontsize=9, color='#536273')
    ax = axes[0, 1]
    x = np.arange(len(ks)); bottom = np.zeros(len(ks))
    for phase, color in zip(PHASES, colors):
        values = np.array([s['phases'][str(k)]['steady'][phase]['mean'] for k in ks])
        ax.bar(x, values, bottom=bottom, color=color, label=LABELS[phase], width=.6)
        bottom += values
    baseline = s['primary']['ar_k1']['ms_per_token']['mean']
    budgets = [baseline*s['phases'][str(k)]['useful_per_steady_round'] for k in ks]
    ax.scatter(x, budgets, marker='_', s=1100, linewidths=3, color='#222e40', label='Unprofiled AR reference (different protocol)', zorder=4)
    ax.set(xticks=x, xticklabels=[f'K={k}' for k in ks], ylabel='Milliseconds / round', title='Phase profile: extra barriers, after round 1')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_ylim(0,max(list(bottom)+budgets)*1.38)
    for i, total in enumerate(bottom):
        ax.text(i,total+2,f'{total:.1f} ms',ha='center',fontsize=9)
    ax = axes[1, 0]
    probe = s.get('batching_probe')
    if probe:
        ws = probe['widths']; widths = [w['width'] for w in ws]
        seq = [w['sequential_ms']['mean'] for w in ws]; block = [w['block_ms']['mean'] for w in ws]
        ax.errorbar(widths, seq, yerr=[w['sequential_ms']['stdev'] for w in ws], marker='o', capsize=4, color='#354456', label='K sequential target steps')
        ax.errorbar(widths, block, yerr=[w['block_ms']['stdev'] for w in ws], marker='s', capsize=4, color='#32ae98', label='One K-position verification (no draft)')
        ax.fill_between(widths, block, seq, color='#32ae98', alpha=.13, label='Maximum budget for draft + overhead')
        ax.set(xticks=widths, xlabel='Block width K', ylabel='Synchronized wall time, ms', title='Diagnostic same-prefix batching ceiling')
        ax.legend(fontsize=8)
    else:
        ax.text(.5,.5,'Batching probe unavailable',ha='center',transform=ax.transAxes)
    ax = axes[1, 1]
    groups = [mode for mode in ('phases', 'layers', 'operators') if any(r['mode'] == mode for r in s['instrumentation'])]; vals=[]
    for mode in groups:
        vals.append([statistics.mean(r['over_stock'] for r in s['instrumentation'] if r['mode']==mode and r['block_size']==k) for k in ks])
    w=.7 / max(1, len(groups))
    for i,(mode,v) in enumerate(zip(groups,vals)):
        ax.bar(np.arange(len(ks))+(i-(len(groups)-1)/2)*w,v,width=w,label=mode,color=colors[i])
    ax.axhline(1,color='#354456',linewidth=1,linestyle='--')
    ax.set(xticks=np.arange(len(ks)),xticklabels=[f'K={k}' for k in ks],ylabel='Instrumented decode time / stock',title='Measurement changes the workload')
    ax.legend(fontsize=8)
    ax.text(.01,.98,'Fine barriers only on selected rounds\nDiagnostic costs ≠ production GPU kernel times',va='top',transform=ax.transAxes,fontsize=9,color='#536273')
    ax.set_ylim(0,max(max(v) for v in vals)*1.4)
    for ax in axes.flat:
        ax.grid(axis='y',alpha=.16)
        ax.set_axisbelow(True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=170, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Completed run folder or raw.json.gz')
    parser.add_argument('--output', type=Path, required=True, help='Output prefix, e.g. docs/results/dflash-profile-2026-09-22')
    parser.add_argument('--markdown', type=Path)
    parser.add_argument('--figure', type=Path)
    args = parser.parse_args()
    source = args.input / 'raw.json.gz' if args.input.is_dir() else args.input
    with gzip.open(source, 'rt', encoding='utf-8') as handle:
        raw = json.load(handle)
    if 'finished' not in raw:
        parser.error('Refusing to publish an unfinished run; finished condition snapshot is missing')
    if raw.get('schema_version') != 1:
        parser.error('Unsupported raw profile schema')
    summary = build_summary(raw)
    summary['source_raw_sha256'] = sha256(source.read_bytes()).hexdigest()
    output = args.output.resolve()
    data_path = Path(str(output) + '.json')
    raw_path = Path(str(output) + '.raw.json.gz')
    markdown_path = args.markdown.resolve() if args.markdown else output.parent.parent / (output.name + '.md')
    figure_path = args.figure.resolve() if args.figure else output.parent.parent / 'assets' / (output.name + '.png')
    for path in [data_path, raw_path, markdown_path, figure_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    if source.resolve() != raw_path.resolve():
        shutil.copyfile(source, raw_path)
    plot(summary, figure_path)
    markdown_path.write_text(markdown(raw, summary, markdown_path, figure_path, data_path, raw_path), encoding='utf-8')
    print(json.dumps({'markdown': str(markdown_path), 'figure': str(figure_path), 'summary': str(data_path),
                      'raw': str(raw_path), 'all_token_ids_match': summary['all_token_ids_match']}, ensure_ascii=False))
    return 0
