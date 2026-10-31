"""Read-only, evidence-backed movie tools; a deterministic V2 baseline."""
from __future__ import annotations

import argparse
from collections import defaultdict
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re
import time

from .candidates import CandidateIndex, normalized
from .common import read_json, read_jsonl, sha256
from .v2_data import DEST, PROPERTIES

INTENTS = {
    'P57': r'导演|導演|执导|執導|\b(?:directors?|directed|helmed)\b',
    'P136': r'类型|類型|\bgenres?\b',
    'P495': r'制作国|製作國|制片国|製片國|哪国|哪个国家|\b(?:country|countries)\b',
    'P364': r'原始语言|原始語言|原声|什么语|哪种语言|\blanguages?\b',
    'P161': r'演员|演員|谁演|誰演|\b(?:cast|actors?|actresses|starring)\b',
    'P58': r'编剧|編劇|\b(?:screenwriters?|writers?|screenplay|wrote|written)\b',
    'P577': r'上映|发行日期|發行日期|哪年|什么时候|\b(?:release|released|when|publication)\b',
    'P915': r'拍摄|拍攝|取景|\b(?:filming|filmed|shot)\b',
    'P840': r'故事(?:发生|發生|地点|地點)|剧情(?:地点|地點)|\b(?:set in|narrative location)\b',
    'P2130': r'成本|预算|預算|\b(?:budget|cost)\b',
    'P2142': r'票房|\bbox office\b',
    'P725': r'配音演员|配音演員|\bvoice actors?\b',
}


def unconsumed(text):
    """Conservative residue check for constraints the finite grammar cannot execute."""
    for pattern in INTENTS.values():
        text = re.sub(pattern, ' ', text)
    text = re.sub(r'\b(?:tell|me|please|can|could|you|who|what|which|when|where|how|much|is|are|was|were|the|a|an|of|in|by|for|to|do|does|did|give|list|all|their|and|or|with|has|have|show|information|about|movies|films|movie|film|other|compare|common|shared|both|origin|originate|date|dates|on|at|its|it|gross|earn|earnings|production|total)\b', ' ', text)
    text = re.sub(r'请问|請問|告诉我|告訴我|分别|分別|有哪些|哪些|什么|什麼|是谁|是誰|共同|共有|以及|多少|日期|制作|製作|电影|電影|和|与|與|及|的|有|是|谁|誰|、', ' ', text)
    return bool(re.search(r'[a-z\u3400-\u9fff]', text))


class Explorer:
    def __init__(self, directory=DEST):
        self.directory = Path(directory)
        self.manifest = read_json(self.directory / 'manifest.json')
        if self.manifest.get('schema') != 'movie-catalog-v2/1':
            raise ValueError('Unsupported catalogue schema')
        for name in ('movies.json', 'entities.json', 'statements.jsonl', 'coverage.json', 'provenance.json'):
            if sha256(self.directory / name) != self.manifest['files'].get(name):
                raise ValueError(f'Catalogue file differs from frozen manifest: {name}')
        movies = read_json(self.directory / 'movies.json')
        self.movies = {m['id']: m for m in movies}
        self.entities = read_json(self.directory / 'entities.json')
        self.rows = read_jsonl(self.directory / 'statements.jsonl')
        self.index = CandidateIndex(movies)
        self.by_pair = defaultdict(list)
        self.reverse = defaultdict(set)
        for row in self.rows:
            self.by_pair[row['subject'], row['property']].append(row)
            if row['type'] == 'entity' and row['snaktype'] == 'value':
                self.reverse[row['property'], row['value']].add(row['subject'])

    def name(self, qid, chinese=False):
        entity = self.entities.get(qid, {})
        value = (entity.get('label_zh') if chinese else None) or entity.get('label', qid)
        return value if value != qid else qid + ('（名称待补齐）' if chinese else ' (name pending)')

    def snak_text(self, snak, chinese=False):
        if snak.get('snaktype') != 'value':
            return snak.get('snaktype', 'unknown')
        value = snak.get('datavalue', {}).get('value')
        if isinstance(value, dict):
            if 'id' in value:
                return f'{self.name(value["id"], chinese)} ({value["id"]})'
            if 'time' in value:
                return f'{value["time"]} [precision={value.get("precision")}]'
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    def describe(self, row, chinese=False):
        value = row['value']
        if row['snaktype'] != 'value':
            return ('未知值' if chinese else 'Unknown value') + f' ({row["snaktype"]})'
        if row['type'] == 'entity':
            return f'{self.name(value, chinese)} ({value})'
        if row['type'] == 'time':
            raw = value['time']
            precision = value.get('precision', 0)
            date = raw.lstrip('+').split('T')[0]
            if precision == 9:
                date = date[:4]
            elif precision == 10:
                date = date[:7]
            elif precision < 9:
                date = f'{raw} [precision={precision}]'
            calendar = value.get('calendarmodel', '').rsplit('/', 1)[-1]
            return f'{date} [precision={precision}, calendar={calendar}]'
        unit = value.get('unit', '1')
        unit_id = unit.rsplit('/', 1)[-1]
        label = self.name(unit_id, chinese) if unit != '1' else ('单位未指定' if chinese else 'unit unspecified')
        bounds = ''
        if value.get('lowerBound') is not None or value.get('upperBound') is not None:
            bounds = f' [{value.get("lowerBound")}, {value.get("upperBound")}]'
        return f'{value["amount"].lstrip("+")} {label}{bounds}'

    def evidence(self, row, chinese=False):
        qualifiers = '; '.join(f'{p}: ' + ', '.join(self.snak_text(s, chinese) for s in snaks)
                               for p, snaks in row['qualifiers'].items())
        return {'statement_id': row['id'], 'movie_id': row['subject'], 'property': row['property'],
                'value': row['value'], 'type': row['type'], 'snaktype': row['snaktype'],
                'qualifiers': row['qualifiers'], 'qualifier_text': qualifiers,
                'references': row['references'], 'source_url': row['source_url'],
                'revision': row['revision'], 'retrieved_at': row.get('retrieved_at'), 'rank': row['rank'],
                'text': f'{self.name(row["subject"], chinese)} | {PROPERTIES[row["property"]][0 if chinese else 1]} | {self.describe(row, chinese)}' +
                        (f' | {qualifiers}' if qualifiers else '')}

    def plan(self, question, confirmed=None):
        if not isinstance(question, str) or not 1 <= len(question.strip()) <= 1000:
            raise ValueError('请输入1—1000字符的问题。')
        if confirmed is not None and (not isinstance(confirmed, list) or len(confirmed) > 5 or
                                      any(not isinstance(q, str) or q not in self.movies for q in confirmed)):
            raise ValueError('确认电影必须是当前目录中最多5个有效QID。')
        chinese = bool(re.search(r'[\u3400-\u9fff]', question))
        ids, candidates, status = self.index.mentions(question)
        if confirmed:
            ids, candidates, status = list(dict.fromkeys([*ids, *confirmed])), [], 'confirmed'
        # Mask title words before interpreting relation and qualifier cues.
        text = normalized(question)
        for qid in dict.fromkeys([*ids, *(c['id'] for c in candidates)]):
            movie = self.movies[qid]
            for name in sorted([movie['label'], movie.get('label_zh') or '', *movie.get('aliases', [])], key=len, reverse=True):
                name = normalized(name)
                if name:
                    text = re.sub(r'(?<!\w)' + re.escape(name) + r'(?!\w)' if re.fullmatch(r'[a-z0-9 ]+', name) else re.escape(name), ' ', text)
        text = re.sub(r'(?<![a-z0-9])q\d+(?![a-z0-9])', ' ', text)
        props = [p for p, pattern in INTENTS.items() if re.search(pattern, text)]
        if 'P725' in props and 'P161' in props:
            props.remove('P161')
        location_country = bool({'P915', 'P495'} <= set(props) or {'P840', 'P495'} <= set(props))
        if 'P915' in props or 'P840' in props:
            if 'P495' in props:
                props.remove('P495')
        operation = 'facts'
        if re.search(r'共同|共有|\b(?:common|shared|both)\b', text):
            operation = 'intersection'
        elif re.search(r'还(?:执导|拍|导)|还(?:有|拍过)|其他.*(?:电影|作品)|\b(?:other films|other movies)\b', text):
            operation = 'reverse'
        elif re.search(r'推荐|推薦|\brecommend', text):
            operation = 'recommend'
        elif re.search(r'哪个.*(?:高|多)|哪部.*(?:高|多)|比较|對比|对比|\b(?:higher|compare|larger|more)\b', text):
            operation = 'compare'
        block = None
        if location_country:
            block = '当前记录拍摄/故事地点；地点所属国家的多跳过滤尚未开放。' if chinese else 'Recorded locations are available, but country-of-location traversal is not implemented.'
        elif re.search(r'角色|饰演|飾演|扮演|\b(?:played|portrayed|role|character)\b', text):
            block = '特定角色需要角色匹配查询，当前仅能查询出演名单，不能用全体演员代替。' if chinese else 'Character-role filtering is not implemented; a full cast list cannot answer this role question.'
        elif re.search(r'封禁|禁映|评分|評分|奖|獎|制片人|製片人|时长|時長|妻子|丈夫|利润|利潤|续集|續集|\b(?:banned|ban|ratings?|awards?|oscar|producers?|runtime|wife|husband|profit|best|worst|sequel|inflation|adjusted|net)\b', text):
            block = '问题包含当前目录/工具不支持的属性或限定条件。' if chinese else 'The question includes an unsupported property or constraint.'
        elif re.search(r'剧情|劇情|简介|簡介|\b(?:plot|synopsis|summari[sz]e)\b', text):
            block = '当前尚未导入可引用的剧情文本。' if chinese else 'Citable plot text has not been imported yet.'
        elif re.search(r'配音(?!演员|演員)|译制|字幕|\b(?:dubbed|dubbing|subtitles?)\b', text):
            block = '配音/字幕需要明确发行版本及其来源，当前目录未建立版本事实。' if chinese else 'Dub/subtitle queries require a sourced release version, which is not available.'
        elif re.search(r'预算|預算|\bbudget\b', text):
            block = '目前只有来源标记为制作成本的记录，不能直接当作公布预算。请改问制作成本。' if chinese else 'Production cost is not automatically an announced budget. Please ask for production cost.'
        elif re.search(r'主演|领衔|領銜|\b(?:lead actors?|stars?)\b', text):
            block = '当前保存的是出演名单，未核实主演身份；请改问演员。' if chinese else 'Cast membership does not establish lead roles. Please ask for the cast.'
        elif re.search(r'全球|中国|中國|美国|美國|香港|内地|\b(?:worldwide|global|united states|china|hong kong|in the uk|in japan)\b', text) or (
                any(p in props for p in ('P577', 'P2142')) and re.search(r'\b(?:in|for)\s+(?:the\s+)?[a-z]|在.+(?:上映|发行|發行|票房)', text)):
            # Do not silently answer a regional constraint with an unfiltered value.
            block = '当前可列出各记录的地区限定；指定地区/全球的自动过滤尚未开放，请去掉地区后核对证据。' if chinese else 'Regional constraints are displayed in evidence but automatic region filtering is not implemented; ask without the region and inspect the qualifiers.'
        elif re.match(r'\s*(?:is|are|was|were|did|does|do|has|have)\b', normalized(question)) or re.search(r'是否|是不是|有没有|有沒有', text):
            block = '目前返回已记录事实列表，不根据缺失边作肯定/否定判断。' if chinese else 'This tool lists recorded facts; missing edges cannot establish yes/no claims.'
        elif re.search(r'\b(?:18|19|20)\d{2}\b|之前|之后|之後|早于|晚于|高于|超过|超過|最早|首映|更早|\b(?:before|after|since|over|under|not|excluding|except|earlier|earliest|premiere|first|latest)\b', text):
            block = '日期、阈值或否定过滤尚未开放；请先查询事实并核对限定条件。' if chinese else 'Date, threshold or negative filtering is not implemented; first inspect the recorded facts and qualifiers.'
        if not block and ids and any(p in props for p in ('P577', 'P2130', 'P2142')) and unconsumed(text):
            block = '日期/金额问题含有尚未支持的筛选条件，请去掉条件后核对原始记录及限定。' if chinese else 'The date/amount question contains an unsupported constraint. Remove it and inspect the recorded values and qualifiers.'
        if not block and re.search(r'\band\b|和|与|與|以及', text):
            if unconsumed(text):
                block = '还有未识别的电影或条件，请补全片名/使用明确QID，或拆开提问。' if chinese else 'Another movie or constraint is unresolved. Supply complete titles/QIDs or separate the questions.'
        return {'method': 'deterministic_tools_v2', 'question': question, 'language': 'zh' if chinese else 'en',
                'movie_ids': ids, 'properties': props, 'operation': operation, 'candidates': candidates,
                'link_status': status, 'blocked': block}

    def answer(self, question, confirmed=None, offset=0, limit=30):
        self._validate_page(offset, limit)
        started = time.perf_counter()
        plan = self.plan(question, confirmed)
        return self._execute(plan, offset, limit, started)

    @staticmethod
    def _validate_page(offset, limit):
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0 or not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
            raise ValueError('分页要求offset≥0，limit为1—100。')

    def execute(self, plan, offset=0, limit=None, *, allowed_movie_ids, question=''):
        """Execute a candidate-bound typed plan without rule-based reparsing.

        The semantic caller must resolve candidate ambiguity before passing a
        ready plan. Non-ready plans produce no evidence or graph operations.
        Pagination is applied after complete set/path computation.
        """
        from .planning import validate_plan

        started = time.perf_counter()
        validated = validate_plan(plan, allowed_movie_ids, catalogue_movie_ids=self.movies)
        limit = validated['limit'] if limit is None else limit
        self._validate_page(offset, limit)
        if not isinstance(question, str) or len(question) > 1000:
            raise ValueError('question must be text of at most 1000 characters')
        legacy = {'method': 'validated_plan_tools_v1', 'question': question,
                  'language': validated['language'], 'movie_ids': validated['movie_ids'],
                  'properties': validated['properties'], 'operation': validated['operation'],
                  'candidates': [], 'link_status': 'confirmed', 'blocked': None}
        if validated['status'] == 'needs_clarification':
            legacy['link_status'] = 'needs_clarification'
        elif validated['status'] == 'unsupported':
            legacy['blocked'] = validated['clarification'] or '; '.join(validated['unhandled_constraints'])
        result = self._execute(legacy, offset, limit, started)
        result['validated_plan'] = validated
        if validated['status'] == 'needs_clarification':
            result['answer'] = validated['clarification'] or '; '.join(validated['unhandled_constraints'])
        return result

    def _execute(self, plan, offset, limit, started):
        question = plan['question']
        zh = plan['language'] == 'zh'
        result = {'question': question, 'plan': plan, 'status': 'answered', 'answer': '', 'evidence': [],
                  'items': [], 'missing': [], 'scope': 'current frozen catalogue', 'total': 0, 'offset': offset,
                  'limit': limit, 'next_offset': None}
        def finish(status, text):
            result.update(status=status, answer=text, elapsed_ms=round((time.perf_counter()-started)*1000, 2))
            return result
        if plan['link_status'] == 'needs_clarification':
            return finish('needs_clarification', '多个电影名称仍有歧义，请用明确QID分别指定。' if zh else 'Multiple title slots are ambiguous. Specify each movie with an explicit QID.')
        if plan['link_status'] == 'needs_confirmation':
            return finish('needs_confirmation', '请确认要查询的电影；已识别的电影会保留。' if zh else 'Please confirm the movie; already resolved movies will be retained.')
        if plan['blocked']:
            return finish('unsupported', plan['blocked'])
        ids, props = plan['movie_ids'], plan['properties']
        if not ids:
            return finish('needs_confirmation' if plan['candidates'] else 'not_found',
                          ('请确认要查询的电影；相似检索不会直接替你选片。' if zh else 'Please confirm the movie. Approximate matches are candidates only.'))
        op = plan['operation']
        if not props and op != 'recommend':
            return finish('needs_clarification', '请说明要查询的属性。' if zh else 'Please specify a supported property.')
        if op in {'intersection', 'compare'} and len(ids) != 2:
            return finish('needs_clarification', '请明确两部电影。' if zh else 'Please specify exactly two movies.')
        if op == 'reverse' and (len(ids) != 1 or props != ['P57']):
            return finish('needs_clarification', '当前支持一部电影→导演→片库其他作品。' if zh else 'Supported path: one movie → director → other catalogue movies.')
        if op == 'compare':
            if any(PROPERTIES[p][2] == 'quantity' for p in props):
                return self.compare(result, ids, props, zh, finish)
            # Attribute comparison is a sourced listing, without inventing a ranking.
            result['comparison'] = {'mode': 'attribute_listing', 'movie_ids': ids}
            op = 'facts'
        if op == 'recommend':
            return finish('unsupported', '内容推荐需要明确筛选条件与相关性验证，当前暂未开放。' if zh else 'Content recommendation awaits explicit filters and relevance validation.')
        records = []
        if op == 'facts':
            for qid in ids:
                for prop in props:
                    rows = self.by_pair[qid, prop]
                    if not any(r['snaktype'] == 'value' for r in rows):
                        result['missing'].append({'movie_id': qid, 'property': prop})
                    records.extend(rows)
        elif op == 'intersection':
            if any(PROPERTIES[p][2] != 'entity' for p in props):
                return finish('unsupported', '共同属性暂限实体集合。' if zh else 'Intersection supports entity-valued properties only.')
            for prop in props:
                a, b = (self.by_pair[qid, prop] for qid in ids)
                shared = {r['value'] for r in a if r['snaktype'] == 'value'} & {r['value'] for r in b if r['snaktype'] == 'value'}
                result['items'].extend({'property': prop, 'entity_id': q, 'label': self.name(q, zh)} for q in sorted(shared))
                records.extend(r for r in [*a, *b] if r['value'] in shared and r['snaktype'] == 'value')
            if not records:
                return finish('no_recorded_overlap', '当前资料未发现共同项；不证明现实中不存在。' if zh else 'No overlap was found in the recorded data; this does not establish real-world absence.')
        elif op == 'reverse':
            first = [r for r in self.by_pair[ids[0], 'P57'] if r['snaktype'] == 'value']
            for edge in first:
                for other in sorted(self.reverse['P57', edge['value']] - {ids[0]}):
                    result['items'].append({'movie_id': other, 'label': self.name(other, zh), 'via': edge['value']})
                    records.append(edge)
                    records.extend(r for r in self.by_pair[other, 'P57'] if r['value'] == edge['value'])
            if not first:
                result['missing'].append({'movie_id': ids[0], 'property': 'P57'})
            elif not records:
                return finish('no_recorded_other_works', '当前片库没有记录该导演的其他作品，不代表完整作品表。' if zh else 'No other works by the recorded director were found in this catalogue; this is not a complete filmography.')
        # Stable de-duplication preserves statement scope, not just tail ID.
        unique = {r['id']: r for r in records}
        records = list(unique.values())
        result['total'] = len(result['items']) if op != 'facts' else len(records)
        if op == 'facts':
            visible = records[offset:offset+limit]
            result['evidence'] = [self.evidence(r, zh) for r in visible]
            lines = [e['text'] for e in result['evidence']]
        else:
            result['items'] = result['items'][offset:offset+limit]
            visible_ids = {i.get('movie_id') or i.get('entity_id') for i in result['items']}
            visible = [r for r in records if r['subject'] in visible_ids or r['value'] in visible_ids]
            if op == 'reverse' and visible:
                visible = first + visible
            result['evidence'] = [self.evidence(r, zh) for r in {r['id']: r for r in visible}.values()]
            lines = [f'{i["label"]} ({i.get("movie_id") or i.get("entity_id")})' +
                     (f' ← {self.name(i["via"], zh)} ({i["via"]})' if 'via' in i else '') for i in result['items']]
        if result['total'] > offset + limit:
            result['next_offset'] = offset + limit
        prefix = ('当前片库记录：' if zh else 'Recorded in the current catalogue:')
        lines.insert(0, prefix)
        if result['total']:
            lines.append((f'共{result["total"]}项，当前显示{min(offset+1,result["total"])}—{min(offset+limit,result["total"])}项。' if zh else
                          f'{result["total"]} items; showing {min(offset+1,result["total"])}–{min(offset+limit,result["total"])}.'))
        if result['missing']:
            missing = ', '.join(f'{self.name(m["movie_id"], zh)} / {PROPERTIES[m["property"]][0 if zh else 1]}' for m in result['missing'])
            lines.append(('缺少事实：' if zh else 'Missing facts: ') + missing)
        return finish('partial' if result['missing'] and records else 'unanswerable' if result['missing'] else 'answered', '\n'.join(lines))

    def compare(self, result, ids, props, zh, finish):
        if len(props) != 1 or props[0] not in {'P2142', 'P2130'}:
            return finish('needs_clarification', '目前数值比较支持同口径票房或制作成本；其他属性可分别查询。' if zh else 'Numeric comparison supports comparable box-office or production-cost amounts; other attributes can be listed separately.')
        pairs = [[r for r in self.by_pair[q, props[0]] if r['snaktype'] == 'value'] for q in ids]
        result['evidence'] = [self.evidence(r, zh) for group in pairs for r in group]
        if any(len(group) != 1 for group in pairs):
            return finish('needs_clarification', '缺少唯一可比较的金额记录；请核对多值/缺失和口径。' if zh else 'No unique comparable amount per movie. Inspect missing or multiple records and scope.')
        a, b = pairs[0][0], pairs[1][0]
        # P3005/P518 are explicit territorial/part scopes, retained verbatim.
        def scope(row):
            return {p: row['qualifiers'][p] for p in ('P3005', 'P518') if p in row['qualifiers']}
        if not scope(a) or scope(a) != scope(b) or a['value'].get('unit') == '1' or a['value'].get('unit') != b['value'].get('unit'):
            return finish('not_comparable', '币种或明确统计范围不一致/缺失，不能可靠比较。' if zh else 'Currency or explicit scope is missing/different; these records are not reliably comparable.')
        # Other qualifiers (dates, methodologies, etc.) must also agree.
        if a['qualifiers'] != b['qualifiers']:
            return finish('not_comparable', '限定条件不同，当前不自动换算或比较。' if zh else 'Qualifiers differ; automatic conversion/comparison is not available.')
        try:
            amounts = [Decimal(r['value']['amount']) for r in (a, b)]
        except (InvalidOperation, KeyError, TypeError):
            return finish('not_comparable', '金额格式无法比较。' if zh else 'Amounts cannot be parsed.')
        if not all(n.is_finite() for n in amounts) or any(r['value'].get(k) is not None for r in (a, b) for k in ('lowerBound', 'upperBound')):
            return finish('not_comparable', '非有限金额或金额区间需要单独核查，当前不作确定排序。' if zh else 'Non-finite amounts or bounded intervals require separate review; no definite ranking is made.')
        winner = None if amounts[0] == amounts[1] else ids[0 if amounts[0] > amounts[1] else 1]
        result['comparison'] = {'winner': winner, 'difference': str(abs(amounts[0]-amounts[1])), 'unit': a['value']['unit']}
        result['total'] = 2
        return finish('answered', ('记录金额相同。' if zh else 'Recorded amounts are equal.') if not winner else
                      (f'{self.name(winner, zh)}的记录金额更高，差额{result["comparison"]["difference"]}（同币种、同限定条件）。' if zh else
                       f'{self.name(winner)} has the higher recorded amount, by {result["comparison"]["difference"]} in the same currency and scope.'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog-dir', type=Path, default=DEST)
    parser.add_argument('--question', required=True)
    parser.add_argument('--movie-id', action='append', dest='confirmed')
    parser.add_argument('--offset', type=int, default=0)
    args = parser.parse_args(argv)
    print(json.dumps(Explorer(args.catalog_dir).answer(args.question, args.confirmed, args.offset), ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
