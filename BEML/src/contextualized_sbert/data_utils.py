import random
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from dataclasses_json import dataclass_json
from flashtext import KeywordProcessor

NamedCluster = Tuple[str, List[str]]
EntityContext = namedtuple("EntityContext", ["left_context", "entity", "right_context"])

@dataclass_json
@dataclass
class EntityAnnotation:
  start: int
  end: int
  text: str = ""
  label: str = ""


@dataclass_json
@dataclass
class AnnotatedASIN:
  asin: str
  text: str
  entities: List[EntityAnnotation] = field(default_factory=list)


"""
输入属性为关键词的列表phrases,包含多个商品的属性集合的列表docs
返回phrase2context_idx为各个属性关键词在all_contexts中对应的上下文的开始索引和结束索引元组的列表, all_contexts为每个属性关键词在商品短语中对应的上下文，以列表形式进行存储
"""
def match_context(phrases: List[str],
                  docs: List[List[str]],
                  sampling=-1):
  raw_texts = [" ".join(doc) for doc in docs]

  # string matching
  phrase2context: Dict[str, List[EntityContext]] = defaultdict(list)
  kw_processor = KeywordProcessor()
  kw_processor.add_keywords_from_list(phrases)
  for raw_text in raw_texts:
    keywords_found = kw_processor.extract_keywords(raw_text, span_info=True)
    for kw, start, end in keywords_found:
      left_ctx = raw_text[:start].strip()
      right_ctx = raw_text[end:].strip()
      phrase2context[kw].append(EntityContext(left_ctx, kw, right_ctx))
  phrase2context = dict(phrase2context)

  phrase2context_idx = dict()
  all_contexts = []
  for phrase in phrases:
    # 从名为 phrase2context 的字典中获取与特定短语（phrase）相关联的上下文（contexts）（有多个），
    # 如果字典中没有与短语相关的上下文信息，则创建一个包含默认上下文的列表，并将其关联到该短语。
    contexts = phrase2context.get(phrase, [EntityContext("", phrase, "")])
    if sampling > 0 and len(contexts) > sampling:
      contexts = random.sample(contexts, sampling)
    start = len(all_contexts)
    end = start
    for context in contexts:
      all_contexts.append(context)
      end += 1
    phrase2context_idx[phrase] = (start, end)
  return phrase2context_idx, all_contexts


