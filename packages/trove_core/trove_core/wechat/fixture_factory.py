from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import random

from .models import Account, Conversation, Message

@dataclass(frozen=True)
class FixtureData:
    accounts: list[Account]
    conversations: list[Conversation]
    messages: list[Message]

    def to_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
            for msg in self.messages:
                f.write(json.dumps(msg.safe_dict(), ensure_ascii=False) + '\n')


def generate_fixture(seed: int = 20260621) -> FixtureData:
    random.seed(seed)
    base = datetime(2026, 6, 20, 9, 0, tzinfo=timezone.utc)
    accounts = [
        Account('acct-personal', 'Personal-WeChat', '个人微信'),
        Account('acct-work', 'Work-WeChat', '工作微信'),
    ]
    conversations = [
        Conversation('conv-example_edu-private', 'acct-work', '客户-示例教育-私聊', 'private', 2),
        Conversation('conv-sales-review', 'acct-work', '私域成交复盘群', 'group', 6),
        Conversation('conv-trove-team', 'acct-work', 'TROVE 产品小组', 'group', 5),
        Conversation('conv-family', 'acct-personal', '家庭事务', 'group', 4),
    ]
    def m(account_id: str, label: str, conv_id: str, title: str, ctype: str, sender: str, sender_id: str, minutes: int, content: str, shard: str, local: int, me: bool=False) -> Message:
        return Message(account_id=account_id, account_label=label, conversation_id=conv_id, conversation_title=title, conversation_type=ctype, sender_id=sender_id, sender_name=sender, timestamp=base + timedelta(minutes=minutes), content=content, shard_id=shard, local_id=local, sent_by_me=me)
    messages = [
        # Duplicate local_id across shards is intentional.
        m('acct-work','Work-WeChat','conv-example_edu-private','客户-示例教育-私聊','private','林老师','cust-lin',1,'我们看了方案，功能认可，但是价格太高，预算审批也没过。','message_0',1),
        m('acct-work','Work-WeChat','conv-example_edu-private','客户-示例教育-私聊','private','我','me-work',3,'收到，我先整理一个基础版报价，并把客户成交卡点写清楚。','message_0',2, True),
        m('acct-work','Work-WeChat','conv-example_edu-private','客户-示例教育-私聊','private','林老师','cust-lin',8,'如果能先做三个月试点，下周三我可以推动校长确认。','message_1',1),
        m('acct-work','Work-WeChat','conv-sales-review','私域成交复盘群','group','销售-成员甲','sales-aj',12,'示例教育这个客户之前卡在价格太高和预算审批，下一步要给基础版试点。','message_0',10),
        m('acct-work','Work-WeChat','conv-sales-review','私域成交复盘群','group','运营-小周','ops-zhou',14,'复盘结论：先降低首单风险，再用三个月试点推动成交。','message_0',11),
        m('acct-work','Work-WeChat','conv-sales-review','私域成交复盘群','group','我','me-work',15,'我负责明天发新版报价，成员甲负责约下周三复盘。','message_0',12, True),
        m('acct-work','Work-WeChat','conv-trove-team','TROVE 产品小组','group','产品-成员乙','pm-bai',25,'今天决定把 Hyper Search 做成 evidence-first，所有答案必须带 citation。','message_0',20),
        m('acct-work','Work-WeChat','conv-trove-team','TROVE 产品小组','group','工程-小陈','eng-chen',28,'API 只绑定 127.0.0.1，search evidence context 都要 local token。','message_0',21),
        m('acct-work','Work-WeChat','conv-trove-team','TROVE 产品小组','group','设计-小夏','design-xia',30,'Web Console UI 保持极简：搜索框、Vault status、Evidence results、Context panel、Settings。','message_0',22),
        m('acct-work','Work-WeChat','conv-trove-team','TROVE 产品小组','group','我','me-work',33,'团队决定 7月15日上线 fixture search smoke，先不依赖 ZVEC。','message_1',20, True),
        m('acct-personal','Personal-WeChat','conv-family','家庭事务','group','妈妈','mom',40,'周末记得买牛奶，这条是个人生活消息，不应出现在客户筛选结果里。','message_0',1),
        m('acct-personal','Personal-WeChat','conv-family','家庭事务','group','我','me-personal',43,'好的，我会顺路买。','message_0',2, True),
    ]
    return FixtureData(accounts, conversations, messages)
