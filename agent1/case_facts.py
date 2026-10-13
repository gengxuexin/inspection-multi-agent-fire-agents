# case_facts.py
import json
import os
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List
import logging


logger = logging.getLogger("fire_agents.agent1")

class FactLevel(str, Enum):
    KNOWN = "known"
    INFERRED = "inferred"
    ASSUMPTION = "assumption"
    MANUAL = "manual_confirm"


LEVEL_PRIORITY = {
    FactLevel.MANUAL: 100,
    FactLevel.KNOWN: 90,
    FactLevel.INFERRED: 70,
    FactLevel.ASSUMPTION: 50,
}


@dataclass
class FactItem:
    key: str
    value: Any
    level: FactLevel
    source: str = ""
    note: str = ""
    required: bool = False
    aliases: List[str] = field(default_factory=list)


class FactRegistry:
    def __init__(self):
        self.items: Dict[str, FactItem] = {}

    def add(self, key, value, level, source="", note="", required=False, aliases=None):
        self.items[key] = FactItem(
            key=key, value=value, level=level,
            source=source, note=note,
            required=required,
            aliases=aliases or [],
        )

    def add_dict(self, data: dict, level: FactLevel, source=""):
        for k, v in data.items():
            if isinstance(v, dict) and "value" in v:
                self.add(
                    k, v["value"], level,
                    v.get("source", source),
                    v.get("note", ""),
                    v.get("required", False) or level == FactLevel.MANUAL,
                    v.get("aliases", []),
                )
            else:
                self.add(
                    k, v, level, source,
                    required=(level == FactLevel.MANUAL),
                )

    def get(self, key, default=None):
        item = self.items.get(key)
        return default if item is None else item.value

    def to_mapping(self, include_assumptions=True, include_manual=False,
                   include_empty=False):
        """输出模板回填 mapping。高优先级覆盖低优先级。"""
        mapping = {}
        # 升序：先低优先级，后高优先级覆盖
        items = sorted(self.items.values(),
                       key=lambda x: LEVEL_PRIORITY.get(x.level, 0))
        for item in items:
            if item.level == FactLevel.ASSUMPTION and not include_assumptions:
                continue
            if item.level == FactLevel.MANUAL and not include_manual:
                continue
            if item.value in (None, "") and not include_empty:
                continue
            val = "" if item.value is None else str(item.value)
            mapping[item.key] = val
            for a in item.aliases:
                mapping[a] = val
        return mapping

    def missing_required(self):
        return [x for x in self.items.values()
                if x.required and x.value in (None, "")]

    def to_json(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {level.value: {} for level in FactLevel}
        for item in self.items.values():
            data[item.level.value][item.key] = {
                "value": item.value,
                "source": item.source,
                "note": item.note,
                "required": item.required,
                "aliases": item.aliases,
            }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def build_xinghua_facts() -> FactRegistry:
    reg = FactRegistry()

    BRIEF = (
        "2025年11月14日凌晨2时15分，位于某市高新区的星华工业园3号厂房发生重大火灾事故。"
        "起火建筑为单层钢结构厂房，建筑面积约4500平方米。火灾造成3人死亡、5人重伤，"
        "直接财产损失约1200万元。起火部位位于3号厂房东南角的夹层仓储区。起火原因为："
        "厂房承租方“锐创电子科技有限公司”在未办理动火审批手续、未清理周边可燃物且无专人监护的情况下，"
        "违规雇佣无特种作业操作证的工人王某进行电焊作业。电焊产生的高温焊渣掉落，"
        "引燃了夹层下方违规堆放的聚氨酯泡沫包装材料，火势迅速蔓延。"
        "出租方星华实业发展有限公司将未经消防验收合格且不符合消防安全条件的厂房出租给锐创公司，"
        "未在租赁合同中明确双方的消防安全管理职责，对共用消防设施缺乏统一管理，"
        "厂房内原有的室内消火栓系统因管网漏水长期处于瘫痪状态，火灾发生时未能正常供水，"
        "违规在厂房东侧搭建彩钢板库房，占用了防火间距，导致火势向相邻2号厂房蔓延。"
        "承租方锐创电子科技有限公司违规在厂房内设置员工宿舍（“三合一”场所），"
        "且住宿区与生产区未进行有效的防火分隔，违规使用易燃可燃材料（聚氨酯泡沫）作为包装及隔音填充物，"
        "严重违反动火作业安全管理规定，违规电焊作业，疏散通道被生产物料堵塞，安全出口上锁，"
        "导致火灾发生时多名夜班员工无法逃生。直接肇事者王某无电工资质，违规进行明火作业，操作不当引发火灾。"
    )

    NOTICE = (
        "经调查，2025年11月14日凌晨2时15分，位于某市高新区的星华工业园3号厂房发生重大火灾事故。"
        "起火建筑为单层钢结构厂房，建筑面积约4500平方米。火灾造成3人死亡、5人重伤，"
        "直接财产损失约1200万元。起火部位位于3号厂房东南角的夹层仓储区。起火原因为："
        "厂房承租方“锐创电子科技有限公司”在未办理动火审批手续、未清理周边可燃物且无专人监护的情况下，"
        "违规雇佣无特种作业操作证的工人王某进行电焊作业。电焊产生的高温焊渣掉落，"
        "引燃了夹层下方违规堆放的聚氨酯泡沫包装材料，火势迅速蔓延。"
        "出租方星华实业发展有限公司将未经消防验收合格且不符合消防安全条件的厂房出租给锐创公司，"
        "未在租赁合同中明确双方的消防安全管理职责，对共用消防设施缺乏统一管理，"
        "厂房内原有的室内消火栓系统因管网漏水长期处于瘫痪状态，火灾发生时未能正常供水，"
        "违规在厂房东侧搭建彩钢板库房，占用了防火间距，导致火势向相邻2号厂房蔓延。"
        "承租方锐创电子科技有限公司违规在厂房内设置员工宿舍（“三合一”场所），"
        "且住宿区与生产区未进行有效的防火分隔，违规使用易燃可燃材料（聚氨酯泡沫）作为包装及隔音填充物，"
        "严重违反动火作业安全管理规定，违规电焊作业，疏散通道被生产物料堵塞，安全出口上锁，"
        "导致火灾发生时多名夜班员工无法逃生。直接肇事者王某无电工资质，违规进行明火作业，操作不当引发火灾。"
        "以上事实有现场检查记录、询问笔录、现场照片等证据证实。"
        "根据《中华人民共和国消防法》等有关规定，拟对你（单位）作出行政处罚。你（单位）有权进行陈述和申辩。"
    )

    # ================= known =================
    known = {
        "案件名称": "星华工业园“11·14”重大火灾事故案",
        "火灾时间": "2025年11月14日凌晨2时15分",
        "违法时间": "2025年11月14日",
        "案发地点": "某市高新区星华工业园3号厂房",
        "违法地点": "某市高新区星华工业园3号厂房",
        "建筑情况": "单层钢结构厂房，建筑面积约4500平方米",
        "伤亡后果": "3人死亡、5人重伤",
        "财产损失": "直接财产损失约1200万元",
        "起火部位": "3号厂房东南角夹层仓储区",
        "起火原因": "锐创电子科技有限公司未办理动火审批手续、未清理周边可燃物且无专人监护，违规雇佣无特种作业操作证的工人王某进行电焊作业，高温焊渣掉落引燃夹层下方违规堆放的聚氨酯泡沫包装材料。",
        "出租方": "星华实业发展有限公司",
        "承租方": "锐创电子科技有限公司",
        "直接责任人": "王某",
        "出租方违法事实": "将未经消防验收合格且不符合消防安全条件的厂房出租；未在租赁合同中明确双方消防安全管理职责；对共用消防设施缺乏统一管理；室内消火栓系统因管网漏水长期瘫痪，火灾时未能正常供水；违规在厂房东侧搭建彩钢板库房，占用防火间距，导致火势向相邻2号厂房蔓延。",
        "承租方违法事实": "违规在厂房内设置员工宿舍（“三合一”场所），住宿区与生产区未有效防火分隔；违规使用聚氨酯泡沫作为包装及隔音填充物；违反动火作业安全管理规定，违规电焊作业；疏散通道被生产物料堵塞，安全出口上锁，导致夜班员工无法逃生。",
        "王某违法事实": "无电工资质，违规进行明火作业，操作不当引发火灾。",
        "证据": "现场检查记录、询问笔录、现场照片、火灾事故认定书、证人证言、物证、鉴定意见、视频资料等",
        "执行方式和期限": "自收到本决定书之日起十五日内到指定银行缴纳罚款；逾期不缴纳的，每日按罚款数额的百分之三加处罚款，加处罚款的数额不超过罚款的数额。",
        "复议机关": "某市人民政府",
        "管辖法院": "某市人民法院",
        "附件清单": "1.行政处罚决定书；2.现场检查记录；3.询问笔录；4.现场照片；5.火灾事故认定书；6.证据清单；7.法律文书送达回证。",
        "询问时间": "2025年11月14日",
        "询问地点": "某市高新区星华工业园3号厂房",
        "询问人": "小行星",
        "被询问人": "王某",
        "检查日期": "2025年11月14日",
        "检查时间": "2025年11月14日9时00分至11时30分",
        "消防救援机构名称": "某市高新区消防救援大队",
        "执行告知单位": "某市高新区消防救援大队",
        "告知人": "小行星",
        "被告知人": "王某",
        "被处罚人": "星华实业发展有限公司、锐创电子科技有限公司、王某",
        "具体问题": "厂房未经消防验收合格且不符合消防安全条件出租；未在租赁合同中明确双方消防安全管理职责；对共用消防设施缺乏统一管理，室内消火栓系统因管网漏水长期瘫痪，火灾时未能正常供水；违规在厂房东侧搭建彩钢板库房，占用防火间距；违规在厂房内设置员工宿舍（“三合一”场所），住宿区与生产区未进行有效防火分隔；违规使用易燃可燃材料（聚氨酯泡沫）作为包装及隔音填充物；违反动火作业安全管理规定，违规电焊作业；疏散通道被生产物料堵塞，安全出口上锁。",
        "简要案情": BRIEF,
        "告知内容": NOTICE,
    }
    reg.add_dict(known, FactLevel.KNOWN, source="案情.txt/询问笔录/处罚决定书")

    # ================= inferred =================
    inferred = {
        "年份": "2025",
        "文号": "000111",
        "编号续": "000111",
        "编号": "X消监字",
        "日期": "2025年11月14日",
        "决定日期": "2025年11月14日",
        "改正期限": "2025年11月21日",
        "案件来源": "4",
        "案件来源其他": "消防救援机构在火灾事故调查中发现",
        "案由": "违规电焊作业引发火灾事故",
        "违法行为项数": "11",
        "立案意见": "经初步调查，星华实业发展有限公司、锐创电子科技有限公司及王某涉嫌违反消防法律法规，建议立案调查。",
        "立案审批": "同意立案。",
        "负责人": "王大队长",
        "执法人员": "小行星",
        "消防监督检查员签名": "小行星",
        "检查类型": "其他检查",
        "备注": "已制发责令限期改正通知书，拟立案处罚。",
        "复核结论": "被告知人未提出陈述和申辩，依法作出行政处罚决定。",
        "听证处理结果": "被告知人放弃听证，依法作出行政处罚决定。",
    }
    reg.add_dict(inferred, FactLevel.INFERRED, source="由案件编号/时间/材料推导")

    # ================= assumption =================
    assumption = {
        "记录人": "小月亮",
        "单位法定代表人": "李某某、张某某",
        "法定代表人": "李某某、张某某",
        "被告知单位法定代表人": "张某某",
        "统一社会信用代码": "91370100MA3TEST0X、91370100MA4TEST0X",
        "个人姓名": "王某",
        "个人性别": "男",
        "个人出生日期": "1985年6月12日",
        "身份证件种类": "居民身份证",
        "证件号码": "370100198506120000",
        "个人住址": "某市高新区星华工业园3号厂房员工宿舍",
        "个人联系方式": "13900000000",
        "单位名称": "星华实业发展有限公司、锐创电子科技有限公司",
        "被检查单位": "星华实业发展有限公司、锐创电子科技有限公司",
        "被检查单位场所名称": "星华实业发展有限公司、锐创电子科技有限公司",
        "单位地址": "某市高新区星华工业园3号厂房",
        "地址": "某市高新区星华工业园3号厂房",
        "单位联系方式": "0573-88888888、13800000000",
        "联系方式": "0573-88888888、13800000000",
        "联系电话": "0573-88888888、13800000000",
        "联系人电话": "0573-88888888、13800000000",
        "被检查单位随同检查人员签名": "李某某、张某某",
        "被检查单位签收人": "李某某、张某某",
        "被告知人签名": "王某",
        "被处罚人签名": "星华实业发展有限公司、锐创电子科技有限公司、王某",
        "送达日期": "2025年11月14日",
        "陈述申辩内容": "对上述告知事项，你（单位）是否提出陈述和申辩？",
        "陈述申辩回答": "不提出陈述和申辩。",
        "听证询问": "对上述告知事项，你是否要求听证？",
        "听证回答": "不要求听证。",
        "处罚决定": "对星华实业发展有限公司罚款捌万元并责令停产停业整顿；对锐创电子科技有限公司罚款拾万元并责令停产停业整顿；对王某处行政拘留五日，因涉嫌犯罪移送司法机关。",
        "法律依据": "《中华人民共和国消防法》第六十条、第六十一条、第六十二条、第六十三条、第六十四条、第六十八条等",
        "违法行为人信息": "星华实业发展有限公司，地址：某市高新区星华工业园3号厂房，法定代表人：李某某，统一社会信用代码：91370100MA3TEST0X，联系电话：0573-88888888；锐创电子科技有限公司，地址：某市高新区星华工业园3号厂房，法定代表人：张某某，统一社会信用代码：91370100MA4TEST0X，联系电话：13800000000；王某，男，1985年6月12日出生，居民身份证，号码370100198506120000，户籍所在地某省某市某区某街道，现住某市高新区星华工业园3号厂房员工宿舍，联系方式13900000000。",
    }
    reg.add_dict(assumption, FactLevel.ASSUMPTION, source="测试假设，可替换")

    # ================= manual_confirm =================
    manual = {
        "真实统一社会信用代码": "",
        "真实法定代表人": "",
        "真实身份证号": "",
        "罚款金额": "",
        "是否要求听证": "",
        "是否提出陈述申辩": "",
        "是否移送司法机关": "",
        "伤亡人员名单": "",
        "鉴定意见": "",
        "送达回证": "",
    }
    reg.add_dict(manual, FactLevel.MANUAL, source="必须人工确认")

    return reg


if __name__ == "__main__":
    reg = build_xinghua_facts()
    out = os.path.join(os.path.dirname(__file__), "inputs",
                       "xinghua_11_14_case_facts.json")
    reg.to_json(out)
    logger.info(f"已导出：{out}")
    logger.info("缺失必填：", [x.key for x in reg.missing_required()])