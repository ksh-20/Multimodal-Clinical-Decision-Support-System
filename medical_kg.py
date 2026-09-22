"""
medical_kg.py — Knowledge graph loader and query interface.
Backed by kg_relationships.json; uses NetworkX for graph traversal.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx

from config import KG_JSON_PATH

log = logging.getLogger(__name__)


class MedicalKG:
    def __init__(self, json_path: Path = KG_JSON_PATH) -> None:
        with open(json_path, "r", encoding="utf-8") as fh:
            self._raw = json.load(fh)

        self.G: nx.DiGraph = nx.DiGraph()
        for node in self._raw["nodes"]:
            self.G.add_node(node["id"], **{k: v for k, v in node.items() if k != "id"})

        for edge in self._raw["edges"]:
            self.G.add_edge(
                edge["source"],
                edge["target"],
                relation=edge["relation"],
                note=edge.get("note", ""),
            )

        self.clinical_rules: List[Dict[str, Any]] = self._raw.get("clinical_rules", [])
        log.info(
            "KG loaded: %d nodes, %d edges, %d rules",
            self.G.number_of_nodes(),
            self.G.number_of_edges(),
            len(self.clinical_rules),
        )

    def node_attrs(self, node_id: str) -> Dict[str, Any]:
        return dict(self.G.nodes.get(node_id, {}))

    def neighbors_by_relation(
        self,
        node_id: str,
        relation: Optional[str] = None,
        direction: str = "out",
    ) -> List[Dict[str, Any]]:
        if node_id not in self.G:
            return []
        if direction == "out":
            edges = self.G.out_edges(node_id, data=True)
            results = [
                {"node": t, "relation": d["relation"], "note": d.get("note", ""), "attrs": self.node_attrs(t)}
                for _, t, d in edges
                if relation is None or d["relation"] == relation
            ]
        else:
            edges = self.G.in_edges(node_id, data=True)
            results = [
                {"node": s, "relation": d["relation"], "note": d.get("note", ""), "attrs": self.node_attrs(s)}
                for s, _, d in edges
                if relation is None or d["relation"] == relation
            ]
        return results

    def complications_of(self, disease_id: str) -> List[str]:
        return [
            n["attrs"].get("name", n["node"])
            for n in self.neighbors_by_relation(disease_id, relation="causes")
            if n["attrs"].get("type") == "complication"
        ]

    def recommendations_for(self, node_id: str) -> List[str]:
        return [
            n["attrs"].get("name", n["node"])
            for n in self.neighbors_by_relation(node_id, relation="recommends")
        ]

    def drugs_for(self, node_id: str) -> List[str]:
        return [
            n["attrs"].get("name", n["node"])
            for n in self.neighbors_by_relation(node_id, relation="treats")
        ]

    def risk_factors_for(self, disease_id: str) -> List[str]:
        return [
            n["attrs"].get("name", n["node"])
            for n in self.neighbors_by_relation(disease_id, relation="increases_risk", direction="in")
        ]

    def dr_context(self, grade: int) -> Dict[str, Any]:
        grade_id = f"dr_grade{grade}"
        attrs = self.node_attrs(grade_id)
        return {
            "grade": grade,
            "name": attrs.get("name", f"Grade {grade}"),
            "referrable": attrs.get("referrable", grade >= 2),
            "monitoring_interval_months": attrs.get("monitoring_interval_months", 12),
            "recommendations": self.recommendations_for(grade_id),
        }

    def skin_context(self, label: str) -> Dict[str, Any]:
        label_map = {
            "Actinic keratosis": "actinic_keratosis",
            "Atopic Dermatitis": "atopic_dermatitis",
            "Benign keratosis": "benign_keratosis",
            "Chickenpox": "chickenpox",
            "Epidermolysis Bullosa": "epidermolysis_bullosa",
            "Melanoma": "melanoma",
            "Normal": "normal_skin",
            "Squamous cell carcinoma": "squamous_cell_carcinoma",
            "Tinea Ringworm Candidiasis": "tinea_candidiasis",
        }
        node_id = label_map.get(label)
        if not node_id:
            return {"label": label}
        attrs = self.node_attrs(node_id)
        return {
            "label": label,
            "malignant_potential": attrs.get("malignant_potential", "unknown"),
            "icd10": attrs.get("icd10", ""),
            "urgency": attrs.get("urgency", "routine"),
            "recommendations": self.recommendations_for(node_id),
        }

    def diabetes_context(self, is_positive: bool, glucose: Optional[float] = None) -> Dict[str, Any]:
        recs = self.recommendations_for("diabetes_t2") if is_positive else []
        complications = self.complications_of("diabetes_t2") if is_positive else []
        risk_factors = self.risk_factors_for("diabetes_t2")
        return {
            "positive": is_positive,
            "glucose_value": glucose,
            "recommendations": recs,
            "complications_to_watch": complications,
            "known_risk_factors": risk_factors,
        }

    def applicable_rules(self, finding_types: List[str], skin_label: Optional[str] = None, dr_grade: Optional[int] = None) -> List[Dict[str, Any]]:
        matched = []
        for rule in self.clinical_rules:
            trigger = rule.get("trigger", {})
            triggered = False

            if "skin_label" in trigger and skin_label in trigger["skin_label"]:
                triggered = True
            if "dr_grade" in trigger and dr_grade == trigger["dr_grade"]:
                triggered = True
            if "finding_type" in trigger and trigger["finding_type"] in finding_types:
                triggered = True

            if triggered:
                matched.append(rule)
        return matched

    def context_for_findings(
        self,
        finding_types: List[str],
        skin_label: Optional[str] = None,
        dr_grade: Optional[int] = None,
        diabetes_positive: Optional[bool] = None,
        glucose: Optional[float] = None,
        known_conditions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        ctx: Dict[str, Any] = {}

        if "diabetic_retinopathy" in finding_types and dr_grade is not None:
            ctx["dr"] = self.dr_context(dr_grade)

        if "skin_lesion" in finding_types and skin_label:
            ctx["skin"] = self.skin_context(skin_label)

        if "diabetes_risk" in finding_types and diabetes_positive is not None:
            ctx["diabetes"] = self.diabetes_context(diabetes_positive, glucose)


        conditions_lower = [c.lower() for c in (known_conditions or [])]
        if "hypertension" in conditions_lower:
            ctx["hypertension_note"] = (
                "Hypertension co-existing with diabetes significantly accelerates DR progression "
                "and increases cardiovascular risk. Target BP < 130/80 mmHg."
            )

        ctx["applicable_rules"] = self.applicable_rules(finding_types, skin_label, dr_grade)
        return ctx


_kg_singleton: Optional[MedicalKG] = None


def get_kg() -> MedicalKG:
    global _kg_singleton
    if _kg_singleton is None:
        _kg_singleton = MedicalKG()
    return _kg_singleton
