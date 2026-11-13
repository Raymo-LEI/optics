"""Experiment data for the synthetic backend, independent of robot fixture slots.

Free text is designed by the node Agent. This module packages validated targets
and keeps explicit developer fixtures separate; it never invents a grid layout.
"""
from copy import deepcopy
import math
import re
import unicodedata

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .experiments import SIMPLE, catalog, load_experiment, match_registered_assembly
from .sim_io import decode_json
from .team_artifacts import digest, encode

MAX_COMPONENTS = 128  # Bounds local evidence size, not a hardware inventory limit.
# Experiment-contract source of a design that came from a model-style decider, by run provider. The plan-time
# binding check in algorithm_sim applies to every value in DESIGN_SOURCES.
DESIGN_SOURCE_BY_PROVIDER = {"openai": "openai_design", "agent": "agent_design"}
DESIGN_SOURCES = frozenset(DESIGN_SOURCE_BY_PROVIDER.values())


class Component(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    object_id: str = Field(pattern=r"^[A-Za-z0-9_\u3400-\u9fff.-]{1,64}$")
    label: str = Field(default="", max_length=120)
    position_m: list[float] = Field(min_length=3, max_length=3)
    yaw_rad: float = 0.
    movable: bool = True


class ExperimentDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    title: str = Field(min_length=1, max_length=200)
    components: list[Component] = Field(min_length=1, max_length=MAX_COMPONENTS)

    @model_validator(mode="after")
    def unique_ids(self):
        if len({c.object_id for c in self.components}) != len(self.components):
            raise ValueError("Experiment component IDs must be unique")
        return self


def reference_for(request):
    """References are optional defaults, never an experiment admission whitelist."""
    options = catalog()
    selected = match_registered_assembly(request, options)
    if selected:
        return selected
    text = unicodedata.normalize("NFKC", request).strip()
    # Relax only plain assembly phrasing. Additional dimensions/requirements must
    # go through the generic contract and remain visible rather than be discarded.
    plain = re.sub(
        r"^(?:please|help me|i want to|on the optical table|assemble|build|run|test|simulate|a|an|one|,|\s)+",
        "",
        text,
        count=1,
        flags=re.I,
    )
    plain = re.sub(r"[.!\s]+$", "", plain)
    for key, item in options.items():
        if key == SIMPLE:
            continue
        for alias in sorted(item["aliases"], key=len, reverse=True):
            if re.fullmatch(re.escape(alias) + r"\s*(?:interferometer)?", plain, re.I):
                return key
    # An explicit five-part Michelson is a different reference, not fifteen parts.
    if re.search(r"\b5\s*-?\s*components?\b", text, re.I) and re.search(r"\bMichelson\b", text, re.I):
        simplified = re.sub(r"\b5\s*-?\s*components?\b\s*(?:of\s+)?", "", text, flags=re.I)
        if reference_for(simplified) == "michelson_full":
            return SIMPLE
    return None


def build_experiment(request):
    if not request.strip():
        raise ValueError("The experiment request cannot be empty")
    assumptions = ["The bench is an assumed assembled state. One movable object starts with an offset so in-node correction and rollback can be checked."]
    selected = None
    if request.lstrip().startswith("{"):
        definition = ExperimentDefinition.model_validate(decode_json(request)).model_dump()
        title, components = definition["title"], definition["components"]
        source = "user_definition"
        assumptions.append("Target poses come from the user definition. Only the format and algorithm flow are checked; the optical design is not treated as valid.")
    else:
        selected = reference_for(request)
        if selected:
            scene, layout = load_experiment(selected)
            title = catalog()[selected]["title"]
            components = [{"object_id":c["object_id"], "label":c["object_id"],
                           "position_m":c["target_pose"]["position_m"],
                           "yaw_rad":2*math.atan2(c["target_pose"]["orientation_xyzw"][2],
                                                c["target_pose"]["orientation_xyzw"][3]),
                           "movable":c["movable"]} for c in layout["components"]]
            source = "registered_reference"
            assumptions.append(f"This round reuses reference layout {title} ({len(components)} components). When no count is specified, that count is an explicit default assumption.")
        else:
            raise ValueError("Free-text experiments must be interpreted and designed by the node agent. The offline fixture does not invent a placeholder layout")
    ExperimentDefinition.model_validate({"title":title, "components":components})
    spec = {"schema_version":"algorithm-experiment/v1", "title":title, "request":request,
            "source":source, "reference_id":selected, "components":components,
            "component_count":len(components), "assumptions":assumptions,
            "pending_requirements":["Topology, component parameters, physical constraints, and performance targets have not been verified by a real design or simulation."],
            "design_verified":False, "request_sha256":digest(request.encode())}
    spec["contract_sha256"] = digest(encode(spec))
    if not selected:
        revision = "algorithm-" + spec["contract_sha256"][:16]
        layout = {"layout_id":revision, "scene_revision":revision,
                  "constraint_revision":"algorithm-proxy-v1", "tolerance":{"position_m":.001,"angle_rad":math.radians(.5)},
                  "components":[{"object_id":c["object_id"], "movable":c["movable"],
                     "target_pose":{"frame_id":"optical_table", "position_m":deepcopy(c["position_m"]),
                                    "orientation_xyzw":[0.,0.,math.sin(c["yaw_rad"]/2),math.cos(c["yaw_rad"]/2)]}}
                                for c in components]}
        scene = {"scene_revision":revision, "constraint_revision":"algorithm-proxy-v1",
                 "objects":{c["object_id"]:{"movable":c["movable"],"label":c["label"]} for c in components}}
    return selected or "algorithm_custom", scene, layout, spec


def verify_experiment(spec, request):
    payload = {k:v for k,v in spec.items() if k != "contract_sha256"}
    if (digest(encode(payload)) != spec.get("contract_sha256")
            or spec.get("request") != request or spec.get("request_sha256") != digest(request.encode())
            or spec.get("design_verified") is not False):
        raise ValueError("The experiment definition, original request, or verification scope has changed")
    ExperimentDefinition.model_validate({"title":spec["title"], "components":[
        {k:v for k,v in c.items() if k in Component.model_fields} for c in spec["components"]]})


def package_design(request, intent, design, provider="openai"):
    """Package an actual Agent candidate without altering its chosen geometry."""
    minimal = {"title":design["title"], "components":[
        {k:v for k,v in c.items() if k in Component.model_fields} for c in design["components"]]}
    _, scene, layout, _ = build_experiment(encode(minimal).decode())
    spec = {"schema_version":"algorithm-experiment/v1", "title":design["title"], "request":request,
            "source":DESIGN_SOURCE_BY_PROVIDER.get(provider, "openai_design"), "reference_id":None, "components":deepcopy(design["components"]),
            "component_count":len(design["components"]), "rationale":design["rationale"],
            "connections":deepcopy(design["connections"]),
            "requirements_coverage":deepcopy(design["requirements_coverage"]),
            "assumptions":[*intent["assumptions"], *design["assumptions"],
                           "Actuation and observation are algorithmic simulations. The target design comes from the Act agent. The initial loading area is not a design target."],
            "pending_requirements":[*design["unresolved_requirements"], "Real mechanical execution and the physical optical path still need independent simulation."],
            "execution_limits":{"position_m":intent["position_tolerance_m"] or .001,
                                "angle_rad":intent["angle_tolerance_rad"] or math.radians(.5)},
            "design_verified":False, "request_sha256":digest(request.encode()),
            "intent_sha256":digest(encode(intent)), "design_sha256":digest(encode(design))}
    spec["contract_sha256"] = digest(encode(spec))
    revision = "agent-" + spec["contract_sha256"][:16]
    scene["scene_revision"] = layout["scene_revision"] = revision
    layout["layout_id"] = revision
    layout["tolerance"] = deepcopy(spec["execution_limits"])
    return scene, layout, spec


def verify_target(spec, layout):
    expected = {c["object_id"]:c for c in spec["components"]}
    actual = layout["components"]
    if len(actual) != len(expected) or {c["object_id"] for c in actual} != set(expected):
        raise ValueError("The target layout's component set does not match this experiment definition")
    for item in actual:
        c, pose = expected[item["object_id"]], item["target_pose"]
        angle = 2*math.atan2(pose["orientation_xyzw"][2], pose["orientation_xyzw"][3])
        if (item["movable"] != c["movable"] or pose["position_m"] != c["position_m"]
                or abs(math.atan2(math.sin(angle-c["yaw_rad"]), math.cos(angle-c["yaw_rad"]))) > 1e-12):
            raise ValueError("A target pose or movable range does not match the experiment definition")
