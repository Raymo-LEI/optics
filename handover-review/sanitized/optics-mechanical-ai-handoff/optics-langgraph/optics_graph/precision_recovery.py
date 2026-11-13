"""Allowed revisions for the registered contact-assembly workflow."""

from ast import literal_eval
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class PrecisionRepair(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["reacquire_views", "seat_first", "clearance_first", "extend_alignment_search",
                    "reseat_drifted_parts", "space_packed_carriers", "guard_camera_margin", "replan_packed_lift",
                    "refine_repair_seating", "reobserve_checkpoint", "replan_supported_turn_lift",
                    "replan_corridor_lift", "replan_visual_compensation", "stop"]
    diagnosis: str = Field(min_length=1, max_length=300)


VISION_AGENT = "precision_vision_agent"
MOTION_AGENT = "precision_motion_agent"
OPTICS_AGENT = "precision_optics_agent"
METROLOGY_AGENT = "precision_metrology_agent"
AGENT_NODES = (VISION_AGENT, MOTION_AGENT, OPTICS_AGENT, METROLOGY_AGENT)

# Evidence codes come from the simulator's terminal report. Never route an
# unbound artifact, a format error or an arbitrary error message as a repair.
FAILURES = {
    "IMAGE_REACQUISITION_BUDGET_EXHAUSTED": (VISION_AGENT, ("reacquire_views",)),
    "OBSERVATION_IK_REJECTED": (VISION_AGENT, ("reacquire_views",)),
    "COARSE_IMAGE_POSE": (VISION_AGENT, ("reacquire_views",)),
    "FINE_IMAGE_POSE": (VISION_AGENT, ("reacquire_views",)),
    "MULTIFACE_IMAGE_POSE": (VISION_AGENT, ("reacquire_views",)),
    "ROBOT_SELF_CONTACT": (MOTION_AGENT, ("clearance_first", "seat_first")),
    "ROBOT_JOINT_PATH_REJECTED": (MOTION_AGENT, ("clearance_first", "seat_first")),
    "HELD_TRANSFER_PATH_REJECTED": (MOTION_AGENT, ("seat_first", "clearance_first")),
    "MODULE_CONTACT_PENETRATION_GT_1MM": (MOTION_AGENT, ("seat_first", "clearance_first")),
    "VISUAL_REPLACEMENT_BUDGET_EXHAUSTED": (MOTION_AGENT, ("seat_first",)),
    "FINAL_VISUAL_LAYOUT_CHECK_FAILED": (MOTION_AGENT, ("reseat_drifted_parts",)),
    "PACKED_CARRIER_LIFT_CONTACT": (MOTION_AGENT, ("space_packed_carriers",)),
    "COARSE_OPTICAL_ARMS": (OPTICS_AGENT, ("extend_alignment_search",)),
    "CAM_RECEIVER_ALIGNMENT": (OPTICS_AGENT, ("extend_alignment_search",)),
    "FIRST_FRINGE_FEEDBACK": (OPTICS_AGENT, ("extend_alignment_search",)),
    "PD_RECEIVER_ALIGNMENT": (OPTICS_AGENT, ("extend_alignment_search",)),
    "FINAL_FRINGE_FEEDBACK": (OPTICS_AGENT, ("extend_alignment_search",)),
}

ACTION_DESCRIPTIONS = {
    "reacquire_views": "Acquire more calibration views and keep the original image-stability threshold",
    "seat_first": "Infer the arm pose from the placement first, and use a slower lowering speed",
    "clearance_first": "Prefer a joint trajectory with a collision-avoiding intermediate pose",
    "extend_alignment_search": "Increase the optical-feedback search count and keep the original acceptance threshold",
    "reseat_drifted_parts": "Image again and pick-and-place every component that failed the full-bench recheck, then recheck the whole bench",
    "space_packed_carriers": "Replan the order and lowering clearance of neighboring bases, and assemble from one end within the original tolerance",
    "guard_camera_margin": "For components that fail independent mechanical review, tighten the camera-triggered replay margin and keep the original acceptance threshold",
}


def repair_evidence(result, previous_actions):
    if (result.get("bindings_verified") is not True
            or result.get("added_sensor_noise") != {"rgb_std_8bit": 0., "optical_std": 0.}):
        return None, None
    verification = result.get("verification") or {}
    code = "FINAL_BODY_POSE_OUTSIDE_1MM_1DEG"
    if (verification.get("failure_reasons") == [code]
            and all(verification.get(key, {}).get("passed") is True
                    for key in ("sequence", "bindings", "optical", "sensor_feedback"))
            and "guard_camera_margin" not in previous_actions):
        bodies = verification.get("mechanical", {}).get("objects", {})
        affected = sorted(name for name, body in bodies.items() if body.get("passed") is False)
        if affected:
            return METROLOGY_AGENT, {"failure_code":code, "allowed_actions":["guard_camera_margin"],
                                     "actions":{"guard_camera_margin":ACTION_DESCRIPTIONS["guard_camera_margin"]},
                                     "guard_parts":affected}
    failure = result.get("failure")
    if not isinstance(failure, str):
        return None, None
    code, _, details = failure.removeprefix("RuntimeError: ").partition(":")
    # Older workers stored the structured contact record as a Python tuple.
    if code == "TRANSPORT" and len(details) < 240 and "reseat_drifted_parts" in previous_actions:
        try:
            time_s, phase, contact = literal_eval(details)
        except (ValueError, SyntaxError, TypeError):
            pass
        else:
            if type(time_s) in (float, int) and isinstance(phase, str) and phase.endswith(": lift") and contact == "carried module/other module":
                code = "PACKED_CARRIER_LIFT_CONTACT"
    agent, actions = FAILURES.get(code, (None, ()))
    allowed = [action for action in actions if action not in previous_actions]
    if not allowed:
        return None, None
    return agent, {"failure_code": code,
                   "allowed_actions": allowed,
                   "actions": {action: ACTION_DESCRIPTIONS[action] for action in allowed}}
