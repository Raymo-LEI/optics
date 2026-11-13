import math
import unittest
from copy import deepcopy

from optics_graph.contracts import (
    build_plan,
    canonical_hash,
    load_example,
    load_scene,
    validate_document,
    validate_layout,
    validate_plan,
)


def _codes(report):
    return {issue["code"] for issue in report["issues"]}


class LayoutContractTests(unittest.TestCase):
    def setUp(self):
        self.layout = load_example("layout")
        self.scene = load_scene()

    def assert_rejected(self, code, **kwargs):
        report = validate_layout(self.layout, **kwargs)
        self.assertFalse(report["passed"])
        self.assertIsNone(report["validation_id"])
        self.assertIn(code, _codes(report))
        return report

    def test_four_original_documents_and_replay_plan(self):
        for name in ("layout", "plan", "observation", "result"):
            with self.subTest(name=name):
                self.assertTrue(validate_document(load_example(name), name)["passed"])
                self.assertTrue(validate_document(load_example(name))["passed"])
        validation = validate_layout(self.layout)
        self.assertTrue(validation["passed"])
        self.assertEqual(validation["arm_lengths_m"], {"x": 0.2, "y": 0.2})
        self.assertEqual(validation["layout_sha256"], canonical_hash(self.layout))
        self.assertEqual(validate_layout(self.layout), validation)
        plan = build_plan(self.layout, validation)
        self.assertEqual(len(plan["actions"]), 14)
        self.assertTrue(validate_plan(plan, self.layout, validation)["passed"])
        self.assertFalse(validate_plan(load_example("plan"), self.layout, validation)["passed"])

    def test_rejects_schema_errors_and_non_finite_values(self):
        pose = self.layout["components"][1]["target_pose"]
        for bad_position in ([0.3, "0.2", 0], [0.3, float("nan"), 0], [0.3, 0.2]):
            with self.subTest(bad_position=bad_position):
                pose["position_m"] = bad_position
                self.assert_rejected("V_SCHEMA")

    def test_rejects_duplicate_objects_and_changed_marker(self):
        self.layout["components"][3] = deepcopy(self.layout["components"][2])
        self.assert_rejected("V_COMPONENT_SET")
        self.layout = load_example("layout")
        self.layout["components"][1]["marker_id"] = 10
        self.assert_rejected("V_COMPONENT_SET")

    def test_rejects_changed_fixed_pose_and_accepts_equivalent_quaternion(self):
        fixed = self.layout["components"][4]["target_pose"]
        fixed["orientation_xyzw"] = [-q for q in fixed["orientation_xyzw"]]
        self.assertTrue(validate_layout(self.layout)["passed"])
        fixed["position_m"][1] += 0.01
        self.assert_rejected("V_COMPONENT_SET")

    def test_rejects_invalid_quaternion_and_off_plane_pose(self):
        target = self.layout["components"][2]["target_pose"]
        target["orientation_xyzw"] = [0, 0, 0, 2]
        self.assert_rejected("V_SCHEMA")
        target["orientation_xyzw"] = [math.sqrt(0.5), 0, 0, math.sqrt(0.5)]
        self.assert_rejected("V_SCHEMA")
        target["orientation_xyzw"] = [0, 0, 0, 1]
        target["position_m"][2] = 0.01
        self.assert_rejected("V_SCHEMA")

    def test_rotated_corners_include_clearance(self):
        splitter = self.layout["components"][1]["target_pose"]
        splitter["position_m"][1] = 0.034  # Axis-aligned fit, but yaw=-45° requires ~0.0383 m.
        self.assert_rejected("V_WORKSPACE")

    def test_footprint_clearance_rejects_non_intersecting_but_close_boxes(self):
        self.layout["components"][2]["target_pose"]["position_m"][0] = 0.354
        report = self.assert_rejected("V_OVERLAP")
        issue = next(issue for issue in report["issues"] if issue["code"] == "V_OVERLAP")
        self.assertGreater(issue["actual"], 0)
        self.assertLess(issue["actual"], issue["expected"])

    def test_rejects_splitter_angle_that_breaks_a_return_path(self):
        self.layout["components"][1]["target_pose"]["orientation_xyzw"] = [0, 0, 0, 1]
        self.assert_rejected("V_TOPOLOGY")
        self.assert_rejected("V_RAY")

    def test_rejects_ray_blockage_and_non_planar_beam_height(self):
        self.scene["obstacles"].append({
            "obstacle_id": "example_occluder", "height_m": 0.07,
            "polygon_m": [[0.19, 0.19], [0.21, 0.19], [0.21, 0.21], [0.19, 0.21]],
        })
        report = self.assert_rejected("V_RAY", scene=self.scene)
        self.assertTrue(any(issue.get("occluder_id") == "example_occluder" for issue in report["issues"]))
        self.scene["obstacles"].clear()
        self.scene["objects"]["mirror_y_01"]["optical_center_height_m"] = 0.052
        self.assert_rejected("V_RAY", scene=self.scene)

    def test_explicit_request_requires_measurable_residuals(self):
        requirements = {"arm_lengths_m": {"x": 0.20, "y": 0.20}}
        self.assertTrue(validate_layout(self.layout, requirements=requirements)["passed"])
        requirements["arm_lengths_m"]["x"] = 0.28
        report = self.assert_rejected("V_REQUEST", requirements=requirements)
        self.assertEqual(report["issues"][0]["reason"], "UNSATISFIABLE_REQUEST")
        self.assertEqual(report["issues"][0]["actual"], 0.20)
        self.assert_rejected("V_REQUEST", requirements={"unsupported_constraint": True})
        self.assert_rejected("V_REQUEST", requirements={"max_position_error_m": 0.002})
        self.layout["tolerance"]["position_m"] = 0.002
        self.assertTrue(validate_layout(self.layout, requirements={"max_position_error_m": 0.002})["passed"])

    def test_missing_configuration_and_unknown_revision_fail(self):
        del self.scene["illustrative_reach_polygon_m"]
        self.assert_rejected("CONSTRAINT_CONFIG_MISSING", scene=self.scene)
        self.layout["scene_revision"] = "unknown-revision"
        self.assert_rejected("V_SCHEMA")


class PlanContractTests(unittest.TestCase):
    def setUp(self):
        self.layout = load_example("layout")
        self.validation = validate_layout(self.layout)
        self.plan = build_plan(self.layout, self.validation)

    def assert_rejected(self, code):
        result = validate_plan(self.plan, self.layout, self.validation)
        self.assertFalse(result["passed"])
        self.assertIn(code, _codes(result))

    def test_rejects_stale_hash_and_placeholder_receipt(self):
        self.layout["tolerance"]["position_m"] = 0.005
        self.assert_rejected("PLAN_VALIDATION_MISMATCH")
        self.layout = load_example("layout")
        self.validation["validation_id"] = "forged-validation"
        self.plan["validation_id"] = "forged-validation"
        self.assert_rejected("PLAN_VALIDATION_MISMATCH")

    def test_rejects_widened_tolerance_and_edited_pose(self):
        self.plan["actions"][3]["args"]["tolerance"]["position_m"] *= 2
        self.assert_rejected("PLAN_TOLERANCE")
        self.plan["actions"][4]["args"]["target_pose"]["position_m"][0] += 0.01
        self.assert_rejected("PLAN_TARGET")

    def test_rejects_observation_reference_outside_current_run_order(self):
        self.plan["actions"][2]["args"]["observation_ref"] = self.plan["actions"][5]["action_id"]
        self.assert_rejected("PLAN_OBSERVATION_REF")
        self.plan["actions"][2]["args"]["observation_ref"] = self.plan["actions"][1]["action_id"]
        self.plan["actions"][1]["args"]["object_ids"] = ["laser_01"]
        self.assert_rejected("PLAN_OBSERVATION_REF")

    def test_rejects_fixed_component_and_cycle(self):
        self.plan["actions"][2]["args"]["object_id"] = "laser_01"
        self.assert_rejected("PLAN_OBJECT")
        self.plan["actions"][1]["depends_on"] = [self.plan["actions"][2]["action_id"]]
        self.assert_rejected("PLAN_DEPENDENCY")

    def test_rejects_missing_verify_and_unknown_primitive(self):
        del self.plan["actions"][4]
        self.assert_rejected("PLAN_SEQUENCE")
        self.plan["actions"][1]["primitive"] = "joint_command"
        self.assert_rejected("V_SCHEMA")
